# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Encoder attention warmup coverage: ``_warm_kernels`` compiles every gather/attend
variant a body bucket could reach, so nothing compiles mid-request.

CPU-only, and not about numerics -- ``test_spyre_encoder_attn.py`` covers those.
``_warm_kernels`` is called once per distinct ``buffer_rows`` seen in ``forward()``,
which happens automatically because warmup's own dummy runs pass
``force_attention=True`` and visit every body bucket. Gather is cheap and keyed on
``(buffer_rows, extent)``, so it is rewarmed at every body bucket; attention/store are
keyed only on ``extent`` (the sequence's own padded length) and are warmed exactly once
per distinct extent, however many buffer_rows values reach it.
"""

import torch
from torch._dynamo.utils import counters

from spyre_inference.v1.attention.backends import spyre_attn
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    ENCODER_BLOCK_SIZE,
    SpyreEncoderAttentionImpl,
    _blocks_for,
)


def _make_impl(num_heads=4, num_kv_heads=1, head_size=64):
    return SpyreEncoderAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
    )


class TestWarmKernelsCoversTheBucket:
    """``_warm_kernels`` must exercise gather for every extent this ``buffer_rows``
    body bucket can reach, and attention/store exactly once per distinct extent --
    regardless of how many different buffer_rows values reach it."""

    def test_gathers_every_power_of_two_extent_up_to_the_buffer(self, monkeypatch):
        buffer_rows = 256
        seen_gathers: set[int] = set()
        impl = _make_impl()
        real_run_gather = impl._run_gather

        def counting_run_gather(query, key, value, row_index):
            seen_gathers.add(row_index.shape[0])
            return real_run_gather(query, key, value, row_index)

        monkeypatch.setattr(impl, "_run_gather", counting_run_gather)
        impl._warm_kernels(
            buffer_rows,
            impl.num_heads,
            impl.num_kv_heads,
            impl.head_size,
            impl.model_dtype,
            torch.device("cpu"),
        )

        expected_extents = set()
        extent = ENCODER_BLOCK_SIZE
        while extent <= buffer_rows:
            expected_extents.add(extent)
            extent *= 2
        assert seen_gathers == expected_extents

    def test_attn_and_store_compile_once_per_extent_not_per_buffer(self, monkeypatch):
        """The expensive kernel must not be rewarmed for an extent already seen at
        a smaller buffer_rows -- that is the whole point of splitting gather/store
        (cheap, keyed on buffer_rows) from attention (expensive, keyed only on the
        sequence's own padded length).
        """
        impl = _make_impl()
        calls = {"n": 0}
        real_run_attn = impl._run_attn

        def counting_run_attn(*args, **kwargs):
            calls["n"] += 1
            return real_run_attn(*args, **kwargs)

        monkeypatch.setattr(impl, "_run_attn", counting_run_attn)
        args = (impl.num_heads, impl.num_kv_heads, impl.head_size, impl.model_dtype, torch.device("cpu"))

        impl._warm_kernels(64, *args)
        first = calls["n"]
        assert first == 1  # exactly one extent (64) reachable at buffer_rows=64

        impl._warm_kernels(128, *args)
        # Only the new extent (128) should trigger a fresh compile; extent 64
        # was already warmed by the buffer_rows=64 call above.
        assert calls["n"] == first + 1

    def test_is_idempotent_per_buffer_size(self, monkeypatch):
        """A second call at the same ``buffer_rows`` must not re-warm anything."""
        impl = _make_impl()
        calls = {"n": 0}
        real_run_gather = impl._run_gather

        def counting_run_gather(*args, **kwargs):
            calls["n"] += 1
            return real_run_gather(*args, **kwargs)

        monkeypatch.setattr(impl, "_run_gather", counting_run_gather)
        args = (impl.num_heads, impl.num_kv_heads, impl.head_size, impl.model_dtype, torch.device("cpu"))

        impl._warm_kernels(128, *args)
        first = calls["n"]
        assert first > 0
        impl._warm_kernels(128, *args)
        assert calls["n"] == first

    def test_a_real_sequence_at_any_length_lands_on_a_warmed_extent(self):
        """Every length a body bucket can hold maps onto an extent ``_warm_kernels``
        visited -- checked against ``_blocks_for`` directly, since that is what
        ``_build_plans`` uses to pick a request's extent."""
        buffer_rows = 512
        warmed_extents = set()
        extent = ENCODER_BLOCK_SIZE
        while extent <= buffer_rows:
            warmed_extents.add(extent)
            extent *= 2

        checked = 0
        for length in range(1, buffer_rows + 1, 7):
            assert _blocks_for(length) * ENCODER_BLOCK_SIZE in warmed_extents
            checked += 1
        assert checked > 20, "too few lengths checked -- test is near-vacuous"


class TestWarmupComplete:
    """``mark_warmup_complete`` gates the late-compile diagnostic, not correctness."""

    def test_flag_round_trips(self):
        saved = spyre_attn._warmup_complete
        try:
            spyre_attn._warmup_complete = False
            assert not spyre_attn.is_warmup_complete()
            spyre_attn.mark_warmup_complete()
            assert spyre_attn.is_warmup_complete()
        finally:
            spyre_attn._warmup_complete = saved


def test_call_kernel_warns_only_after_warmup_is_marked_complete(monkeypatch, caplog):
    """``_call_kernel`` must stay silent until warmup claims coverage, mirroring how
    a real server calls ``mark_warmup_complete()`` only once startup finishes.

    ``warning_once`` dedups globally on the message, so the label must be unique to
    this test run or an earlier test's warning would mask a real regression here.
    """
    saved = spyre_attn._warmup_complete
    label = f"test kernel {id(test_call_kernel_warns_only_after_warmup_is_marked_complete)}"

    # Simulates "a compile happened during this call" without a real Inductor
    # compile: _call_kernel only compares the counter before/after its own call.
    def fn(x):
        counters["stats"]["unique_graphs"] += 1
        return x + 1

    try:
        spyre_attn._warmup_complete = False

        with caplog.at_level("WARNING"):
            spyre_attn._call_kernel(label, fn, 1)
        assert "compiled outside warmup" not in caplog.text

        spyre_attn.mark_warmup_complete()
        with caplog.at_level("WARNING"):
            spyre_attn._call_kernel(label, fn, 1)
        assert "compiled outside warmup" in caplog.text
    finally:
        spyre_attn._warmup_complete = saved
