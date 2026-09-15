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

"""Encoder attention warmup coverage: ``_warm_block_fns`` compiles every block-kernel
variant a body bucket could reach, so nothing compiles mid-request.

CPU-only, and not about numerics -- ``test_spyre_encoder_attn.py`` covers those. Under
the flash design there is only one shape axis left (per-request block count), driven
entirely by the body bucket a request lands on: ``_warm_block_fns`` is called once per
distinct ``buffer_rows`` seen in ``forward()``, which happens automatically because
warmup's own dummy runs pass ``force_attention=True`` and visit every body bucket.
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


class TestWarmBlockFnsCoversTheBucket:
    """``_warm_block_fns`` must exercise every ``(num_blocks, needs_gather)`` a
    request landing on this ``buffer_rows`` body bucket could actually reach."""

    def test_warms_every_power_of_two_block_count_up_to_the_buffer(self, monkeypatch):
        buffer_rows = 256
        seen: set[tuple[int, bool]] = set()
        impl = _make_impl()
        real_run_block = impl._run_block

        def counting_run_block(query, key, value, row_index, mask_tiles, num_blocks, *rest):
            needs_gather = rest[3]
            seen.add((num_blocks, needs_gather))
            return real_run_block(query, key, value, row_index, mask_tiles, num_blocks, *rest)

        monkeypatch.setattr(impl, "_run_block", counting_run_block)
        impl._warm_block_fns(
            buffer_rows,
            impl.num_heads,
            impl.num_kv_heads,
            impl.head_size,
            impl.model_dtype,
            torch.device("cpu"),
        )

        expected_block_counts = set()
        num_blocks = 1
        while num_blocks * ENCODER_BLOCK_SIZE <= buffer_rows:
            expected_block_counts.add(num_blocks)
            num_blocks *= 2
        # Every block count gathers; only the one that fills the buffer exactly
        # additionally skips the gather (torch-spyre#4033).
        assert seen == {(n, True) for n in expected_block_counts} | {
            (buffer_rows // ENCODER_BLOCK_SIZE, False)
        }

    def test_is_idempotent_per_buffer_size(self, monkeypatch):
        """A second call at the same ``buffer_rows`` must not re-warm anything."""
        impl = _make_impl()
        calls = {"n": 0}
        real_run_block = impl._run_block

        def counting_run_block(*args, **kwargs):
            calls["n"] += 1
            return real_run_block(*args, **kwargs)

        monkeypatch.setattr(impl, "_run_block", counting_run_block)
        args = (impl.num_heads, impl.num_kv_heads, impl.head_size, impl.model_dtype, torch.device("cpu"))

        impl._warm_block_fns(128, *args)
        first = calls["n"]
        assert first > 0
        impl._warm_block_fns(128, *args)
        assert calls["n"] == first

    def test_a_real_sequence_at_any_length_lands_on_a_warmed_variant(self):
        """Every length a body bucket can hold maps onto a ``(num_blocks, needs_gather)``
        pair ``_warm_block_fns`` visited -- checked against ``_blocks_for`` directly,
        since that is what ``_build_plans`` uses to pick a request's variant."""
        buffer_rows = 512
        warmed_block_counts = set()
        num_blocks = 1
        while num_blocks * ENCODER_BLOCK_SIZE <= buffer_rows:
            warmed_block_counts.add(num_blocks)
            num_blocks *= 2

        checked = 0
        for length in range(1, buffer_rows + 1, 7):
            assert _blocks_for(length) in warmed_block_counts
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
