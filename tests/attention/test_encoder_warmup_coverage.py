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
``force_attention=True`` and visit every body bucket. Gather and store are cheap and
keyed on ``buffer_rows`` (gather also on ``extent``), so both are rewarmed at every
body bucket; attention is keyed only on ``extent`` (the sequence's own padded length)
and is warmed exactly once per distinct extent, however many buffer_rows values reach
it -- calling it again on an already-seen extent is just a cache hit.
"""

import torch
from torch._dynamo.utils import counters

from spyre_inference.v1.attention.backends import spyre_attn
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    ENCODER_LEN_ALIGNMENT,
    SpyreEncoderAttentionImpl,
    _alignment_units_for,
)


def _dummy_qkv(buffer_rows, num_heads, num_kv_heads, head_size, dtype, device):
    query = torch.zeros((buffer_rows, num_heads, head_size), dtype=dtype, device=device)
    key = torch.zeros((buffer_rows, num_kv_heads, head_size), dtype=dtype, device=device)
    value = torch.zeros((buffer_rows, num_kv_heads, head_size), dtype=dtype, device=device)
    return query, key, value


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
    """``_warm_kernels`` must exercise gather and store for every ``(buffer_rows,
    extent)`` pair this body bucket can reach, using the caller's own query/key/
    value layout -- not a substitute dummy that could warm a different Dynamo
    specialization than what a real request actually presents."""

    def test_gathers_every_power_of_two_extent_up_to_the_buffer(self, monkeypatch):
        buffer_rows = 256
        seen_gathers: set[int] = set()
        impl = _make_impl()
        real_run = impl._run_fused

        def counting_run(out, row_index, *args, **kwargs):
            seen_gathers.add(row_index.shape[0])
            return real_run(out, row_index, *args, **kwargs)

        # The gather is inside the fused graph; its row table is arg 2.
        monkeypatch.setattr(impl, "_run_fused", counting_run)
        query, key, value = _dummy_qkv(
            buffer_rows,
            impl.num_heads,
            impl.num_kv_heads,
            impl.head_size,
            impl.model_dtype,
            torch.device("cpu"),
        )
        impl._warm_kernels(
            query,
            key,
            value,
            torch.zeros_like(query),
            impl.num_heads,
            impl.num_kv_heads,
            impl.head_size,
        )

        expected_extents = set()
        extent = ENCODER_LEN_ALIGNMENT
        while extent <= buffer_rows:
            expected_extents.add(extent)
            extent *= 2
        assert seen_gathers == expected_extents

    def test_store_rewarms_every_buffer_size_not_just_the_first_to_reach_an_extent(
        self, monkeypatch
    ):
        """Regression: store's own cache key includes ``buffer_rows`` (unlike
        attention's, which is keyed only on ``extent``), so it must be called for
        every ``(buffer_rows, extent)`` pair -- not skipped just because some
        *other*, larger buffer_rows already reached that extent. Warmup sweeps
        largest-first, so a bug that gated store on "extent already seen" would
        only ever warm the largest bucket's variant, leaving every smaller
        bucket's store uncompiled until a real request hit it.
        """
        impl = _make_impl()
        seen_out_rows: set[int] = set()
        real_run = impl._run_fused

        def counting_run(out, *args, **kwargs):
            seen_out_rows.add(out.shape[0])
            return real_run(out, *args, **kwargs)

        # The store is fused into the attention call, so hook that.
        monkeypatch.setattr(impl, "_run_fused", counting_run)

        for buffer_rows in (256, 128, 64):  # largest first, like real warmup
            query, key, value = _dummy_qkv(
                buffer_rows,
                impl.num_heads,
                impl.num_kv_heads,
                impl.head_size,
                impl.model_dtype,
                torch.device("cpu"),
            )
            impl._warm_kernels(
                query,
                key,
                value,
                torch.zeros_like(query),
                impl.num_heads,
                impl.num_kv_heads,
                impl.head_size,
            )

        assert seen_out_rows == {64, 128, 256}

    def test_store_is_warmed_against_the_caller_s_own_output_tensor(self, monkeypatch):
        """Regression: the store's destination layout is part of its cache key too.

        vLLM hands ``forward`` an output built as ``torch.empty(rows, H*D).view(
        -1, H, D)``; warmup used to build its own ``torch.zeros((rows, H, D))``
        destination instead. Same shape, different Spyre device layout, so the
        graph warmup compiled was not the one real traffic could reuse and the
        store recompiled on first use -- observed on hardware for three
        (buffer_rows, extent) pairs warmup demonstrably did sweep. Warm against
        the caller's real output tensor, exactly as gather does for query.
        """
        impl = _make_impl()
        buffer_rows = 128
        query, key, value = _dummy_qkv(
            buffer_rows,
            impl.num_heads,
            impl.num_kv_heads,
            impl.head_size,
            impl.model_dtype,
            torch.device("cpu"),
        )
        # Built the way vLLM builds it: a view of a 2-D allocation.
        output = torch.zeros(
            (buffer_rows, impl.num_heads * impl.head_size), dtype=impl.model_dtype
        ).view(-1, impl.num_heads, impl.head_size)

        seen_out = []
        real_run = impl._run_fused

        def recording_run(out, *args, **kwargs):
            seen_out.append(out)
            return real_run(out, *args, **kwargs)

        # The store is fused into the attention call, so hook that.
        monkeypatch.setattr(impl, "_run_fused", recording_run)
        impl._warm_kernels(
            query,
            key,
            value,
            output,
            impl.num_heads,
            impl.num_kv_heads,
            impl.head_size,
        )

        assert seen_out, "warmup must exercise the store"
        assert all(o is output for o in seen_out), (
            "store must be warmed against the caller's output, not a substitute"
        )

    def test_gather_is_warmed_against_the_caller_s_own_query_layout(self, monkeypatch):
        """Regression: a fused-QKV projection (``qkv.split(...)``) hands out a
        strided view, and that stride is part of the compiled gather function's
        cache key under ``dynamic=False``. ``_warm_kernels`` must gather from the
        exact tensors it was called with, not a freshly built contiguous dummy --
        otherwise it warms a different specialization than a real strided
        request needs, recompiling on that request's first arrival.
        """
        impl = _make_impl(num_heads=4, num_kv_heads=1, head_size=64)
        buffer_rows = 64
        fused = torch.zeros((buffer_rows, 4 * 64 + 64 + 64), dtype=impl.model_dtype)
        q_flat, k_flat, v_flat = fused.split([4 * 64, 64, 64], dim=-1)
        query = q_flat.view(buffer_rows, 4, 64)
        key = k_flat.view(buffer_rows, 1, 64)
        value = v_flat.view(buffer_rows, 1, 64)
        assert not query.is_contiguous(), "test setup must exercise a genuinely strided view"

        seen_strides: list[tuple] = []
        real_run = impl._run_fused

        def recording_run(out, row_index, q, k, v, *args, **kwargs):
            seen_strides.append(q.stride())
            return real_run(out, row_index, q, k, v, *args, **kwargs)

        # The gather is inside the fused graph, which takes the buffers directly.
        monkeypatch.setattr(impl, "_run_fused", recording_run)
        impl._warm_kernels(
            query,
            key,
            value,
            torch.zeros_like(query),
            impl.num_heads,
            impl.num_kv_heads,
            impl.head_size,
        )

        assert seen_strides and all(s == query.stride() for s in seen_strides)

    def test_is_idempotent_per_buffer_size(self, monkeypatch):
        """A second call at the same ``buffer_rows`` must not re-warm anything."""
        impl = _make_impl()
        calls = {"n": 0}
        real_run = impl._run_fused

        def counting_run(*args, **kwargs):
            calls["n"] += 1
            return real_run(*args, **kwargs)

        monkeypatch.setattr(impl, "_run_fused", counting_run)
        query, key, value = _dummy_qkv(
            128,
            impl.num_heads,
            impl.num_kv_heads,
            impl.head_size,
            impl.model_dtype,
            torch.device("cpu"),
        )

        impl._warm_kernels(
            query,
            key,
            value,
            torch.zeros_like(query),
            impl.num_heads,
            impl.num_kv_heads,
            impl.head_size,
        )
        first = calls["n"]
        assert first > 0
        impl._warm_kernels(
            query,
            key,
            value,
            torch.zeros_like(query),
            impl.num_heads,
            impl.num_kv_heads,
            impl.head_size,
        )
        assert calls["n"] == first

    def test_a_real_sequence_at_any_length_lands_on_a_warmed_extent(self):
        """Every length a body bucket can hold maps onto an extent ``_warm_kernels``
        visited -- checked against ``_alignment_units_for`` directly, since that is what
        ``_build_plans`` uses to pick a request's extent."""
        buffer_rows = 512
        warmed_extents = set()
        extent = ENCODER_LEN_ALIGNMENT
        while extent <= buffer_rows:
            warmed_extents.add(extent)
            extent *= 2

        checked = 0
        for length in range(1, buffer_rows + 1, 7):
            assert _alignment_units_for(length) * ENCODER_LEN_ALIGNMENT in warmed_extents
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


def test_warmup_stops_at_the_longest_reachable_extent(monkeypatch):
    """A request's extent cannot exceed its own length, so warming past the model
    length compiles the most expensive graphs for shapes nothing can reach.

    With max_model_len=512 and a 2048-row body bucket, the sweep used to build
    extents 1024 and 2048 too -- on hardware that was most of the grouping
    warmup cost (680s -> 224s once capped).
    """
    impl = _make_impl()
    monkeypatch.setattr(impl, "_max_extent", 128)
    # Grouping gathers group*extent rows, which legitimately exceeds the extent;
    # this test is about the extent axis, so measure it on its own.
    monkeypatch.setattr(impl, "_batched_attn", False)
    seen: set[int] = set()
    real_run = impl._run_fused

    def counting_run(out, row_index, *args, **kwargs):
        seen.add(row_index.shape[0])
        return real_run(out, row_index, *args, **kwargs)

    monkeypatch.setattr(impl, "_run_fused", counting_run)
    query, key, value = _dummy_qkv(
        512,
        impl.num_heads,
        impl.num_kv_heads,
        impl.head_size,
        impl.model_dtype,
        torch.device("cpu"),
    )
    impl._warm_kernels(
        query,
        key,
        value,
        torch.zeros_like(query),
        impl.num_heads,
        impl.num_kv_heads,
        impl.head_size,
    )

    assert seen, "warmup must still exercise the reachable extents"
    assert max(seen) <= 128, f"warmed unreachable extents: {sorted(seen)}"


def test_warmup_stops_at_max_num_seqs_groups(monkeypatch):
    """A group holds one member per request, so it cannot exceed ``max_num_seqs``.

    The sweep bounds the group axis by ``buffer_rows // extent``, which at a small
    batch size is far larger: with max_num_seqs=4 and a 2048-row bucket it built
    groups up to 32, a fifth of the warm variants unreachable by any plan.
    """
    impl = _make_impl()
    monkeypatch.setattr(impl, "_max_group", 4)
    monkeypatch.setattr(impl, "_max_extent", ENCODER_LEN_ALIGNMENT)
    buffer_rows = 2048
    seen: set[int] = set()
    real_run = impl._run_fused

    def counting_run(out, row_index, *args, **kwargs):
        # row_index holds group * extent rows; extent is pinned to one unit above.
        seen.add(row_index.shape[0] // ENCODER_LEN_ALIGNMENT)
        return real_run(out, row_index, *args, **kwargs)

    monkeypatch.setattr(impl, "_run_fused", counting_run)
    query, key, value = _dummy_qkv(
        buffer_rows,
        impl.num_heads,
        impl.num_kv_heads,
        impl.head_size,
        impl.model_dtype,
        torch.device("cpu"),
    )
    impl._warm_kernels(
        query,
        key,
        value,
        torch.zeros_like(query),
        impl.num_heads,
        impl.num_kv_heads,
        impl.head_size,
    )

    assert seen, "warmup must still exercise the reachable groups"
    assert max(seen) <= 4, f"warmed groups no plan can reach: {sorted(seen)}"
