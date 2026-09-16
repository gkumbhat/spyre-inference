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

"""Probes for token-level pooling (``AllPool``) on Spyre.

``TOKEN_POOLING_TASKS`` in ``spyre_inference/v1/pool/spyre_pooler.py`` drops
``token_embed`` / ``token_classify`` while the pooler runs on Spyre, on the
grounds that ``AllPool`` hands out ``torch.split`` views of ``hidden_states``.
These probes test that claim primitive by primitive, then against the real
upstream ``TokenPooler``, so a verdict does not depend on a full model run.

Run in this order and read the first failure:

    uv run pytest -m "not upstream" tests/test_spyre_token_pooling_probes.py -v \
        -W "error::torch_spyre.ops.fallbacks.FallbackWarning"

Unlike ``test_spyre_fallback_probes.py`` these are plain asserts, not strict
xfail: the outcome is what is being measured. Once the verdict is known, the
ones that fail should move to ``test_spyre_fallback_probes.py`` as strict
xfail with a torch-spyre issue number, and the ones that pass become guards.

A pass in a full-file run is provisional. The slice-copy bug these probes chase
only fires once a differently-sized slice has already run in the process, so an
earlier probe arms later ones and a green result can mean "not yet armed". Before
believing any pass, re-run that node ID alone in a fresh process.

Shapes mirror ``dslim/bert-base-NER``: hidden 768, 9 labels, a ragged batch of
short sequences packed on dim 0.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.pooler.activations import PoolerClassify, PoolerNormalize
from vllm.model_executor.layers.pooler.tokwise.heads import (
    TokenClassifierPoolerHead,
    TokenEmbeddingPoolerHead,
)
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.pool.spyre_pooler import (
    SpyreNormalize,
    copy_pooler_output_to_cpu,
    patch_normalize_for_spyre,
    select_rows,
)
from spyre_testing_plugin.pytest_plugin import spyre_available

HIDDEN = 768
NUM_LABELS = 9

# Ragged and not stick-aligned on purpose: every chunk after the first starts at
# a storage offset that is not a multiple of the 256-token KV alignment, which is
# the property the TOKEN_POOLING_TASKS comment is worried about.
SEQ_LENS = [7, 13, 4, 20]

# A Spyre fp16 H2D->D2H round trip is NOT bit-exact: payload comes back 1-2 fp16
# ULP away from zero (measured max|diff| 2**-8 == 0.003906 over [-3.3, 3.3]),
# while integers survive exactly. The device stick format's dtype is evidently
# not quite IEEE fp16 (see get_device_dtype in slot_major_kv_layout). So
# `atol=0` compares a tensor against something the hardware cannot return, and
# every such assertion fails for reasons unrelated to what it means to test.
# Use ROUND_TRIP_RTOL for payload, and the exact integer fingerprint in column 0
# for addressing -- a wrong-rows bug substitutes different random values and
# shows up as an O(1) relative error, orders of magnitude above this noise.
ROUND_TRIP_RTOL = 4e-3
ROUND_TRIP_ATOL = 1e-3

# fp16 elements per 128-byte stick; the modulus in torch-spyre#3798's guard.
ELEMS_PER_STICK = 64


@pytest.fixture()
def spyre_device():
    if not spyre_available():
        pytest.skip("Spyre device not available")
    return torch.device("spyre")


def _packed_hidden_states(seq_lens: list[int]) -> torch.Tensor:
    """``[sum(seq_lens), HIDDEN]`` fp16 with a per-row fingerprint.

    Row ``i`` carries ``i`` in column 0 on top of the noise, so a chunk read from
    the wrong storage offset shows up as an off-by-N row index rather than as a
    tolerance miss.
    """
    total = sum(seq_lens)
    hs = torch.randn(total, HIDDEN, dtype=torch.float16)
    hs[:, 0] = torch.arange(total, dtype=torch.float16)
    return hs


def _pooling_metadata(seq_lens: list[int], task: str) -> PoolingMetadata:
    """A finished-prefill ``PoolingMetadata`` with a CPU pooling cursor.

    Matches what ``TorchSpyreModelRunner._pool`` builds: the cursor lives on CPU
    because upstream's ``cumsum[1:] - 1`` view is not stick-aligned on Spyre.
    """
    lens = torch.tensor(seq_lens, dtype=torch.int64)
    metadata = PoolingMetadata(
        prompt_lens=lens,
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=[PoolingParams(task=task, use_activation=True) for _ in seq_lens],
        pooling_states=[PoolingStates() for _ in seq_lens],
    )
    metadata.build_pooling_cursor(
        np.array(seq_lens, dtype=np.int32),
        seq_lens_cpu=lens,
        device=torch.device("cpu"),
    )
    return metadata


# ---------------------------------------------------------------------------
# 1. The split itself — AllPool.forward
# ---------------------------------------------------------------------------


def test_allpool_split_d2h(spyre_device):
    """``torch.split`` on dim 0 then D2H each chunk, exactly as AllPool does.

    This is the primitive TOKEN_POOLING_TASKS was closed over. Chunk ``k>0`` is a
    contiguous view at ``storage_offset = sum(seq_lens[:k]) * HIDDEN``. HIDDEN is a
    stick multiple, so every such offset is divisible by 64 and clears
    torch-spyre#3798's host-offset guard — unlike the NUM_LABELS-wide case in
    ``test_token_classify_softmax_of_split_view``. torch-spyre#3770 showed such
    offsets being read as 0 elsewhere, so the values still need checking.
    """
    hs_cpu = _packed_hidden_states(SEQ_LENS)
    hs = convert(hs_cpu, spyre_device)

    chunks = list(torch.split(hs, SEQ_LENS))
    assert [c.shape[0] for c in chunks] == SEQ_LENS

    offset = 0
    for k, (chunk, n) in enumerate(zip(chunks, SEQ_LENS)):
        got = convert(chunk, "cpu")
        expected = hs_cpu[offset : offset + n]
        assert got.shape == expected.shape, f"chunk {k}: {got.shape} != {expected.shape}"
        first_row = int(got[0, 0].item())
        assert first_row == offset, (
            f"chunk {k} starts at row {first_row}, expected {offset} — "
            "the view's storage offset was not honored"
        )
        torch.testing.assert_close(got, expected, atol=ROUND_TRIP_ATOL, rtol=ROUND_TRIP_RTOL)
        offset += n


def test_allpool_split_d2h_after_different_split(spyre_device):
    """Same, but a differently-sized split runs first in the same process.

    torch-spyre#3826's extent bug only fires from the second distinct
    length onward, so arm the probe rather than depend on test ordering. #3909
    fixed that for slice *writes*; this is the read direction.
    """
    warm_lens = [3, 29, 8]
    warm_cpu = _packed_hidden_states(warm_lens)
    for chunk in torch.split(convert(warm_cpu, spyre_device), warm_lens):
        convert(chunk, "cpu")

    hs_cpu = _packed_hidden_states(SEQ_LENS)
    hs = convert(hs_cpu, spyre_device)

    offset = 0
    for k, (chunk, n) in enumerate(zip(torch.split(hs, SEQ_LENS), SEQ_LENS)):
        got = convert(chunk, "cpu")
        torch.testing.assert_close(
            got,
            hs_cpu[offset : offset + n],
            atol=ROUND_TRIP_ATOL,
            rtol=ROUND_TRIP_RTOL,
            msg=f"chunk {k} mismatch",
        )
        offset += n


def test_allpool_split_clone_d2h(spyre_device):
    """``chunk.clone()`` before D2H — the ``clone_finished`` / workaround path.

    ``.clone()`` is the documented workaround for torch-spyre#3826 on the write
    side. If ``test_allpool_split_d2h`` fails and this passes, a clone in a Spyre
    ``AllPool`` subclass is enough to enable token pooling on device.
    """
    hs_cpu = _packed_hidden_states(SEQ_LENS)
    hs = convert(hs_cpu, spyre_device)

    offset = 0
    for k, (chunk, n) in enumerate(zip(torch.split(hs, SEQ_LENS), SEQ_LENS)):
        got = convert(chunk.clone(), "cpu")
        torch.testing.assert_close(
            got,
            hs_cpu[offset : offset + n],
            atol=ROUND_TRIP_ATOL,
            rtol=ROUND_TRIP_RTOL,
            msg=f"chunk {k} mismatch",
        )
        offset += n


def test_allpool_split_concat(spyre_device):
    """``torch.concat`` of two split views — AllPool's chunked-prefill reassembly.

    Only reached when ``enable_chunked_prefill`` is on, but the model runner does
    not currently forbid that for pooling, so it is part of the surface.
    """
    hs_cpu = _packed_hidden_states(SEQ_LENS)
    hs = convert(hs_cpu, spyre_device)
    chunks = list(torch.split(hs, SEQ_LENS))

    joined = torch.concat([chunks[1], chunks[2]], dim=0)
    expected = torch.concat(
        [
            hs_cpu[SEQ_LENS[0] : SEQ_LENS[0] + SEQ_LENS[1]],
            hs_cpu[SEQ_LENS[0] + SEQ_LENS[1] : SEQ_LENS[0] + SEQ_LENS[1] + SEQ_LENS[2]],
        ],
        dim=0,
    )
    torch.testing.assert_close(
        convert(joined, "cpu"), expected, atol=ROUND_TRIP_ATOL, rtol=ROUND_TRIP_RTOL
    )


def test_allpool_index_select_chunks_d2h(spyre_device):
    """Per-request rows via ``select_rows`` instead of ``torch.split``.

    ``SpyreCLSPool`` / ``SpyreLastPool`` already gather with host-built indices to
    dodge this class of bug; this asks whether the same trick carries the ragged
    per-request gather ``AllPool`` needs. If the split probes fail and this
    passes, a ``SpyreAllPool`` built on ``select_rows`` is the whole fix.

    The warm-up arms the differently-sized-slice condition that made the split
    probes fail at chunk 0.
    """
    warm_lens = [3, 29, 8]
    warm = convert(_packed_hidden_states(warm_lens), spyre_device)
    for chunk in torch.split(warm, warm_lens):
        convert(chunk, "cpu")

    hs_cpu = _packed_hidden_states(SEQ_LENS)
    hs = convert(hs_cpu, spyre_device)

    offset = 0
    for k, n in enumerate(SEQ_LENS):
        rows = select_rows(hs, torch.arange(offset, offset + n, dtype=torch.int64))
        got = convert(rows, "cpu")
        assert got.shape == (n, HIDDEN), f"chunk {k}: {got.shape}"
        first_row = int(got[0, 0].item())
        assert first_row == offset, f"chunk {k} starts at row {first_row}, expected {offset}"
        torch.testing.assert_close(
            got,
            hs_cpu[offset : offset + n],
            atol=ROUND_TRIP_ATOL,
            rtol=ROUND_TRIP_RTOL,
            msg=f"chunk {k} mismatch",
        )
        offset += n


# ---------------------------------------------------------------------------
# 2. What the token heads do to each chunk
# ---------------------------------------------------------------------------


def test_token_head_fp32_cast_of_split_view(spyre_device):
    """``pooled_data.to(float32)`` on a split view — both token heads open with this.

    ``head_dtype`` defaults to float32 for the pooling runner, and
    ``SpyreEmbeddingPoolerHead`` exists precisely because an on-device fp16->fp32
    cast after a gather corrupts values. Expect this to be the first real
    blocker; the fix is a token-head analog that casts on CPU, or
    ``--hf-overrides '{"head_dtype": "model"}'``.
    """
    hs_cpu = _packed_hidden_states(SEQ_LENS)
    hs = convert(hs_cpu, spyre_device)

    offset = 0
    for k, (chunk, n) in enumerate(zip(torch.split(hs, SEQ_LENS), SEQ_LENS)):
        got = chunk.to(torch.float32)
        expected = hs_cpu[offset : offset + n].to(torch.float32)
        torch.testing.assert_close(
            got.cpu(), expected, atol=1e-3, rtol=1e-3, msg=f"chunk {k} mismatch"
        )
        offset += n


def test_token_embed_normalize_of_split_view(spyre_device):
    """``SpyreNormalize`` (rsqrt L2) on a split view — token_embed activation."""
    hs_cpu = _packed_hidden_states(SEQ_LENS)
    hs = convert(hs_cpu, spyre_device)
    norm = SpyreNormalize()
    ref = PoolerNormalize()

    offset = 0
    for k, (chunk, n) in enumerate(zip(torch.split(hs, SEQ_LENS), SEQ_LENS)):
        got = convert(norm(chunk), "cpu").float()
        expected = ref(hs_cpu[offset : offset + n].float())
        torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2, msg=f"chunk {k} mismatch")
        offset += n


def test_token_classify_softmax_of_split_view(spyre_device):
    """``PoolerClassify`` softmax over a 9-wide last dim — token_classify activation.

    ``NUM_LABELS`` is not a multiple of the 64-element stick, so each row is padded
    to a whole stick on device while host offsets stay multiples of 9.
    torch-spyre#3798: ``_validate_reoffset_supported`` rejects a copy when the
    *host* ``storage_offset % elems_per_stick != 0``, and PR #3578 routes every
    eager op on an offset view through that copy — so chunk 0 (offset 0) clears the
    guard and chunk 1 (offset 63) does not. Nothing to do with reductions: ``x + x``
    fails identically. See scripts/repro_softmax_sub_stick_view.py.
    """
    logits_cpu = torch.randn(sum(SEQ_LENS), NUM_LABELS, dtype=torch.float16)
    logits = convert(logits_cpu, spyre_device)
    act = PoolerClassify()

    offset = 0
    for k, (chunk, n) in enumerate(zip(torch.split(logits, SEQ_LENS), SEQ_LENS)):
        got = convert(act(chunk), "cpu").float()
        expected = act(logits_cpu[offset : offset + n].float())
        torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2, msg=f"chunk {k} mismatch")
        offset += n


def test_token_classify_softmax_of_whole_buffer(spyre_device):
    """``PoolerClassify`` softmax over the whole flat ``[T, NUM_LABELS]`` buffer.

    Distinguishes "softmax is broken on Spyre" from "torch-spyre#3798's host-offset
    guard rejected the view" — ``test_token_classify_softmax_of_split_view`` only
    shows the latter. A whole tensor has ``storage_offset == 0``, so it always
    clears the guard. Softmax over the last dim is row-independent, so applying it to
    the flat buffer before the partition is bit-identical to upstream's
    per-chunk application, and costs one op on one bucketable shape instead of B
    ragged ones. If this passes, the activation can move on-device without
    reintroducing raggedness.
    """
    logits_cpu = torch.randn(sum(SEQ_LENS), NUM_LABELS, dtype=torch.float16)
    act = PoolerClassify()

    got = convert(act(convert(logits_cpu, spyre_device)), "cpu").float()

    expected = act(logits_cpu.float())
    torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(got.sum(-1), torch.ones(got.shape[0]), atol=1e-2, rtol=1e-2)


def test_fp16_round_trip_is_not_bit_exact(spyre_device):
    """Document the round trip, since it decides what every other probe may assert.

    Integers survive exactly; fractions come back 1-2 fp16 ULP away from zero. So
    ``atol=0`` compares against something the hardware cannot return, and the
    integer fingerprint in column 0 is the only exact signal available -- which is
    what makes it usable for checking row addressing.
    """
    x_cpu = _packed_hidden_states(SEQ_LENS)
    got = convert(convert(x_cpu, spyre_device), "cpu")

    assert torch.equal(got[:, 0], x_cpu[:, 0]), "integer fingerprint column must be exact"
    torch.testing.assert_close(got, x_cpu, atol=ROUND_TRIP_ATOL, rtol=ROUND_TRIP_RTOL)


_XFAIL_3798 = pytest.mark.xfail(
    strict=True,
    reason=(
        "torch-spyre#3798: _validate_reoffset_supported rejects a device copy when "
        "the *host* storage_offset is not a multiple of elems_per_stick (64 for "
        "fp16). A row narrower than one stick is padded to a whole stick on device, "
        "so every row boundary IS stick-aligned there, but the host offset is a "
        "multiple of the unpadded width and the guard cannot tell the two apart. "
        "PR #3578 routes every eager op on an offset view through that copy, so "
        "eager mode fails for all of them. This is what keeps AllPool off the "
        "device for token_classify, whose width is num_labels (9 for "
        "dslim/bert-base-NER): only the offset-0 chunk clears the guard. When this "
        "XPASSes, an on-device SpyreAllPool becomes possible -- though a host "
        "partition is still cheaper, since ragged per-request lengths recompile."
    ),
)


@pytest.mark.parametrize(
    ("width", "start"),
    [
        # width is a stick multiple -> start * width always divisible by 64.
        (768, 1),
        (768, 7),
        (64, 1),
        # gcd(9, 64) == 1, so only a start that is itself a multiple of 64 aligns.
        (9, 64),
        pytest.param(9, 1, marks=_XFAIL_3798),
        pytest.param(9, 7, marks=_XFAIL_3798),
        pytest.param(40, 1, marks=_XFAIL_3798),
    ],
)
def test_eager_op_on_offset_view(spyre_device, width, start):
    """An eager op on a dim-0 view, across torch-spyre#3798's alignment boundary.

    This is the pooling requirement in one line: ``AllPool`` gives each request a
    ``torch.split`` view at host offset ``start * width``, and every op applied to
    it must work. ``x + x`` is deliberately the simplest op available -- the guard
    is in the copy that materializes the view, not in any particular kernel, so
    softmax and elementwise addition fail identically.

    The parametrization is the evidence: at width 9, ``x[64:]`` works and ``x[1:]``
    does not, though both row boundaries are equally stick-aligned on device.
    ``scripts/repro_softmax_sub_stick_view.py`` is the standalone form of this.
    """
    rows = start + max(SEQ_LENS)
    x_cpu = torch.randn(rows, width, dtype=torch.float16)
    x = convert(x_cpu, spyre_device)

    view = x[start:]
    assert view.storage_offset() == start * width
    aligned = view.storage_offset() % ELEMS_PER_STICK == 0
    assert aligned == (start * width % ELEMS_PER_STICK == 0)

    got = convert(view + view, "cpu")

    expected = x_cpu[start:] + x_cpu[start:]
    torch.testing.assert_close(
        got,
        expected,
        atol=ROUND_TRIP_ATOL,
        rtol=ROUND_TRIP_RTOL,
        msg=f"width={width} start={start} offset={start * width} aligned={aligned}",
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_token_classify_classifier_matmul(spyre_device, dtype):
    """``nn.Linear(768, 9)`` on device — ``BertForTokenClassification.classifier``.

    That classifier lives in the *model* forward, not the pooler, and is built at
    ``head_dtype`` (float32 by default). float32 is expected to fail — Spyre has
    no fp32 batch matmul, which is what ``configure_pooling_for_spyre`` already
    keys off for rerankers. If float16 passes, forcing ``head_dtype`` to the
    model dtype is the cheaper route than a CPU classifier.
    """
    x_cpu = torch.randn(sum(SEQ_LENS), HIDDEN, dtype=dtype)
    layer = nn.Linear(HIDDEN, NUM_LABELS, dtype=dtype)
    weight, bias = layer.weight.detach().float(), layer.bias.detach().float()

    out = layer.to(spyre_device)(convert(x_cpu, spyre_device))

    expected = nn.functional.linear(x_cpu.float(), weight, bias)
    torch.testing.assert_close(convert(out, "cpu").float(), expected, atol=1e-1, rtol=1e-1)


# ---------------------------------------------------------------------------
# 3. The real upstream TokenPooler, Spyre vs CPU
# ---------------------------------------------------------------------------


def _run_token_pooler(pooler: TokenPooler, hs: torch.Tensor, task: str):
    metadata = _pooling_metadata(SEQ_LENS, task)
    raw = pooler(hidden_states=hs, pooling_metadata=metadata)
    return copy_pooler_output_to_cpu(raw, [True] * len(SEQ_LENS))


def _index_select_pooling(
    hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
) -> list[torch.Tensor]:
    """``AllPool.forward`` with ``select_rows`` in place of ``torch.split``.

    A stand-in for the ``SpyreAllPool`` that would ship if the split views turn
    out to be the only blocker. Chunked prefill is out of scope here: every
    request in ``_pooling_metadata`` is a finished single-chunk prefill.
    """
    counts = pooling_metadata.get_pooling_cursor().num_scheduled_tokens_cpu.tolist()
    chunks = []
    offset = 0
    for n in counts:
        chunks.append(
            select_rows(hidden_states, torch.arange(offset, offset + n, dtype=torch.int64))
        )
        offset += n
    return chunks


def _pooling_method(impl: str):
    return AllPool() if impl == "split" else _index_select_pooling


@pytest.mark.parametrize("impl", ["split", "index_select"])
@pytest.mark.parametrize("head_dtype", [None, torch.float32])
def test_token_embed_pooler_matches_cpu(spyre_device, head_dtype, impl):
    """AllPool + TokenEmbeddingPoolerHead on Spyre vs the same graph on CPU.

    ``head_dtype=None`` isolates the pooling path from the fp32 cast; ``float32``
    is what the pooling runner actually configures. ``impl`` A/Bs upstream's
    ``torch.split`` against the ``select_rows`` gather.
    """
    hs_cpu = _packed_hidden_states(SEQ_LENS)

    def build() -> TokenPooler:
        return TokenPooler(
            pooling=_pooling_method(impl),
            head=TokenEmbeddingPoolerHead(
                head_dtype=head_dtype, projector=None, activation=PoolerNormalize()
            ),
        )

    reference = _run_token_pooler(build(), hs_cpu, "token_embed")

    on_spyre = build()
    assert patch_normalize_for_spyre(on_spyre) == 1, "normalize head was not patched"
    got = _run_token_pooler(on_spyre, convert(hs_cpu, spyre_device), "token_embed")

    assert len(got) == len(reference) == len(SEQ_LENS)
    for k, (g, r) in enumerate(zip(got, reference)):
        assert g is not None and r is not None
        assert g.shape == r.shape == (SEQ_LENS[k], HIDDEN), f"request {k}: {g.shape}"
        torch.testing.assert_close(
            g.float(), r.float(), atol=1e-2, rtol=1e-2, msg=f"request {k} mismatch"
        )


@pytest.mark.parametrize("impl", ["split", "index_select"])
@pytest.mark.parametrize("head_dtype", [None, torch.float16])
def test_token_classify_pooler_matches_cpu(spyre_device, head_dtype, impl):
    """AllPool + TokenClassifierPoolerHead (fp16 classifier) on Spyre vs CPU.

    float32 is deliberately not parametrized here — see
    ``test_token_classify_classifier_matmul``. This measures whether the rest of
    the token_classify graph works once the dtype problem is taken off the table.
    """
    hs_cpu = _packed_hidden_states(SEQ_LENS)
    classifier_cpu = nn.Linear(HIDDEN, NUM_LABELS, dtype=torch.float16)

    def build(classifier: nn.Module) -> TokenPooler:
        return TokenPooler(
            pooling=_pooling_method(impl),
            head=TokenClassifierPoolerHead(
                classifier=classifier,
                logit_mean=None,
                logit_sigma=None,
                head_dtype=head_dtype,
                activation=PoolerClassify(),
            ),
        )

    reference = _run_token_pooler(build(classifier_cpu), hs_cpu, "token_classify")

    on_spyre_classifier = copy.deepcopy(classifier_cpu).to(spyre_device)
    got = _run_token_pooler(
        build(on_spyre_classifier), convert(hs_cpu, spyre_device), "token_classify"
    )

    assert len(got) == len(reference) == len(SEQ_LENS)
    for k, (g, r) in enumerate(zip(got, reference)):
        assert g is not None and r is not None
        assert g.shape == r.shape == (SEQ_LENS[k], NUM_LABELS), f"request {k}: {g.shape}"
        torch.testing.assert_close(
            g.float(), r.float(), atol=1e-2, rtol=1e-2, msg=f"request {k} mismatch"
        )

        # A flipped argmax only indicts the device if the reference was not
        # near-tied: round-trip noise (see ROUND_TRIP_RTOL) is enough to swap two
        # labels whose scores are within a couple of ULP, whereas a wrong-rows or
        # wrong-extent bug substitutes unrelated values and flips decisive rows.
        flipped = (g.argmax(-1) != r.argmax(-1)).nonzero().flatten().tolist()
        if flipped:
            top2 = r.float().topk(2, dim=-1).values
            margins = (top2[:, 0] - top2[:, 1])[flipped]
            decisive = [
                (int(row), round(float(m), 5))
                for row, m in zip(flipped, margins.tolist())
                if m > 1e-2
            ]
            assert not decisive, (
                f"request {k}: labels differ on rows the reference was decisive "
                f"about (row, margin): {decisive}"
            )
            print(
                f"request {k}: {len(flipped)} label flip(s), all near-ties "
                f"(max margin {margins.max().item():.2e}) — round-trip noise"
            )
