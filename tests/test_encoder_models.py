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

"""Spyre product embed tests vs cached HF refs and reranker smoke tests.

Regenerate embed refs: ``python tests/data/generate_encoder_embed_refs.py``
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from vllm import LLM
from vllm.config import PoolerConfig

EMBEDDING_MODELS = [
    "ibm-granite/granite-embedding-125m-english",
    "ibm-granite/granite-embedding-278m-multilingual",
    "intfloat/multilingual-e5-large",
    "sentence-transformers/all-roberta-large-v1",
]

# None of the product encoder models above ship with LAST pooling (CLS or MEAN).
# Force LAST on a small CLS model so SpyreLastPool is covered end-to-end.
LAST_POOLING_MODEL = "ibm-granite/granite-embedding-125m-english"
LAST_POOLING_PROMPTS = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
]

# Cross-encoder reranker smoke (classify / score path). One model is enough —
# both BGE variants share XLMRobertaForSequenceClassification.
RERANKER_MODELS = [
    "BAAI/bge-reranker-v2-m3",
]

# Token classification (token_classify / AllPool path). One vector per *token*
# rather than per request, so the pooler runs on the host: upstream ``AllPool``
# hands out ``torch.split`` views whose host storage_offset is a multiple of
# num_labels, and torch-spyre#3798 rejects a device copy of an offset view unless
# that offset is a multiple of the 64-element stick. Both architectures apply
# ``self.classifier`` inside ``forward``, so it stays on Spyre in float16 (see
# TorchSpyrePlatform._force_fp16_head_for_token_classification).
TOKEN_CLASSIFY_MODELS = [
    "dslim/bert-base-NER",
    "Jean-Baptiste/roberta-large-ner-english",
]
TOKEN_CLASSIFY_PROMPTS = [
    "My name is Wolfgang and I live in Berlin",
    "George Washington went to Washington",
]
# Per-token argmax must match HF on this fraction of tokens. Not 1.0: the head
# runs in float16 here against HF float32, so a near-tied token can flip.
LABEL_AGREEMENT_MIN = 0.95

# Match upstream check_embeddings_close(tol=1e-2).
COSINE_MIN = 0.99

_REF_PATH = Path(__file__).parent / "data" / "encoder_embed_refs.json"
_REFERENCES: dict = json.loads(_REF_PATH.read_text()) if _REF_PATH.exists() else {}


def _cosine(a: list[float], b: list[float]) -> float:
    return F.cosine_similarity(
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32),
        dim=0,
    ).item()


def _hf_last_token_embeddings(model: str, prompts: list[str]) -> list[list[float]]:
    """CPU HF last-nonpad-token + L2 (matches vLLM LastPool + normalize)."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    hf = AutoModel.from_pretrained(model)
    hf.eval()
    with torch.inference_mode():
        enc = tok(
            prompts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        hs = hf(**enc).last_hidden_state
        idx = enc["attention_mask"].sum(dim=1) - 1
        emb = hs[torch.arange(hs.size(0)), idx]
        emb = F.normalize(emb.float(), p=2, dim=-1)
    return emb.tolist()


def _hf_token_labels(model: str, prompts: list[str]) -> list[list[int]]:
    """CPU HF per-token argmax labels (matches vLLM AllPool + classifier + softmax)."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    hf = AutoModelForTokenClassification.from_pretrained(model)
    hf.eval()

    labels = []
    with torch.inference_mode():
        for prompt in prompts:
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64)
            logits = hf(**enc).logits[0]
            labels.append(logits.float().argmax(dim=-1).tolist())
    return labels


@pytest.mark.uses_subprocess
@pytest.mark.parametrize("model", EMBEDDING_MODELS)
def test_encoder_embed_models(model: str) -> None:
    """Spyre embeddings match cached HF references within cosine tolerance."""
    ref = _REFERENCES.get(model)
    if ref is None:
        pytest.skip(f"No HF ref for {model}; run tests/data/generate_encoder_embed_refs.py")

    prompts = ref["prompts"]
    llm = LLM(
        model=model,
        runner="pooling",
        max_model_len=64,
        max_num_seqs=1,
        enforce_eager=True,
    )
    outputs = llm.embed(prompts)
    assert len(outputs) == len(prompts)

    for prompt, out, ref_emb in zip(prompts, outputs, ref["embeddings"]):
        emb = out.outputs.embedding
        assert len(emb) == len(ref_emb), (
            f"{model}: dim mismatch {len(emb)} vs cached {len(ref_emb)}"
        )
        assert all(math.isfinite(x) for x in emb)
        sim = _cosine(emb, ref_emb)
        assert sim >= COSINE_MIN, (
            f"{model}: cosine {sim:.4f} < {COSINE_MIN} vs cached HF reference for prompt {prompt!r}"
        )


@pytest.mark.uses_subprocess
def test_encoder_embed_last_pooling() -> None:
    """SpyreLastPool path: force LAST on granite-125m and match HF last-token.

    Product encoder models in ``EMBEDDING_MODELS`` are CLS or MEAN only; this
    override exercises the LAST gather + normalize path that
    ``configure_pooling_for_spyre`` patches to ``SpyreLastPool``.
    """
    prompts = LAST_POOLING_PROMPTS
    ref_embs = _hf_last_token_embeddings(LAST_POOLING_MODEL, prompts)

    llm = LLM(
        model=LAST_POOLING_MODEL,
        runner="pooling",
        max_model_len=64,
        max_num_seqs=1,
        enforce_eager=True,
        pooler_config=PoolerConfig(seq_pooling_type="LAST"),
    )
    outputs = llm.embed(prompts)
    assert len(outputs) == len(prompts)

    for prompt, out, ref_emb in zip(prompts, outputs, ref_embs):
        emb = out.outputs.embedding
        assert len(emb) == len(ref_emb), (
            f"LAST {LAST_POOLING_MODEL}: dim mismatch {len(emb)} vs HF {len(ref_emb)}"
        )
        assert all(math.isfinite(x) for x in emb)
        sim = _cosine(emb, ref_emb)
        assert sim >= COSINE_MIN, (
            f"LAST {LAST_POOLING_MODEL}: cosine {sim:.4f} < {COSINE_MIN} "
            f"vs HF last-token for prompt {prompt!r}"
        )


@pytest.mark.uses_subprocess
@pytest.mark.parametrize("model", RERANKER_MODELS)
def test_encoder_rerank_models(model: str) -> None:
    """Load reranker and return one finite score via LLM.score()."""
    llm = LLM(
        model=model,
        runner="pooling",
        max_model_len=64,
        max_num_seqs=1,
        enforce_eager=True,
    )
    scores = llm.score("What is Spyre?", "An IBM AI accelerator.")
    assert len(scores) == 1
    assert math.isfinite(scores[0].outputs.score)


@pytest.mark.uses_subprocess
@pytest.mark.parametrize("model", TOKEN_CLASSIFY_MODELS)
def test_encoder_token_classify_models(model: str) -> None:
    """Per-token labels match HF, with the pooler on the host.

    Covers the whole token_classify path end to end: the encoder and the float16
    classifier run on Spyre, the ``[num_tokens, num_labels]`` logits come back in
    one D2H, and upstream ``AllPool`` partitions them per request on the host.

    Asserts shape as well as labels — one row per prompt token — because a partition
    that silently drops or duplicates rows is the failure mode this path is exposed
    to, and argmax agreement alone would not catch a shifted split.
    """
    ref_labels = _hf_token_labels(model, TOKEN_CLASSIFY_PROMPTS)

    llm = LLM(
        model=model,
        runner="pooling",
        max_model_len=64,
        max_num_seqs=len(TOKEN_CLASSIFY_PROMPTS),
        enforce_eager=True,
        pooler_config=PoolerConfig(task="token_classify"),
    )
    outputs = llm.encode(TOKEN_CLASSIFY_PROMPTS, pooling_task="token_classify")
    assert len(outputs) == len(TOKEN_CLASSIFY_PROMPTS)

    for prompt, out, ref in zip(TOKEN_CLASSIFY_PROMPTS, outputs, ref_labels):
        data = torch.as_tensor(out.outputs.data).float()
        assert data.ndim == 2, f"{model}: expected [num_tokens, num_labels], got {data.shape}"
        assert data.shape[0] == len(ref), (
            f"{model}: {data.shape[0]} token rows vs HF {len(ref)} for prompt {prompt!r} — "
            "the per-request partition dropped or duplicated rows"
        )
        assert torch.isfinite(data).all(), f"{model}: non-finite logits for prompt {prompt!r}"

        got = data.argmax(dim=-1).tolist()
        agreement = sum(g == r for g, r in zip(got, ref)) / len(ref)
        assert agreement >= LABEL_AGREEMENT_MIN, (
            f"{model}: per-token label agreement {agreement:.3f} < {LABEL_AGREEMENT_MIN} "
            f"vs HF for prompt {prompt!r}\n  spyre {got}\n  hf    {ref}"
        )
