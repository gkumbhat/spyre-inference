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

"""Cheap unit tests for ``configure_pooling_for_spyre`` patching.

No Spyre hardware: builds minimal ``SequencePooler`` / ``DispatchPooler`` /
``TokenPooler`` graphs and checks CLS/LAST become ``SpyreCLSPool`` /
``SpyreLastPool`` while MEAN and FP32 heads stay on the CPU fallback path.

Also covers which classifier ``configure_pooling_for_spyre`` is allowed to move.
Sequence-classification models pass ``self.classifier`` into the pooler head, so a
CPU pooler needs it on CPU; ``*ForTokenClassification`` models apply it inside
``forward`` on Spyre activations, where moving it would strand a CPU weight.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.model_executor.layers.pooler.activations import PoolerClassify, PoolerNormalize
from vllm.model_executor.layers.pooler.seqwise.heads import (
    ClassifierPoolerHead,
    EmbeddingPoolerHead,
)
from vllm.model_executor.layers.pooler.seqwise.methods import CLSPool, LastPool, MeanPool
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.model_executor.layers.pooler.special import DispatchPooler
from vllm.model_executor.layers.pooler.tokwise.heads import TokenClassifierPoolerHead
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler

from spyre_inference.v1.pool.spyre_pooler import (
    SpyreCLSPool,
    SpyreEmbeddingPoolerHead,
    SpyreLastPool,
    SpyreNormalize,
    configure_pooling_for_spyre,
    pooler_owned_classifier,
)

_SPYRE = torch.device("cpu")  # configure only needs a device label for logging


def _embed_pooler(pooling) -> SequencePooler:
    return SequencePooler(
        pooling=pooling,
        head=EmbeddingPoolerHead(activation=PoolerNormalize()),
    )


def _model_with_pooler(pooler: nn.Module) -> nn.Module:
    model = nn.Module()
    model.pooler = pooler
    return model


def test_configure_pooling_patches_cls_to_spyre_cls_pool():
    model = _model_with_pooler(_embed_pooler(CLSPool()))
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    assert isinstance(model.pooler.pooling, SpyreCLSPool)
    assert isinstance(model.pooler.head, SpyreEmbeddingPoolerHead)
    assert isinstance(model.pooler.head.activation, SpyreNormalize)


def test_configure_pooling_patches_last_to_spyre_last_pool():
    model = _model_with_pooler(_embed_pooler(LastPool()))
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    assert isinstance(model.pooler.pooling, SpyreLastPool)
    assert isinstance(model.pooler.head, SpyreEmbeddingPoolerHead)


def test_configure_pooling_mean_falls_back_to_cpu():
    model = _model_with_pooler(_embed_pooler(MeanPool()))
    assert configure_pooling_for_spyre(model, _SPYRE) is False
    # MEAN is unsupported (#3507); leave the upstream method in place on CPU.
    assert isinstance(model.pooler.pooling, MeanPool)
    assert not isinstance(model.pooler.pooling, (SpyreCLSPool, SpyreLastPool))


def test_configure_pooling_dispatch_patches_embed_cls():
    """DispatchPooler (real embed models) must still install SpyreCLSPool."""
    pooler = DispatchPooler({"embed": _embed_pooler(CLSPool())})
    model = _model_with_pooler(pooler)
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    embed = model.pooler.poolers_by_task["embed"]
    assert isinstance(embed.pooling, SpyreCLSPool)


def test_configure_pooling_dispatch_patches_embed_last():
    pooler = DispatchPooler({"embed": _embed_pooler(LastPool())})
    model = _model_with_pooler(pooler)
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    embed = model.pooler.poolers_by_task["embed"]
    assert isinstance(embed.pooling, SpyreLastPool)


class _RecordingLinear(nn.Linear):
    """Records direct ``.to()`` calls so a CPU-only host can tell what moved."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_calls: list = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


def _seq_cls_model(dtype: torch.dtype = torch.float32) -> nn.Module:
    """Reranker shape: the classifier is handed to the pooler head (bert.py:805)."""
    classifier = _RecordingLinear(8, 2, dtype=dtype)
    model = _model_with_pooler(
        SequencePooler(
            pooling=CLSPool(),
            head=ClassifierPoolerHead(classifier=classifier, activation=PoolerClassify()),
        )
    )
    model.classifier = classifier
    return model


def _token_classify_model(dtype: torch.dtype = torch.float32) -> nn.Module:
    """NER shape: the head has no classifier; ``forward`` applies it (bert.py:862)."""
    model = _model_with_pooler(
        TokenPooler(
            pooling=AllPool(),
            head=TokenClassifierPoolerHead(classifier=None, activation=PoolerClassify()),
        )
    )
    model.classifier = _RecordingLinear(8, 9, dtype=dtype)
    return model


def test_pooler_owned_classifier_finds_the_seq_cls_classifier():
    model = _seq_cls_model()
    assert pooler_owned_classifier(model, model.pooler) is model.classifier


def test_pooler_owned_classifier_finds_classifier_through_dispatch_pooler():
    """Regression: DispatchPooler.poolers_by_task is a plain dict, not an
    nn.ModuleDict, so ``pooler.modules()`` never descends into it. A version of
    ``pooler_owned_classifier`` that used ``.modules()`` directly reported the
    reranker's own classifier as unowned, skipped the CPU fallback, and ran an
    FP32 batchmatmul on Spyre. This mirrors DispatchPooler.for_seq_cls exactly,
    the real reranker shape (bert.py:805 / roberta.py:307).
    """
    classifier = _RecordingLinear(8, 2, dtype=torch.float32)
    seq_cls_pooler = SequencePooler(
        pooling=CLSPool(),
        head=ClassifierPoolerHead(classifier=classifier, activation=PoolerClassify()),
    )
    token_classify_pooler = TokenPooler(
        pooling=AllPool(),
        head=TokenClassifierPoolerHead(classifier=classifier, activation=PoolerClassify()),
    )
    dispatch = DispatchPooler({"classify": seq_cls_pooler, "token_classify": token_classify_pooler})
    model = _model_with_pooler(dispatch)
    model.classifier = classifier

    assert pooler_owned_classifier(model, model.pooler) is classifier


def test_configure_pooling_dispatch_seq_cls_fp32_classifier_moves_to_cpu():
    """The reranker end-to-end shape: FP32 classifier inside a DispatchPooler
    must still be detected and moved -- this is the exact case that regressed.
    """
    classifier = _RecordingLinear(8, 2, dtype=torch.float32)
    seq_cls_pooler = SequencePooler(
        pooling=CLSPool(),
        head=ClassifierPoolerHead(classifier=classifier, activation=PoolerClassify()),
    )
    dispatch = DispatchPooler({"classify": seq_cls_pooler})
    model = _model_with_pooler(dispatch)
    model.classifier = classifier

    assert configure_pooling_for_spyre(model, _SPYRE) is False
    assert classifier.to_calls, "DispatchPooler-wrapped FP32 classifier should have moved to CPU"


def test_pooler_owned_classifier_ignores_a_forward_owned_classifier():
    model = _token_classify_model()
    assert pooler_owned_classifier(model, model.pooler) is None


def test_pooler_owned_classifier_handles_no_classifier():
    model = _model_with_pooler(_embed_pooler(CLSPool()))
    assert pooler_owned_classifier(model, model.pooler) is None


def test_configure_pooling_fp32_classifier_falls_back_to_cpu():
    """A reranker's FP32 head sends the pooler and its classifier to CPU."""
    model = _seq_cls_model(dtype=torch.float32)
    assert configure_pooling_for_spyre(model, _SPYRE) is False
    # CLS was patched before the FP32 check; on-Spyre is still False.
    assert isinstance(model.pooler.pooling, SpyreCLSPool)
    assert model.classifier.to_calls, "pooler-owned classifier should have been moved"


def test_configure_pooling_leaves_forward_owned_classifier_alone():
    """Token classification: pooler to CPU, classifier untouched on the device.

    ``TokenPooler`` has no Spyre path, so the pooler runs on the host and the token
    task stays available. The classifier must not follow it: ``forward`` applies it
    to Spyre activations, so a CPU weight there raises a device mismatch.
    ``TorchSpyrePlatform._force_fp16_head_for_token_classification`` keeps it in
    float16 so it can run on device at all.
    """
    model = _token_classify_model(dtype=torch.float16)
    assert configure_pooling_for_spyre(model, _SPYRE) is False
    assert model.classifier.to_calls == [], (
        f"forward-owned classifier was moved: {model.classifier.to_calls}"
    )


def test_configure_pooling_token_pooler_fp32_classifier_still_not_moved():
    """Even an FP32 forward-owned classifier stays put — moving it cannot help.

    Guards against reintroducing the unconditional ``model.classifier.to("cpu")``:
    the FP32 case is the tempting one, and it is exactly the case that used to
    break these models.
    """
    model = _token_classify_model(dtype=torch.float32)
    assert configure_pooling_for_spyre(model, _SPYRE) is False
    assert model.classifier.to_calls == []


def test_configure_pooling_no_pooler_returns_false():
    assert configure_pooling_for_spyre(nn.Module(), _SPYRE) is False
