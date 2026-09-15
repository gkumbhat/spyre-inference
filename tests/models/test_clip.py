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

"""Cheap unit tests for the Spyre CLIP embedding model adaptation.

No Spyre hardware, no real ``VllmConfig``: ``CLIPEmbeddingModel.__init__`` is
monkeypatched to build a minimal stand-in with real ``nn.LayerNorm`` instances
(mirroring ``vision_model``/``text_model``'s shape) so the subclass's own
``__init__`` runs for real and its boundary-norm swap can be checked directly.
``tests/models/test_model_registration.py`` covers the ``_ADAPTED_ARCHS``
registration wiring generically for every architecture, this one included.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from spyre_inference.custom_ops.layer_norm import SpyreLayerNorm
from spyre_inference.models.clip import SpyreCLIPEmbeddingModel, _to_spyre_layer_norm


def _fake_clip_embedding_init(hidden_size: int = 64, with_post_norm: bool = True):
    """Monkeypatch target for ``CLIPEmbeddingModel.__init__``: skips the real
    ``VllmConfig``/weight-construction machinery, but sets up
    ``text_model``/``vision_model`` with real ``nn.LayerNorm`` boundary norms,
    matching what ``SpyreCLIPEmbeddingModel.__init__`` reads and swaps."""

    def _init(self, *, vllm_config, prefix: str = "") -> None:
        torch.nn.Module.__init__(self)
        self.text_model = types.SimpleNamespace(
            final_layer_norm=torch.nn.LayerNorm(hidden_size)
        )
        self.vision_model = types.SimpleNamespace(
            pre_layrnorm=torch.nn.LayerNorm(hidden_size),
            post_layernorm=(torch.nn.LayerNorm(hidden_size) if with_post_norm else None),
        )

    return _init


def _build(monkeypatch, with_post_norm: bool = True) -> SpyreCLIPEmbeddingModel:
    from vllm.model_executor.models.clip import CLIPEmbeddingModel

    monkeypatch.setattr(
        CLIPEmbeddingModel, "__init__", _fake_clip_embedding_init(with_post_norm=with_post_norm)
    )
    return SpyreCLIPEmbeddingModel(vllm_config=None)


def test_boundary_norms_swapped_to_spyre_layer_norm(monkeypatch):
    model = _build(monkeypatch)

    assert isinstance(model.text_model.final_layer_norm, SpyreLayerNorm)
    assert isinstance(model.vision_model.pre_layrnorm, SpyreLayerNorm)
    assert isinstance(model.vision_model.post_layernorm, SpyreLayerNorm)


def test_missing_post_layernorm_is_left_none(monkeypatch):
    """CLIPVisionTransformer.post_layernorm can be None (require_post_norm=False);
    the swap must not crash on a missing/None attribute."""
    model = _build(monkeypatch, with_post_norm=False)

    assert isinstance(model.vision_model.pre_layrnorm, SpyreLayerNorm)
    assert model.vision_model.post_layernorm is None


@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_to_spyre_layer_norm_preserves_shape_eps_affine_bias(elementwise_affine, bias):
    hidden_size, eps = 128, 1e-6
    original = torch.nn.LayerNorm(
        hidden_size, eps=eps, elementwise_affine=elementwise_affine, bias=bias
    )

    patched = _to_spyre_layer_norm(original, SpyreLayerNorm)

    assert isinstance(patched, SpyreLayerNorm)
    assert patched.normalized_shape == original.normalized_shape
    assert patched.eps == original.eps
    assert patched.elementwise_affine == original.elementwise_affine
    assert (patched.bias is not None) == (original.bias is not None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
