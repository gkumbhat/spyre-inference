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

"""Cheap unit tests for the CLIP boundary-LayerNorm patch installer.

No Spyre hardware: exercises `_patch_boundary_layer_norms`/`_to_spyre_layer_norm`
against minimal stand-in classes (mirroring CLIPVisionTransformer's shape) rather
than constructing real vLLM CLIP transformers, and checks `install_spyre_patches`
wires the exact attribute names/classes it's supposed to.
"""

from __future__ import annotations

import sys

import pytest
import torch

from spyre_inference.models.clip import (
    _patch_boundary_layer_norms,
    _to_spyre_layer_norm,
)


class _SpyreMarkerLayerNorm(torch.nn.LayerNorm):
    """Stand-in for SpyreLayerNorm: only identity matters for these tests."""


def _fresh_vision_like_cls(with_post_norm: bool = True):
    """A new class each call, so `_spyre_patched` state never leaks across tests."""

    class _FakeVisionTransformer:
        def __init__(self, hidden_size: int, eps: float = 1e-5):
            self.pre_layrnorm = torch.nn.LayerNorm(hidden_size, eps=eps)
            self.post_layernorm = (
                torch.nn.LayerNorm(hidden_size, eps=eps) if with_post_norm else None
            )
            # Untouched attribute: only pre_layrnorm/post_layernorm are boundary norms.
            self.layer_norm1 = torch.nn.LayerNorm(hidden_size, eps=eps)

    return _FakeVisionTransformer


def test_patch_boundary_layer_norms_swaps_matching_attrs():
    cls = _fresh_vision_like_cls()
    _patch_boundary_layer_norms(
        cls, ("pre_layrnorm", "post_layernorm"), _SpyreMarkerLayerNorm
    )

    instance = cls(hidden_size=64)

    assert isinstance(instance.pre_layrnorm, _SpyreMarkerLayerNorm)
    assert isinstance(instance.post_layernorm, _SpyreMarkerLayerNorm)
    # Not in attr_names: left as a plain nn.LayerNorm.
    assert not isinstance(instance.layer_norm1, _SpyreMarkerLayerNorm)
    assert cls._spyre_patched is True


@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_patch_boundary_layer_norms_preserves_shape_eps_affine_bias(
    elementwise_affine, bias
):
    hidden_size, eps = 128, 1e-6
    original = torch.nn.LayerNorm(
        hidden_size, eps=eps, elementwise_affine=elementwise_affine, bias=bias
    )

    patched = _to_spyre_layer_norm(original, _SpyreMarkerLayerNorm)

    assert isinstance(patched, _SpyreMarkerLayerNorm)
    assert patched.normalized_shape == original.normalized_shape
    assert patched.eps == original.eps
    assert patched.elementwise_affine == original.elementwise_affine
    assert (patched.bias is not None) == (original.bias is not None)


def test_patch_boundary_layer_norms_skips_missing_attr():
    """CLIPVisionTransformer.post_layernorm can be None (require_post_norm=False);
    the patch must not crash trying to swap a missing/None attribute."""
    cls = _fresh_vision_like_cls(with_post_norm=False)
    _patch_boundary_layer_norms(
        cls, ("pre_layrnorm", "post_layernorm"), _SpyreMarkerLayerNorm
    )

    instance = cls(hidden_size=64)

    assert isinstance(instance.pre_layrnorm, _SpyreMarkerLayerNorm)
    assert instance.post_layernorm is None


def test_patch_boundary_layer_norms_idempotent():
    cls = _fresh_vision_like_cls()
    _patch_boundary_layer_norms(cls, ("pre_layrnorm",), _SpyreMarkerLayerNorm)
    wrapped_init = cls.__init__

    _patch_boundary_layer_norms(cls, ("pre_layrnorm",), _SpyreMarkerLayerNorm)

    # Second call is a no-op: same __init__, no double-wrapping.
    assert cls.__init__ is wrapped_init

    instance = cls(hidden_size=32)
    assert isinstance(instance.pre_layrnorm, _SpyreMarkerLayerNorm)


def test_install_spyre_patches_wires_expected_attrs(monkeypatch):
    """Verify the exact (class, attr_names, spyre_cls) wiring without mutating
    the real vLLM CLIP classes -- a rename of pre_layrnorm/post_layernorm/
    final_layer_norm upstream should fail this test."""
    from spyre_inference.models import clip as clip_patches

    calls = []
    monkeypatch.setattr(
        clip_patches,
        "_patch_boundary_layer_norms",
        lambda cls, attr_names, spyre_cls: calls.append((cls, attr_names, spyre_cls)),
    )

    clip_patches.install_spyre_patches()

    from vllm.model_executor.models import clip

    from spyre_inference.custom_ops.layer_norm import SpyreLayerNorm

    assert (
        clip.CLIPVisionTransformer,
        ("pre_layrnorm", "post_layernorm"),
        SpyreLayerNorm,
    ) in calls
    assert (clip.CLIPTextTransformer, ("final_layer_norm",), SpyreLayerNorm) in calls


def test_install_pooling_model_patches_includes_clip(monkeypatch):
    """models/__init__.py's install_pooling_model_patches must call
    clip.install_spyre_patches() alongside bert/roberta."""
    import spyre_inference.models.bert as bert_mod
    import spyre_inference.models.clip as clip_mod
    import spyre_inference.models.roberta as roberta_mod
    from spyre_inference import models

    called = set()
    monkeypatch.setattr(bert_mod, "install_spyre_patches", lambda: called.add("bert"))
    monkeypatch.setattr(roberta_mod, "install_spyre_patches", lambda: called.add("roberta"))
    monkeypatch.setattr(clip_mod, "install_spyre_patches", lambda: called.add("clip"))

    models.install_pooling_model_patches()

    assert called == {"bert", "roberta", "clip"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
