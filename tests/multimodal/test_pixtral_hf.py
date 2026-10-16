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

"""Tests for `spyre_inference/multimodal/pixtral_hf.py` and its dispatch."""

import inspect

import pytest
import torch

pixtral = pytest.importorskip("vllm.model_executor.models.pixtral")

HIDDEN_SIZE = 256


class PixtralHFVisionModel(torch.nn.Module):
    """Dispatch keys on the tower's class name."""

    def __init__(self) -> None:
        super().__init__()
        self.ln_pre = torch.nn.RMSNorm(HIDDEN_SIZE)


def _model() -> torch.nn.Module:
    model = torch.nn.Module()
    model.vision_tower = PixtralHFVisionModel()
    return model


def test_hf_tower_is_dispatched_to_its_patches():
    from spyre_inference.multimodal import apply_multimodal_patches
    from spyre_inference.multimodal.pixtral_hf import _DefaultLayoutNorm

    model = _model()
    apply_multimodal_patches(model, torch.device("cpu"))
    assert isinstance(model.vision_tower.ln_pre, _DefaultLayoutNorm)


def test_patch_is_idempotent():
    from spyre_inference.multimodal import pixtral_hf

    model = _model()
    pixtral_hf.apply(model, torch.device("cpu"))
    norm = model.vision_tower.ln_pre
    pixtral_hf.apply(model, torch.device("cpu"))
    assert model.vision_tower.ln_pre is norm


def test_pre_transformer_norm_patch_is_transparent_on_cpu():
    from spyre_inference.multimodal.pixtral_hf import patch_pre_transformer_norm

    model = _model()
    x = torch.randn(1, 49, HIDDEN_SIZE)
    expected = model.vision_tower.ln_pre(x)
    patch_pre_transformer_norm(model)
    torch.testing.assert_close(model.vision_tower.ln_pre(x), expected)


def test_hf_tower_attention_goes_through_mm_encoder_attention():
    """The tower's mask and attention live in `MMEncoderAttention`, which Spyre replaces out
    of tree, so nothing here patches them; if upstream builds its own mask again, it needs a
    Spyre path."""
    source = inspect.getsource(pixtral)
    assert "generate_block_attention_mask" not in source
    assert "self.attn = MMEncoderAttention(" in inspect.getsource(pixtral.PixtralHFAttention)
    assert "ln_pre" in inspect.getsource(pixtral.PixtralHFVisionModel)


def test_native_pixtral_is_not_patched():
    """The mixin model adapts its own tower."""
    from spyre_inference.models._vision_encoder import SpyreVisionEncoderMixin
    from spyre_inference.multimodal import apply_multimodal_patches

    class _Native(SpyreVisionEncoderMixin, torch.nn.Module):
        pass

    model = _Native()
    model.vision_tower = PixtralHFVisionModel()
    norm = model.vision_tower.ln_pre
    apply_multimodal_patches(model, torch.device("cpu"))
    assert model.vision_tower.ln_pre is norm
