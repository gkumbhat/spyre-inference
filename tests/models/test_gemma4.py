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

"""`force_text_backbone` must keep repairing plain text gemma-4 checkpoints (the
transformers>=5.16 heterogeneous-config issue) while leaving a real multimodal
Gemma4Config (populated vision_config/audio_config) alone, so vLLM's own
Gemma4ForConditionalGeneration can resolve normally. No card needed: this is pure
EngineArgs/config-object plumbing, mocked at the vllm.transformers_utils.config.get_config
seam `force_text_backbone` imports locally.
"""

from types import SimpleNamespace

import pytest

from spyre_inference.models.gemma4 import (
    _gemma4_multimodal_head_dim_override,
    _gemma4_text_backbone_override,
    force_text_backbone,
)


def _engine_args(**overrides):
    defaults = dict(
        hf_overrides=None,
        hf_config_path=None,
        model="google/gemma-4-26B-A4B",
        trust_remote_code=False,
        revision=None,
        code_revision=None,
        config_format="auto",
        hf_token=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _patch_get_config(monkeypatch, hf_config):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_config",
        lambda *args, **kwargs: hf_config,
    )


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_text"])
def test_force_text_backbone_applies_for_pure_text_checkpoint(monkeypatch, model_type):
    """A plain text checkpoint (no vision_config/audio_config) still gets the fix."""
    hf_config = SimpleNamespace(model_type=model_type, vision_config=None, audio_config=None)
    _patch_get_config(monkeypatch, hf_config)

    engine_args = _engine_args()
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides is _gemma4_text_backbone_override


def test_force_text_backbone_skips_for_vlm_checkpoint_with_vision_config(monkeypatch):
    """A real multimodal Gemma4Config (populated vision_config) must not be forced
    onto the text-only backbone — that would strip vision entirely — but still needs
    the heterogeneous head-dim repair (transformers>=5.16), since
    Gemma4ForConditionalGeneration re-triggers a bare head_dim read on the nested
    text_config."""
    hf_config = SimpleNamespace(
        model_type="gemma4",
        vision_config=SimpleNamespace(model_type="gemma4_vision"),
        audio_config=None,
    )
    _patch_get_config(monkeypatch, hf_config)

    engine_args = _engine_args()
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides is _gemma4_multimodal_head_dim_override


def test_audio_only_checkpoint_is_rejected(monkeypatch):
    """Audio is unsupported, and falling through to the text backbone would silently
    drop the tower -- a model that looks fine but ignores its audio inputs."""
    hf_config = SimpleNamespace(
        model_type="gemma4",
        vision_config=None,
        audio_config=SimpleNamespace(model_type="gemma4_audio"),
    )
    _patch_get_config(monkeypatch, hf_config)

    with pytest.raises(NotImplementedError, match="audio is not supported"):
        force_text_backbone(_engine_args())


def test_vision_plus_audio_checkpoint_is_allowed_for_its_vision_path(monkeypatch):
    """A checkpoint with both towers still works for image and text; only the audio
    input is unsupported, so it warns rather than refusing to load."""
    hf_config = SimpleNamespace(
        model_type="gemma4",
        vision_config=SimpleNamespace(model_type="gemma4_vision"),
        audio_config=SimpleNamespace(model_type="gemma4_audio"),
    )
    _patch_get_config(monkeypatch, hf_config)

    engine_args = _engine_args()
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides is _gemma4_multimodal_head_dim_override


def test_multimodal_head_dim_override_repairs_access_without_touching_architectures():
    text_config = SimpleNamespace(architectures=None)
    config = SimpleNamespace(architectures=["Gemma4ForConditionalGeneration"], text_config=text_config)

    result = _gemma4_multimodal_head_dim_override(config)

    assert result is config
    assert config.architectures == ["Gemma4ForConditionalGeneration"]
    assert config.allow_global_per_layer_attribute_access is True
    assert text_config.allow_global_per_layer_attribute_access is True


def test_force_text_backbone_ignores_unrelated_model_type(monkeypatch):
    hf_config = SimpleNamespace(model_type="llama", vision_config=None, audio_config=None)
    _patch_get_config(monkeypatch, hf_config)

    engine_args = _engine_args()
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides is None


@pytest.mark.parametrize(
    ("hf_config", "expected"),
    [
        (SimpleNamespace(model_type="gemma4", vision_config=object(), audio_config=None), True),
        # Audio is out of scope, so an audio-only config is not "multimodal" here.
        (SimpleNamespace(model_type="gemma4", vision_config=None, audio_config=object()), False),
        # Text-only gemma-4 is already validated in fp16 here; don't move it.
        (SimpleNamespace(model_type="gemma4", vision_config=None, audio_config=None), False),
        (SimpleNamespace(model_type="gemma4_text", vision_config=None, audio_config=None), False),
        (SimpleNamespace(model_type="llama", vision_config=None, audio_config=None), False),
    ],
)
def test_is_multimodal_gemma4_requires_a_vision_tower(hf_config, expected):
    """Gemma-4 overflows fp16's range, but the bf16 switch this gates is scoped to the
    vision checkpoints -- widening it would change every existing Gemma deployment."""
    from spyre_inference.models.gemma4 import is_multimodal_gemma4

    assert is_multimodal_gemma4(hf_config) is expected


def test_platform_selects_bfloat16_for_a_multimodal_gemma4_config():
    """The platform's dtype default runs after vLLM has already resolved
    ``model_config.dtype``, so it has to decide from the model config."""
    import torch

    from spyre_inference.platform import TorchSpyrePlatform

    vision_cfg = SimpleNamespace(
        model_type="gemma4", vision_config=object(), audio_config=None
    )
    text_cfg = SimpleNamespace(model_type="gemma4", vision_config=None, audio_config=None)

    def _config(hf_config):
        return SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config))

    assert TorchSpyrePlatform._default_dtype(_config(vision_cfg)) is torch.bfloat16
    assert TorchSpyrePlatform._default_dtype(_config(text_cfg)) is torch.float16
    # No hf_config at all (bare VllmConfig) must not crash.
    assert TorchSpyrePlatform._default_dtype(_config(None)) is torch.float16


def test_platform_keeps_bfloat16_for_the_nested_text_config():
    """The regression that motivated the sticky marker.

    A multimodal model builds its decoder via ``init_vllm_registered_model`` ->
    ``VllmConfig.with_hf_config``, which re-runs this hook with the bare
    ``gemma4_text`` half (no ``vision_config``). Deciding afresh there downgrades the
    text decoder to fp16 while its weights stay bf16; ``head_dtype`` then disagrees
    with the hidden states and the lm-head silently takes upstream's ``F.linear``
    fallback, which reads a ``weight`` the Spyre path deliberately leaves on CPU.
    """
    import torch

    from spyre_inference.platform import TorchSpyrePlatform

    vision_cfg = SimpleNamespace(
        model_type="gemma4", vision_config=object(), audio_config=None
    )
    model_config = SimpleNamespace(hf_config=vision_cfg)
    outer = SimpleNamespace(model_config=model_config)

    assert TorchSpyrePlatform._default_dtype(outer) is torch.bfloat16

    # with_hf_config deep-copies the ModelConfig and swaps in the text half.
    import copy

    nested_model_config = copy.deepcopy(model_config)
    nested_model_config.hf_config = SimpleNamespace(
        model_type="gemma4_text", vision_config=None, audio_config=None
    )
    nested = SimpleNamespace(model_config=nested_model_config)

    assert TorchSpyrePlatform._default_dtype(nested) is torch.bfloat16, (
        "the nested text config must inherit bf16, not fall back to fp16"
    )


def test_attention_backend_accepts_both_supported_dtypes():
    """bf16 has to be declared on the backend too, or selection rejects the model."""
    import torch

    from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionBackend

    assert torch.float16 in SpyreAttentionBackend.supported_dtypes
    assert torch.bfloat16 in SpyreAttentionBackend.supported_dtypes


def test_multimodal_gemma4_delegates_its_text_half_through_the_model_registry():
    """Why ``Gemma4ForConditionalGeneration`` needs no ``_ADAPTED_ARCHS`` entry.

    It builds its text half with ``init_vllm_registered_model(architectures=
    ["Gemma4ForCausalLM"])``, which resolves through ``ModelRegistry`` and therefore
    already lands on ``SpyreGemma4ForCausalLM`` (PLE aliased scalars + the Spyre MoE
    expert recipe). If upstream ever constructs the text model directly instead,
    those adaptations vanish silently -- hence this tripwire rather than a comment.
    """
    import inspect

    gemma4_mm = pytest.importorskip("vllm.model_executor.models.gemma4_mm")

    source = inspect.getsource(gemma4_mm.Gemma4ForConditionalGeneration.__init__)
    assert "init_vllm_registered_model" in source, (
        "Gemma4ForConditionalGeneration no longer resolves its text model through "
        "the vLLM registry; SpyreGemma4ForCausalLM's MoE/PLE adaptations are being "
        "bypassed. Register a Spyre subclass of it in models/_ADAPTED_ARCHS."
    )
    assert '"Gemma4ForCausalLM"' in source, (
        "Gemma4ForConditionalGeneration no longer names Gemma4ForCausalLM as its "
        "text architecture; check which architecture it resolves now and make sure "
        "spyre_models() adapts that one."
    )


def test_force_text_backbone_respects_user_supplied_hf_overrides(monkeypatch):
    """Skipped entirely (not even a get_config call) when the user already set
    hf_overrides -- existing behavior, unrelated to the vision_config gate."""

    def _boom(*args, **kwargs):
        raise AssertionError("get_config should not be called when hf_overrides is already set")

    monkeypatch.setattr("vllm.transformers_utils.config.get_config", _boom)

    def _user_override(config):
        return config

    engine_args = _engine_args(hf_overrides=_user_override)
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides is _user_override
