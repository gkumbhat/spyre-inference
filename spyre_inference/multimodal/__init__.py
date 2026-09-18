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

"""Per-architecture workarounds for multimodal models on Spyre.

One module per architecture, each exposing `apply(model, device)`. These monkeypatch
upstream vLLM model code, unlike `custom_ops/`, which registers out-of-tree
implementations by layer class.
"""

import torch

from . import clip, pixtral


def apply_multimodal_patches(model: torch.nn.Module, device: torch.device) -> None:
    """Apply every Spyre workaround the loaded model's vision path needs.

    A no-op for text-only models. Call after weights are on the device but before
    compile, which wraps modules in `OptimizedModule` and breaks traversal.
    """
    # Both spellings: mistral-format Pixtral names the tower `vision_encoder`, HF-format
    # Mistral3 `vision_tower`. Ungated, the patches rewrite vLLM's shared module.
    if any(hasattr(model, attr) for attr in ("vision_encoder", "vision_tower")):
        pixtral.apply(model, device)

    # CLIPEmbeddingModel: text_model/vision_model, not vision_encoder/vision_tower.
    # Gated on model_type, not just attribute presence: other architectures (e.g.
    # BLIP-2) also set `vision_model`, and clip.apply() assumes CLIP's specific
    # LayerNorm-based boundary norms.
    hf_config = getattr(model, "config", None)
    if getattr(hf_config, "model_type", None) == "clip":
        clip.apply(model, device)
