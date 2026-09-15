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

"""Spyre adaptation of vLLM's CLIP embedding model.

Only ``vision_model.pre_layrnorm``/``post_layernorm`` and
``text_model.final_layer_norm`` are swapped to ``SpyreLayerNorm``. Those three
sit at the model boundary, outside any per-block compiled graph, which is what
triggers the crash ``SpyreLayerNorm`` works around (see
``spyre_inference.custom_ops.layer_norm``). ``CLIPEncoderLayer.layer_norm1``/
``layer_norm2`` are traced inside the per-block ``torch.compile`` region
already and never hit that crashing path, so they're left as plain
``nn.LayerNorm`` -- swapping them too would be unnecessary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.models.clip import CLIPEmbeddingModel

from spyre_inference.custom_ops.layer_norm import SpyreLayerNorm

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _to_spyre_layer_norm(
    ln: torch.nn.LayerNorm, spyre_cls: type[torch.nn.LayerNorm]
) -> torch.nn.LayerNorm:
    return spyre_cls(
        list(ln.normalized_shape),
        eps=ln.eps,
        elementwise_affine=ln.elementwise_affine,
        bias=ln.bias is not None,
    )


class SpyreCLIPEmbeddingModel(CLIPEmbeddingModel):
    """CLIP embeddings on Spyre: boundary LayerNorms swapped for SpyreLayerNorm."""

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.text_model.final_layer_norm = _to_spyre_layer_norm(
            self.text_model.final_layer_norm, SpyreLayerNorm
        )
        self.vision_model.pre_layrnorm = _to_spyre_layer_norm(
            self.vision_model.pre_layrnorm, SpyreLayerNorm
        )
        if self.vision_model.post_layernorm is not None:
            self.vision_model.post_layernorm = _to_spyre_layer_norm(
                self.vision_model.post_layernorm, SpyreLayerNorm
            )
