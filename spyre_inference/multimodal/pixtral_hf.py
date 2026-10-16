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

"""HF-format Pixtral (Mistral3, `PixtralHFVisionModel`) vision-tower workarounds for Spyre.

Native Mistral-format Pixtral is adapted by `models.pixtral` instead. The tower's attention
runs through vLLM's `MMEncoderAttention`, which `custom_ops/vit_attn.py` replaces; what is
left here is the default-layout norm after the patch conv.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert

logger = init_logger(__name__)


class _DefaultLayoutNorm(nn.Module):
    """Materialize a default-layout input before Pixtral's pre-transformer norm."""

    def __init__(self, norm: nn.Module) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "spyre":
            device = x.device
            x = convert(convert(x, device="cpu").contiguous(), device=device)
        return self.norm(x)


def patch_pre_transformer_norm(model: nn.Module) -> None:
    """Reset the patch-conv layout before RMSNorm's fp32 accumulation.

    Flattening the channel-tiled convolution output preserves a device layout whose
    patch-grid dimension can contain an odd number of fp32 sticks. That layout cannot
    be rescaled for RMSNorm's fp32-to-fp16 conversion. A CPU round trip after flattening
    materializes the logical ``[batch, patches, hidden]`` tensor in its default layout.
    """
    tower = getattr(model, "vision_tower", None)
    if tower is None or isinstance(tower.ln_pre, _DefaultLayoutNorm):
        return
    tower.ln_pre = _DefaultLayoutNorm(tower.ln_pre)
    logger.info("Spyre: Pixtral pre-transformer norm input uses the default device layout.")


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Install the HF-format Pixtral tower workarounds. `device` is unused; it stays for the
    `apply(model, device)` contract the sibling architecture modules share."""
    patch_pre_transformer_norm(model)
