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

"""Spyre adaptations for vLLM's CLIP text/vision towers.

Only ``CLIPVisionTransformer.pre_layrnorm``/``post_layernorm`` and
``CLIPTextTransformer.final_layer_norm`` are swapped to ``SpyreLayerNorm``.
Those three sit at the model boundary, outside any per-block compiled graph,
which is what triggers the crash ``SpyreLayerNorm`` works around (see
``spyre_inference.custom_ops.layer_norm``). ``CLIPEncoderLayer.layer_norm1``/
``layer_norm2`` are traced inside the per-block ``torch.compile`` region
already and never hit that crashing path, so they're left as plain
``nn.LayerNorm`` -- swapping them too would be unnecessary.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def install_spyre_patches() -> None:
    """Swap CLIP's boundary LayerNorms for ``SpyreLayerNorm``."""
    from vllm.model_executor.models import clip

    from spyre_inference.custom_ops.layer_norm import SpyreLayerNorm

    _patch_boundary_layer_norms(
        clip.CLIPVisionTransformer, ("pre_layrnorm", "post_layernorm"), SpyreLayerNorm
    )
    _patch_boundary_layer_norms(
        clip.CLIPTextTransformer, ("final_layer_norm",), SpyreLayerNorm
    )
    logger.info_once(
        "Spyre: CLIP's pre_layrnorm/post_layernorm/final_layer_norm use "
        "SpyreLayerNorm (torch-spyre#4242's unrelated reference bug); "
        "layer_norm1/layer_norm2 inside encoder blocks are unaffected."
    )


def _to_spyre_layer_norm(
    ln: torch.nn.LayerNorm, spyre_cls: type[torch.nn.LayerNorm]
) -> torch.nn.LayerNorm:
    return spyre_cls(
        list(ln.normalized_shape),
        eps=ln.eps,
        elementwise_affine=ln.elementwise_affine,
        bias=ln.bias is not None,
    )


def _patch_boundary_layer_norms(
    cls: type[torch.nn.Module],
    attr_names: tuple[str, ...],
    spyre_cls: type[torch.nn.LayerNorm],
) -> None:
    """Wrap ``cls.__init__`` to swap ``attr_names`` to ``spyre_cls`` after construction.

    Runs after the original ``__init__`` (so weights/eps/etc. are already set
    on the plain ``nn.LayerNorm`` instances) and before weight loading (which
    happens once the full model is constructed), so replacing the submodule
    here doesn't disturb the ``<attr>.weight``/``<attr>.bias`` state-dict keys
    weight loading looks for.
    """
    if getattr(cls, "_spyre_patched", False):
        return

    orig_init = cls.__init__

    def __init__(self, *args, **kwargs) -> None:
        orig_init(self, *args, **kwargs)
        for name in attr_names:
            ln = getattr(self, name, None)
            if ln is None:  # e.g. CLIPVisionTransformer.post_layernorm can be skipped
                continue
            setattr(self, name, _to_spyre_layer_norm(ln, spyre_cls))

    cls.__init__ = __init__  # ty: ignore[invalid-assignment]
    cls._spyre_patched = True
