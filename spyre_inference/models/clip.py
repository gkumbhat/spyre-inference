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

"""CLIP EngineArgs overrides that must land before ModelConfig is built.

The LayerNorm boundary-norm swap lives in ``multimodal/clip.py`` (an
instance-level, post-load patch, applied via ``apply_multimodal_patches``).
This module is the pre-``ModelConfig`` counterpart, following the same
pattern as ``models.gemma4.force_text_backbone``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)

_CLIP_MODEL_TYPES = {"clip"}


def force_disable_chunked_prefill(engine_args: EngineArgs) -> None:
    """CLIP supports vLLM's ``token_embed`` task (``tok_pooling_type="ALL"``),
    which ``spyre_inference.v1.pool.spyre_pooler.SpyreAllPool`` refuses to run
    under chunked prefill (raises ``NotImplementedError`` at construction, so
    it fails even for requests that only ever use the plain ``embed`` task).

    ``ModelConfig.is_chunked_prefill_supported`` should already disable
    chunked prefill for a causal pooling model here -- it excludes MEAN/CLS
    seq-pooling and STEP tok-pooling, but not ALL tok-pooling, which is what
    CLIP (and anything else using ``SpyreAllPool``) actually needs excluded
    too. Rather than patch that vLLM heuristic globally (touching every
    pooling model, not just the ones that hit this), force it off here,
    narrowly, for CLIP specifically -- mirrors
    ``models.gemma4.force_text_backbone``'s early ``get_config()`` probe.
    Skipped when the user explicitly set ``--enable-chunked-prefill``.
    """
    if engine_args.enable_chunked_prefill is not None:
        return
    from vllm.transformers_utils.config import get_config

    try:
        hf_config = get_config(
            engine_args.hf_config_path or engine_args.model,
            engine_args.trust_remote_code,
            engine_args.revision,
            engine_args.code_revision,
            engine_args.config_format,
            token=engine_args.hf_token,
        )
    except Exception:
        return
    if getattr(hf_config, "model_type", None) not in _CLIP_MODEL_TYPES:
        return
    engine_args.enable_chunked_prefill = False
    logger.info("CLIP: disabling chunked prefill (unsupported with token-level pooling).")
