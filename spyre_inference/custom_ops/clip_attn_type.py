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

"""Force CLIP's text tower onto ``AttentionType.ENCODER_ONLY``.

``CLIPTextTransformer`` builds its ``Attention`` layer (``clip.py``'s
``attn_cls=Attention``) without an explicit ``attn_type``, so it silently
defaults to ``AttentionType.DECODER`` -- unlike BERT/RoBERTa, which declare
``@attn_type("encoder_only")``, even though CLIP's text tower is the same
shape: one whole-sequence forward pass, no persisted KV cache across calls.
On Spyre, a DECODER-typed layer routes through the paged-KV attention
builder, which needs block-table buckets sized from a real KV cache -- and
there is none, since ``TorchSpyrePlatform`` correctly skips sizing one for
pooling models. Result: ``num_blocks=N exceeds the largest recorded bucket``
under the default (compiled) mode.

Verified this doesn't change ``CLIPAttention``'s output: it calls
``self.attn(q, k, v)`` with no mask either way, so ``attn_type`` here only
affects KV-cache/scheduling, not masking (confirmed byte-identical embeddings
against the default, both matching the HF ``transformers`` reference).

Must run before ``CLIPTextTransformer``/``CLIPAttention`` are constructed:
``Attention.__init__`` bakes ``attn_type`` into both its chosen backend impl
(``self.impl``) and its KV-cache-spec registration, so patching an
already-constructed instance is too late -- this has to replace the
module-local ``Attention`` name before model loading, which is why it lives
here (called from ``register_all()``, which ``TorchSpyreWorker.init_device``
runs before ``load_model``) rather than in ``multimodal/clip.py`` (whose
``apply(model, device)`` runs after the model -- and its attention layers --
already exist).

Should be fixed upstream in ``vllm.model_executor.models.clip``; this patches
only the module-local ``Attention`` name that module's ``attn_cls=Attention``
resolves against, so no other model is affected.
"""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)


def register() -> None:
    import vllm.model_executor.models.clip as clip_mod
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backend import AttentionType

    if getattr(clip_mod.Attention, "_spyre_forces_encoder_only", False):
        return

    class _SpyreClipTextAttention(Attention):
        _spyre_forces_encoder_only = True

        def __init__(self, *args, **kwargs) -> None:
            kwargs.setdefault("attn_type", AttentionType.ENCODER_ONLY)
            super().__init__(*args, **kwargs)

    clip_mod.Attention = _SpyreClipTextAttention
    logger.debug_once(
        "Patched vllm.model_executor.models.clip.Attention to default to "
        "AttentionType.ENCODER_ONLY"
    )
