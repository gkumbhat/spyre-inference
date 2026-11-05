# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0
"""Settings for the in-engine parallel encoder, read from
``additional_config["parallel_encoder"]``."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_ADDITIONAL_CONFIG_KEY = "parallel_encoder"
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass(frozen=True)
class ParallelEncoderConfig:
    """``encoder_device`` is ``"cpu"`` or ``"spyre:<idx>"`` with idx outside the
    decoder TP set. ``encoder_dtype`` is the tower's compute dtype on CPU, where the
    platform's fp16 is slow; embeddings are cast to the decoder dtype on write.
    ``encoder_num_threads`` caps the CPU encoder's torch threads (``None`` keeps
    torch's default)."""

    enabled: bool = False
    encoder_device: str = "cpu"
    encoder_dtype: str = "bfloat16"
    encoder_num_threads: int | None = None
    max_dispatch_batch_embeds: int | None = None
    # Encoder working set in embedding rows: region size and EncoderCacheManager
    # capacity, decoupled from max_num_batched_tokens. None sizes it for ~16 images.
    encoder_cache_tokens: int | None = None
    # Images dispatched but not yet encoded before new multimodal requests wait.
    max_inflight_encode_images: int | None = None

    def __post_init__(self) -> None:
        if self.encoder_dtype not in _DTYPES:
            raise ValueError(
                f"parallel_encoder.encoder_dtype must be one of {sorted(_DTYPES)}, "
                f"got {self.encoder_dtype!r}"
            )
        self.encoder_device_index()

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "ParallelEncoderConfig | None":
        additional = vllm_config.additional_config
        if not isinstance(additional, dict):
            return None
        raw = additional.get(_ADDITIONAL_CONFIG_KEY)
        if not raw:
            return None
        if isinstance(raw, ParallelEncoderConfig):
            return raw
        return cls(**raw)

    @classmethod
    def is_enabled(cls, vllm_config: "VllmConfig") -> bool:
        pe_config = cls.from_vllm_config(vllm_config)
        return pe_config is not None and pe_config.enabled

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.encoder_dtype]

    def encoder_device_index(self) -> int | None:
        """The Spyre card index, or None for a CPU encoder."""
        if self.encoder_device == "cpu":
            return None
        kind, _, idx = self.encoder_device.partition(":")
        if kind != "spyre" or (idx and not idx.isdigit()):
            raise ValueError(
                "parallel_encoder.encoder_device must be 'cpu' or 'spyre:<index>', "
                f"got {self.encoder_device!r}"
            )
        return int(idx) if idx else 0
