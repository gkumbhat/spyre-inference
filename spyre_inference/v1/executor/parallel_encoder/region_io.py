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
"""Read/write helpers for the parallel-encoder shared region.

The region is a flat ``(num_blocks, block_size_bytes)`` int8 grid, one block per
embedding row in ``model_config.dtype``; shape and dtype are agreed out of band.
Both sides are host-only: Spyre's merge (``SpyreVisionEncoderMixin.embed_input_ids``)
takes image rows on the host, so nothing here touches a device.
"""

import torch
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import ECSharedRegion

from spyre_inference.v1.executor.parallel_encoder.region import is_contiguous


class RegionWriter:
    """Encoder side. ``write_batch`` returns once the bytes are in the region, so the
    caller can post completion straight after."""

    def __init__(self, region: ECSharedRegion) -> None:
        self._region = region

    def write_batch(self, items: list[tuple[str, list[int], torch.Tensor]]) -> None:
        """Write ``(mm_hash, block_ids, embedding)`` with embedding ``(rows, hidden)``
        on the host, in the region dtype, one row per block."""
        block_size = self._region.block_size_bytes
        blocks = self._region.blocks
        for mm_hash, block_ids, src in items:
            src_bytes = src.reshape(-1).view(torch.int8)
            allocated = len(block_ids) * block_size
            assert src_bytes.numel() <= allocated, (
                f"parallel-encoder: embedding for {mm_hash} is {src_bytes.numel()} bytes "
                f"but only {allocated} are allocated; shape={list(src.shape)}"
            )
            if is_contiguous(block_ids):
                start = block_ids[0]
                blocks[start : start + len(block_ids)].reshape(-1)[: src_bytes.numel()].copy_(
                    src_bytes
                )
                continue
            for i, block_idx in enumerate(block_ids):
                chunk = src_bytes[i * block_size : (i + 1) * block_size]
                blocks[block_idx, : chunk.numel()] = chunk


class RegionReader:
    """Decoder-worker side: region rows -> ``encoder_cache`` host tensors.

    Rows are cloned out because the scheduler reuses freed blocks for later images
    while the worker's cache entry may still be live."""

    def __init__(self, region: ECSharedRegion, dtype: torch.dtype) -> None:
        self._region = region
        self._dtype = dtype

    def load(self, encoder_cache: dict[str, torch.Tensor], loads: dict[str, list[int]]) -> None:
        blocks = self._region.blocks
        for mm_hash, block_ids in loads.items():
            if mm_hash in encoder_cache:
                continue
            if is_contiguous(block_ids):
                start = block_ids[0]
                rows = blocks[start : start + len(block_ids)].clone()
            else:
                rows = blocks[torch.tensor(block_ids, dtype=torch.long)]
            encoder_cache[mm_hash] = rows.view(self._dtype).reshape(len(block_ids), -1)
