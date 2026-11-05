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
"""The encoder process: runs a multimodal model's vision tower and projector only.

Spawned with the Spyre plugin excluded (see ``ParallelEncoderExecutor``), so on a CPU
encoder it resolves vLLM's stock ``CpuPlatform`` and stock model classes. The language
model is skipped through ``mm_config.mm_encoder_only``. It consumes ``EncodeBatch``
messages, runs ``embed_multimodal``, writes the projected rows into the shared region at
the scheduler's block IDs, and posts ``EncodeDone``.
"""

import contextlib
import os
import queue
import time
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Any, cast

import torch

from spyre_inference.v1.executor.parallel_encoder.channel import (
    EncodeBatch,
    EncodeDone,
    EncodeItem,
)
from spyre_inference.v1.executor.parallel_encoder.profiling import PROFILE_ENABLED

if TYPE_CHECKING:
    from vllm.config import VllmConfig

READY = "READY"


def _logger():
    from vllm.logger import init_logger

    return init_logger(__name__)


class _EncProfiler:
    """Gated (``PARALLEL_ENCODER_PROFILE``) timing: encode compute vs region write vs
    idle."""

    def __init__(self) -> None:
        self.enabled = PROFILE_ENABLED
        self._interval = float(os.environ.get("PARALLEL_ENCODER_PROFILE_INTERVAL", "5"))
        self._last = time.monotonic()
        self._reset()

    def _reset(self) -> None:
        self.batches = 0
        self.items = 0
        self.max_batch = 0
        self.encode_s = 0.0
        self.write_s = 0.0
        self.wait_s = 0.0
        self.max_queued = 0

    def record(self, items: int, encode_s: float, write_s: float) -> None:
        self.batches += 1
        self.items += items
        self.max_batch = max(self.max_batch, items)
        self.encode_s += encode_s
        self.write_s += write_s

    def record_queued(self, queued: int) -> None:
        self.max_queued = max(self.max_queued, queued)

    def add_wait(self, wait_s: float) -> None:
        self.wait_s += wait_s

    def maybe_log(self) -> None:
        now = time.monotonic()
        dt = now - self._last
        if dt < self._interval:
            return
        b = max(self.batches, 1)
        busy = self.encode_s + self.write_s
        _logger().info(
            "parallel-encoder PROFILE[enc] %.1fs: %d batches / %d items "
            "(avg=%.1f max=%d) | encode mean=%.2fms | write mean=%.2fms | "
            "busy=%.0f%% idle=%.0f%% | max queued=%d",
            dt,
            self.batches,
            self.items,
            self.items / b,
            self.max_batch,
            1e3 * self.encode_s / b,
            1e3 * self.write_s / b,
            100 * busy / dt if dt else 0.0,
            100 * self.wait_s / dt if dt else 0.0,
            self.max_queued,
        )
        self._last = now
        self._reset()


class EncoderProc:
    """Vision-encoder-only model host, constructed inside the child.

    ``vllm_config`` is the executor's clone: world size 1, eager, compute dtype, and
    ``mm_encoder_only``. ``region_dtype`` is the decoder's dtype, which the region rows
    are written in.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        distributed_init_method: str,
        region_engine_id: str,
        region_num_blocks: int,
        region_block_size_bytes: int,
        region_dtype: torch.dtype,
        max_batch_embeds: int,
        num_threads: int | None,
    ) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
            ECSharedRegion,
        )
        from vllm.model_executor.model_loader import get_model
        from vllm.platforms import current_platform
        from vllm.v1.worker.cpu_model_runner import _set_torch_accelerator_to_noop
        from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

        from spyre_inference.v1.executor.parallel_encoder.region_io import RegionWriter

        # The loader calls torch.accelerator.empty_cache(), which raises without an
        # accelerator; vLLM's CPU runner installs the same no-ops.
        _set_torch_accelerator_to_noop()
        if num_threads:
            torch.set_num_threads(num_threads)
        self.vllm_config = vllm_config
        self.region_dtype = region_dtype
        self.max_batch_embeds = max_batch_embeds
        self.device = torch.device("cpu")

        logger = _logger()
        logger.info(
            "parallel-encoder: encoder process on %s platform, dtype=%s, threads=%d (pid=%d)",
            type(current_platform).__name__,
            vllm_config.model_config.dtype,
            torch.get_num_threads(),
            os.getpid(),
        )

        # initialize_model_parallel() and some CustomOps read the current config.
        with set_current_vllm_config(vllm_config):
            init_worker_distributed_environment(
                vllm_config,
                rank=0,
                distributed_init_method=distributed_init_method,
                local_rank=0,
                backend=current_platform.dist_backend,
            )
            self.model = get_model(vllm_config=vllm_config)
        self.model.eval()

        # Attach (not create) the region prepared by the executor.
        self._region = ECSharedRegion(
            engine_id=region_engine_id,
            num_blocks=region_num_blocks,
            block_size_bytes=region_block_size_bytes,
        )
        self._writer = RegionWriter(self._region)
        self._logged_first_encode = False
        self._prof = _EncProfiler()

    @torch.inference_mode()
    def encode_items(self, items: list[EncodeItem]) -> list[str]:
        """Encode ``items`` and write them to the region; returns their mm_hashes."""
        from vllm.config import set_current_vllm_config
        from vllm.multimodal.utils import group_and_batch_mm_kwargs

        if not items:
            return []
        mm_kwargs = [(item.modality, item.data) for item in items]

        prof = self._prof
        t_enc0 = time.perf_counter() if prof.enabled else 0.0
        outputs: list[torch.Tensor] = []
        with set_current_vllm_config(self.vllm_config):
            for _modality, _num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
                mm_kwargs, device=self.device, pin_memory=False
            ):
                # (num_items, rows, hidden) tensor or a list of (rows, hidden).
                outputs.extend(cast(Any, self.model).embed_multimodal(**mm_kwargs_batch))
        assert len(outputs) == len(items), (
            f"encoder produced {len(outputs)} outputs for {len(items)} items"
        )

        write_items = []
        for item, out in zip(items, outputs):
            out = out.to(self.region_dtype).contiguous()
            assert out.shape[0] == len(item.block_ids), (
                f"parallel-encoder: {item.mm_hash} produced {out.shape[0]} rows, "
                f"the scheduler allocated {len(item.block_ids)}"
            )
            write_items.append((item.mm_hash, item.block_ids, out))

        t_write0 = time.perf_counter() if prof.enabled else 0.0
        self._writer.write_batch(write_items)
        if prof.enabled:
            prof.record(len(items), t_write0 - t_enc0, time.perf_counter() - t_write0)

        if not self._logged_first_encode:
            _logger().info(
                "parallel-encoder: first batch of %d item(s) encoded and written to the "
                "shared region",
                len(items),
            )
            self._logged_first_encode = True
        return [item.mm_hash for item in items]

    def shutdown(self) -> None:
        try:
            self._region.cleanup()
        except Exception:
            _logger().debug("parallel-encoder: region cleanup failed", exc_info=True)


def _take_budget_batch(
    pending: list[EncodeItem], budget: int
) -> tuple[list[EncodeItem], list[EncodeItem]]:
    """Split ``pending`` into the leading items that fit ``budget`` embeddings and the
    rest; the first item is always taken."""
    used = 0
    cut = 0
    for item in pending:
        n = len(item.block_ids)
        if cut > 0 and used + n > budget:
            break
        used += n
        cut += 1
    return pending[:cut], pending[cut:]


def encoder_main(
    vllm_config: "VllmConfig",
    distributed_init_method: str,
    region_engine_id: str,
    region_num_blocks: int,
    region_block_size_bytes: int,
    region_dtype: torch.dtype,
    max_batch_embeds: int,
    num_threads: int | None,
    dispatch_q,
    completion_q,
    ready_pipe: Connection,
    death_pipe: Connection,
) -> None:
    """Child entry point: load, signal READY, then serve ``EncodeBatch`` messages until
    the parent exits (``death_pipe`` EOF). An encode failure is posted on
    ``completion_q`` and then re-raised; the executor's monitor shuts the engine down."""
    # Without the plugin, nothing configures logging here: set it up as vLLM's workers
    # do and route spyre_inference through vllm's handler, or every line is dropped.
    import logging

    from vllm.logger import configure_logging

    configure_logging(vllm_config.logging_config)
    logging.getLogger("spyre_inference").parent = logging.getLogger("vllm")
    logger = _logger()
    proc: EncoderProc | None = None
    try:
        proc = EncoderProc(
            vllm_config,
            distributed_init_method,
            region_engine_id,
            region_num_blocks,
            region_block_size_bytes,
            region_dtype,
            max_batch_embeds,
            num_threads,
        )
        ready_pipe.send({"status": READY})
    except Exception as e:
        logger.exception("parallel-encoder: init failed")
        with contextlib.suppress(Exception):
            ready_pipe.send({"status": "FAILED", "error": f"{type(e).__name__}: {e}"})
        return

    # Continuous batching across dispatches, bounded by max_batch_embeds.
    prof = proc._prof
    pending: list[EncodeItem] = []
    try:
        while True:
            if death_pipe.poll(0):
                logger.info("parallel-encoder: parent exited, shutting down")
                break

            t_wait0 = time.perf_counter() if prof.enabled else 0.0
            if not pending:
                try:
                    batch: EncodeBatch = dispatch_q.get(timeout=0.05)
                except queue.Empty:
                    if prof.enabled:
                        prof.add_wait(time.perf_counter() - t_wait0)
                        prof.maybe_log()
                    continue
                pending.extend(batch.items)
            while True:
                try:
                    batch = dispatch_q.get_nowait()
                except queue.Empty:
                    break
                pending.extend(batch.items)

            if prof.enabled:
                prof.record_queued(len(pending))
            to_encode, pending = _take_budget_batch(pending, max_batch_embeds)
            if prof.enabled:
                prof.add_wait(time.perf_counter() - t_wait0)

            try:
                done_hashes = proc.encode_items(to_encode)
            except Exception as e:
                logger.exception("parallel-encoder: encode failed")
                for item in to_encode:
                    completion_q.put(EncodeDone(mm_hash=item.mm_hash, ok=False, error=str(e)))
                raise
            for mm_hash in done_hashes:
                completion_q.put(EncodeDone(mm_hash=mm_hash, ok=True))
            if prof.enabled:
                prof.maybe_log()
    finally:
        proc.shutdown()
