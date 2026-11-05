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
"""``MultiprocExecutor`` plus one vision-encoder process.

Select with::

    --distributed-executor-backend \\
        spyre_inference.v1.executor.parallel_encoder.executor.ParallelEncoderExecutor
    --additional-config '{"parallel_encoder": {"enabled": true}}'

The platform then picks ``ParallelEncoderScheduler`` itself.
"""

import contextlib
import copy
import multiprocessing.connection
import os
import weakref
from threading import Thread

from vllm.config import CompilationMode, VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.utils.network_utils import get_distributed_init_method, get_loopback_ip, get_open_port
from vllm.utils.system_utils import get_mp_context
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

from spyre_inference.config.parallel_encoder import ParallelEncoderConfig
from spyre_inference.threading_config import encoder_cpu_share
from spyre_inference.v1.executor.parallel_encoder import channel_slot
from spyre_inference.v1.executor.parallel_encoder.channel import EncoderChannel
from spyre_inference.v1.executor.parallel_encoder.encoder_proc import encoder_main
from spyre_inference.v1.executor.parallel_encoder.region import (
    build_region,
    region_block_size_bytes,
    region_engine_id,
    resolve_encoder_cache_size,
)

logger = init_logger(__name__)

_READY_TIMEOUT_S = 600.0

# A CPU encoder must not load the Spyre plugin: it would resolve TorchSpyrePlatform,
# swap in Spyre custom ops and model classes, and open a card. Without a plugin, the
# `+empty` vLLM build resolves no platform unless the target device is named.
_CPU_ENCODER_ENV = {
    "VLLM_PLUGINS": "",
    "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    "VLLM_TARGET_DEVICE": "cpu",
    # Marks the process for tools that need to find it, such as the crash test.
    "SPYRE_PARALLEL_ENCODER_PROC": "1",
}


@contextlib.contextmanager
def _patched_environ(overrides: dict[str, str]):
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class ParallelEncoderExecutor(MultiprocExecutor):
    def _init_executor(self) -> None:
        pe_config = ParallelEncoderConfig.from_vllm_config(self.vllm_config)
        if pe_config is None or not pe_config.enabled:
            raise ValueError(
                "ParallelEncoderExecutor needs additional_config "
                '{"parallel_encoder": {"enabled": true}}.'
            )
        self._pe_config = pe_config
        self._validate_encoder_device()
        # Region and queues exist before any worker spawns, so attachers never wait.
        self._create_region_and_channel()
        super()._init_executor()

    def _validate_encoder_device(self) -> None:
        idx = self._pe_config.encoder_device_index()
        if idx is None:
            return
        raise NotImplementedError(
            f"parallel_encoder.encoder_device={self._pe_config.encoder_device!r}: a Spyre "
            "encoder card is not supported yet; use 'cpu'."
        )

    def _create_region_and_channel(self) -> None:
        mm_budget = MultiModalBudget(self.vllm_config, MULTIMODAL_REGISTRY)
        num_blocks = resolve_encoder_cache_size(mm_budget, self._pe_config)
        if num_blocks <= 0:
            raise ValueError(
                "parallel encoder needs a positive encoder cache size; the model has no "
                "active non-text modality."
            )
        self._region_num_blocks = num_blocks
        self._max_batch_embeds = min(
            self._pe_config.max_dispatch_batch_embeds or mm_budget.encoder_compute_budget,
            num_blocks,
        )
        self._region_block_size_bytes = region_block_size_bytes(self.vllm_config)
        self._region_engine_id = region_engine_id(self.vllm_config)
        self._region = build_region(self.vllm_config, num_blocks)

        ctx = get_mp_context()
        self._dispatch_q = ctx.Queue()
        self._completion_q = ctx.Queue()
        self.encoder_channel = EncoderChannel(self._dispatch_q, self._completion_q)
        # EngineCore builds the scheduler after the executor, in this process.
        channel_slot.publish(self.encoder_channel)

    def _post_init_executor(self) -> None:
        self._spawn_encoder_proc()
        self._start_encoder_monitor()

    def _build_encoder_config(self) -> VllmConfig:
        """World size 1, eager, the encoder's compute dtype, vision-only load."""
        cfg = copy.deepcopy(self.vllm_config)
        pc = cfg.parallel_config
        pc.tensor_parallel_size = 1
        pc.pipeline_parallel_size = 1
        pc.prefill_context_parallel_size = 1
        pc.decode_context_parallel_size = 1
        pc.data_parallel_size = 1
        pc.data_parallel_size_local = 1
        pc.world_size = 1
        pc.distributed_executor_backend = "uni"
        pc.worker_cls = "auto"

        cfg.model_config.dtype = self._pe_config.torch_dtype
        cfg.model_config.enforce_eager = True
        cfg.compilation_config.mode = CompilationMode.NONE
        cfg.compilation_config.custom_ops = ["none"]

        mm_config = cfg.model_config.get_multimodal_config()
        assert mm_config is not None
        mm_config.mm_encoder_only = True

        cfg.kv_transfer_config = None
        cfg.ec_transfer_config = None
        return cfg

    def _spawn_encoder_proc(self) -> None:
        ctx = get_mp_context()
        ready_reader, ready_writer = ctx.Pipe(duplex=False)
        death_reader, death_writer = ctx.Pipe(duplex=False)

        proc = ctx.Process(
            target=encoder_main,
            name="SpyreEncoderProc",
            kwargs=dict(
                vllm_config=self._build_encoder_config(),
                distributed_init_method=get_distributed_init_method(
                    get_loopback_ip(), get_open_port()
                ),
                region_engine_id=self._region_engine_id,
                region_num_blocks=self._region_num_blocks,
                region_block_size_bytes=self._region_block_size_bytes,
                region_dtype=self.vllm_config.model_config.dtype,
                max_batch_embeds=self._max_batch_embeds,
                num_threads=encoder_cpu_share(
                    self.parallel_config.world_size, self._pe_config.encoder_num_threads
                ),
                dispatch_q=self._dispatch_q,
                completion_q=self._completion_q,
                ready_pipe=ready_writer,
                death_pipe=death_reader,
            ),
            daemon=True,
        )
        # A spawned child reads its environment at start(); the TP workers already have.
        with _patched_environ(_CPU_ENCODER_ENV):
            proc.start()
        ready_writer.close()
        death_reader.close()

        if not ready_reader.poll(timeout=_READY_TIMEOUT_S):
            proc.terminate()
            raise RuntimeError("Encoder process did not become ready in time.")
        msg = ready_reader.recv()
        ready_reader.close()
        if msg.get("status") != "READY":
            proc.terminate()
            raise RuntimeError(f"Encoder process failed to start: {msg.get('error')}")

        self._encoder_proc = proc
        self._encoder_death_writer = death_writer
        logger.info(
            "parallel-encoder: encoder process ready (device=%s, pid=%d)",
            self._pe_config.encoder_device,
            proc.pid,
        )

    def _start_encoder_monitor(self) -> None:
        proc = self._encoder_proc
        self_ref = weakref.ref(self)

        def monitor_encoder():
            multiprocessing.connection.wait([proc.sentinel])
            _self = self_ref()
            if not _self or getattr(_self, "shutting_down", False):
                return
            _self.is_failed = True
            logger.error(
                "Encoder process %s died unexpectedly (exit code %s), shutting down executor.",
                proc.name,
                proc.exitcode,
            )
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        Thread(target=monitor_encoder, daemon=True, name="EncoderProcMonitor").start()

    def shutdown(self) -> None:
        super().shutdown()
        death_writer = getattr(self, "_encoder_death_writer", None)
        if death_writer is not None:
            with contextlib.suppress(Exception):
                death_writer.close()
            self._encoder_death_writer = None
        proc = getattr(self, "_encoder_proc", None)
        if proc is not None and proc.is_alive():
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
        region = getattr(self, "_region", None)
        if region is not None:
            try:
                region.cleanup()
            except Exception:
                logger.debug("parallel-encoder: region cleanup failed", exc_info=True)
            self._region = None
