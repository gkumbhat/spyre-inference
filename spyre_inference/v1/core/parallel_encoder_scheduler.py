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

# SPDX-License-Identifier: Apache-2.0

"""Scheduler for the in-engine parallel encoder.

Dispatches image encoding to the encoder process instead of the decoder workers and
keeps scheduling other requests meanwhile. A request's tokens stop just before the
first image whose embedding is not ready; once the encoder reports completion, the
workers load that embedding from the shared region (``encoder_region_loads``) and the
request proceeds through it.

Builds on ``TorchSpyreScheduler``, not ``AsyncScheduler``: CpuPlatform turns async
scheduling off, so the overlap comes from the encoder running in its own process.
Region block allocation is slaved to ``EncoderCacheManager``, which stays the single
eviction authority.
"""

import os
import time

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from spyre_inference.config.parallel_encoder import ParallelEncoderConfig
from spyre_inference.v1.core.scheduler import TorchSpyreScheduler
from spyre_inference.v1.executor.parallel_encoder import channel_slot
from spyre_inference.v1.executor.parallel_encoder.channel import EncodeBatch, EncodeItem
from spyre_inference.v1.executor.parallel_encoder.profiling import PROFILE_ENABLED, mm_item_nbytes
from spyre_inference.v1.executor.parallel_encoder.region import (
    RegionBlockAllocator,
    resolve_encoder_cache_size,
)

logger = init_logger(__name__)


class _SchedProfiler:
    """Gated per-step timing for the parallel-encoder scheduler overhead.

    Measures how much of the (single-threaded) engine-core ``schedule()`` wall
    clock is spent on parallel-encoder coordination vs the base scheduler, plus
    the volume/rate of mm data dispatched (the pixel_values pickled by the mp
    queue feeder thread). Zero-cost unless PARALLEL_ENCODER_PROFILE is set.
    """

    def __init__(self) -> None:
        self.enabled = PROFILE_ENABLED
        self._interval = float(os.environ.get("PARALLEL_ENCODER_PROFILE_INTERVAL", "5"))
        self._last = time.monotonic()
        self._reset()

    def _reset(self) -> None:
        self.steps = 0
        self.sched_s = 0.0
        self.drain_s = 0.0
        self.flush_s = 0.0
        self.max_sched_s = 0.0
        self.dispatch_events = 0
        self.dispatch_items = 0
        self.dispatch_bytes = 0
        self.loads = 0

    def record_step(self, sched_s: float, drain_s: float, flush_s: float, loads: int) -> None:
        self.steps += 1
        self.sched_s += sched_s
        self.drain_s += drain_s
        self.flush_s += flush_s
        self.max_sched_s = max(self.max_sched_s, sched_s)
        self.loads += loads

    def record_dispatch(self, items: int, nbytes: int) -> None:
        self.dispatch_events += 1
        self.dispatch_items += items
        self.dispatch_bytes += nbytes

    def maybe_log(self) -> None:
        now = time.monotonic()
        dt = now - self._last
        if dt < self._interval:
            return
        n = max(self.steps, 1)
        pe_overhead_ms = 1e3 * (self.drain_s + self.flush_s) / n
        logger.info(
            "parallel-encoder PROFILE[sched] %.1fs, %d steps: "
            "schedule mean=%.3fms max=%.3fms | PE-overhead mean=%.3fms "
            "(drain=%.3fms flush=%.3fms) | dispatch %d ev / %d items / "
            "%.1f MB (%.2f MB/s) | region-loads=%d",
            dt,
            self.steps,
            1e3 * self.sched_s / n,
            1e3 * self.max_sched_s,
            pe_overhead_ms,
            1e3 * self.drain_s / n,
            1e3 * self.flush_s / n,
            self.dispatch_events,
            self.dispatch_items,
            self.dispatch_bytes / 1e6,
            (self.dispatch_bytes / 1e6) / dt if dt else 0.0,
            self.loads,
        )
        self._last = now
        self._reset()


# Default encoder-admission backpressure cap (images concurrently in the
# encoder queue) when not configured. Balances keeping the encoder fed
# (pipeline depth) against clogging decode KV with encode-waiting requests.
_DEFAULT_MAX_INFLIGHT_ENCODE_IMAGES = 16


class ParallelEncoderScheduler(TorchSpyreScheduler):
    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)

        if self.is_encoder_decoder:
            raise ValueError("ParallelEncoderScheduler does not support encoder-decoder models.")
        if self.ec_connector is not None:
            raise ValueError(
                "ParallelEncoderScheduler is incompatible with an EC transfer connector; "
                "they are two different disaggregation mechanisms."
            )

        # Size the encoder working set (region + cache manager) for the encoder
        # pipeline, decoupled from the decode-side max_num_batched_tokens the
        # base scheduler used. Must match the executor's region num_blocks and
        # the decoder worker's attach size — all derive from the same helper.
        pe_config = ParallelEncoderConfig.from_vllm_config(vllm_config)
        mm_budget = MultiModalBudget(vllm_config, MULTIMODAL_REGISTRY)
        cache_size = resolve_encoder_cache_size(mm_budget, pe_config)
        if cache_size != self.encoder_cache_manager.cache_size:
            self.encoder_cache_manager = type(self.encoder_cache_manager).create_manager(
                cache_size=cache_size, vllm_config=vllm_config
            )
        self._region_alloc = RegionBlockAllocator(num_blocks=cache_size)

        # Encoder-admission backpressure cap (see config docstring). Bounds
        # images concurrently in the encoder queue; brand-new mm requests wait
        # (no KV) once this is reached, preventing decode-KV clog under encoder
        # saturation.
        self._max_inflight_encode_images = (
            pe_config.max_inflight_encode_images
            if pe_config and pe_config.max_inflight_encode_images
            else _DEFAULT_MAX_INFLIGHT_ENCODE_IMAGES
        )

        # mm_hashes dispatched to the encoder but not yet reported complete.
        self._encoding_inflight: set[str] = set()
        # mm_hashes whose embedding is already loaded into the worker caches
        # (emit a region load exactly once, on first consumption).
        self._loaded_by_workers: set[str] = set()

        # Per-step scratch, reset at the top of schedule().
        self._dispatch_this_step: list[EncodeItem] = []
        self._pending_loads_this_step: dict[str, list[int]] = {}

        self._encoder_channel = channel_slot.take()
        self._prof = _SchedProfiler()
        logger.info(
            "ParallelEncoderScheduler active: encoder working set %d embeddings, "
            "up to %d images in flight",
            cache_size,
            self._max_inflight_encode_images,
        )

    def schedule(self, *args, **kwargs) -> SchedulerOutput:
        prof = self._prof
        t0 = time.perf_counter() if prof.enabled else 0.0
        self._drain_completions()
        t1 = time.perf_counter() if prof.enabled else 0.0
        self._dispatch_this_step = []
        self._pending_loads_this_step = {}

        output = super().schedule(*args, **kwargs)

        # SchedulerOutput has no field for this; it reaches the workers pickled, and
        # TorchSpyreModelRunner reads it with getattr.
        output.encoder_region_loads = self._pending_loads_this_step  # type: ignore[attr-defined]
        # Free region blocks for anything the cache manager evicted this step.
        # A hash can be evicted AND re-dispatched within one pass (the same
        # image referenced by a new request after eviction); the eviction is
        # reported here but the re-dispatch already re-allocated fresh region
        # blocks under the same mm_hash. Freeing then would strand the live
        # allocation (KeyError on its later region load), so skip any hash the
        # cache manager still references. Always drop it from the worker-loaded
        # set: the worker frees the old embedding on this same free list, so a
        # re-encoded copy must be reloaded before its next consumption.
        for mm_hash in output.free_encoder_mm_hashes:
            self._loaded_by_workers.discard(mm_hash)
            if mm_hash not in self.encoder_cache_manager.cached:
                self._region_alloc.free(mm_hash)

        t2 = time.perf_counter() if prof.enabled else 0.0
        self._flush_dispatch()
        if prof.enabled:
            t3 = time.perf_counter()
            prof.record_step(
                sched_s=t3 - t0,
                drain_s=t1 - t0,
                flush_s=t3 - t2,
                loads=len(self._pending_loads_this_step),
            )
            prof.maybe_log()
        return output

    def _drain_completions(self) -> None:
        # Nothing dispatched is outstanding -> no completion can be pending, so
        # skip the queue poll. This keeps the completion IPC off the decode
        # critical path on the (vast majority of) steps with no encode in
        # flight, avoiding a per-step ITL tax.
        if not self._encoding_inflight:
            return
        for done in self._encoder_channel.drain_completions():
            if not done.ok:
                # Encode failure is fatal (matches dead-worker semantics); the
                # encoder process will also exit, tripping the executor's
                # sentinel monitor.
                raise RuntimeError(
                    f"parallel-encoder: encoding failed for {done.mm_hash}: {done.error}"
                )
            self._encoding_inflight.discard(done.mm_hash)

    def _flush_dispatch(self) -> None:
        if not self._dispatch_this_step:
            return
        if self._prof.enabled:
            nbytes = sum(mm_item_nbytes(it.data) for it in self._dispatch_this_step)
            self._prof.record_dispatch(len(self._dispatch_this_step), nbytes)
        self._encoder_channel.dispatch(EncodeBatch(items=self._dispatch_this_step))
        self._dispatch_this_step = []

    # -- the core override -------------------------------------------------

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int], list[int]]:
        """Async variant of the base method.

        Returns an EMPTY ``encoder_inputs_to_schedule`` in all cases so the
        decoder workers never run vision compute; images are dispatched to the
        encoder process instead. ``num_new_tokens`` is truncated to stop before
        any image whose embedding is not yet ready.
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, [], []

        # Encoder-admission backpressure: when the encoder queue is deep, don't
        # admit a brand-new multimodal request. It would prefill its text and
        # then park, holding decode KV while the saturated encoder catches up
        # (KV clog: many "running" reqs, few tokens/s, KV ~100%). Returning 0
        # new tokens leaves it in WAITING with no KV allocated; already-running
        # requests keep decoding. This is the EPD encoder-max_num_seqs analog.
        # Gate on num_computed_tokens == 0 so preempted/prefix-cached resumes
        # (already holding KV) are not throttled.
        if (
            num_computed_tokens == 0
            and len(self._encoding_inflight) >= self._max_inflight_encode_images
        ):
            return [], 0, encoder_compute_budget, [], []

        mm_features = request.mm_features
        assert mm_features is not None and len(mm_features) > 0

        scheduled_this_call: set[str] = set()
        num_embeds_to_schedule = 0

        decode_end = num_computed_tokens + num_new_tokens + shift_computed_tokens
        lo, hi = get_mm_features_in_window(mm_features, start=num_computed_tokens, end=decode_end)

        # Dispatch every not-cached image in the window (like vLLM's inline
        # loop), so a multi-image request encodes its images concurrently
        # rather than one blocking round-trip at a time. Decode can advance no
        # further than the first image whose embedding is not yet ready
        # (in-flight or newly dispatched) — tracked as block_pos.
        block_pos: int | None = None

        for i in range(lo, hi):
            mm_feature = mm_features[i]
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            if item_identifier in scheduled_this_call:
                continue

            # No chunking of a single mm item's decoder tokens: if the window
            # would only partially cover this item, decode must stop before it.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and decode_end < (start_pos + num_encoder_tokens)
                and block_pos is None
            ):
                block_pos = start_pos

            # Register this request as a referrer if the item is already known
            # to the cache manager (dispatched earlier, possibly by another
            # request). This keeps refcounts correct for cross-request reuse.
            cached = self.encoder_cache_manager.check_and_update_cache(request, i)
            needs_reencode = False
            if cached:
                if item_identifier in self._encoding_inflight:
                    # Allocated but not ready — decode must stop before it.
                    if block_pos is None:
                        block_pos = start_pos
                    continue
                if self._region_alloc.is_allocated(item_identifier):
                    # Ready in the region and reachable this step — ensure the
                    # workers load it (once).
                    if block_pos is None and item_identifier not in self._loaded_by_workers:
                        self._pending_loads_this_step[item_identifier] = (
                            self._region_alloc.block_ids(item_identifier)
                        )
                        self._loaded_by_workers.add(item_identifier)
                    continue
                # Fail-safe: the cache manager references this item but its
                # region blocks are gone and no worker holds it. The free-path
                # guards should make this unreachable; if it happens, re-encode
                # rather than KeyError. Its cache slot is already reserved, so
                # re-dispatch without a second reservation.
                logger.warning(
                    "parallel-encoder: cached mm_hash %s missing from region; "
                    "re-encoding (this should not happen).",
                    item_identifier,
                )
                needs_reencode = True

            # Not cached — dispatch it to the encoder. (needs_reencode skips the
            # cache-manager reservation/eviction, which already happened.)
            if not needs_reencode and not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # Encoder budget / cache space exhausted this step. This image
                # (and any after it) can't be dispatched now; decode stops here.
                if block_pos is None:
                    block_pos = start_pos
                break

            # can_allocate() may have evicted freeable entries; reclaim their
            # region blocks now so the free-list stays in sync this pass.
            self._reconcile_region_frees()

            block_ids = self._region_alloc.alloc(item_identifier, num_encoder_embeds)
            if not needs_reencode:
                self.encoder_cache_manager.allocate(request, i)
            self._encoding_inflight.add(item_identifier)
            self._dispatch_this_step.append(
                EncodeItem(
                    mm_hash=item_identifier,
                    modality=mm_feature.modality,
                    block_ids=block_ids,
                    data=mm_feature.data,
                )
            )
            scheduled_this_call.add(item_identifier)
            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds

            # Just dispatched -> not ready this step; decode stops before it.
            if block_pos is None:
                block_pos = start_pos

        if block_pos is not None:
            num_new_tokens = self._truncate_before(
                block_pos, num_computed_tokens, shift_computed_tokens
            )
        return [], num_new_tokens, encoder_compute_budget, [], []

    def _reconcile_region_frees(self) -> None:
        """Free region blocks for mm_hashes the cache manager has already
        evicted this pass, without consuming its ``freed`` list.

        ``EncoderCacheManager.get_freed_mm_hashes`` (consumed by the base
        scheduler into ``SchedulerOutput.free_encoder_mm_hashes``) clears the
        list, so we only *peek* here; ``free`` is idempotent, so the
        post-schedule pass over ``free_encoder_mm_hashes`` harmlessly repeats
        these frees.

        Skip any hash the cache manager references again: ``can_allocate`` may
        have evicted a hash earlier this pass that a later dispatch then
        re-allocated under the same mm_hash. Its region blocks now belong to the
        live allocation; freeing them here would corrupt it.
        """
        for mm_hash in self.encoder_cache_manager.freed:
            if mm_hash in self.encoder_cache_manager.cached:
                continue
            self._region_alloc.free(mm_hash)
            self._loaded_by_workers.discard(mm_hash)

    @staticmethod
    def _truncate_before(
        start_pos: int, num_computed_tokens: int, shift_computed_tokens: int
    ) -> int:
        """Schedule decoder tokens only up to just before an image.

        Mirrors the base scheduler's truncation math. Returns 0 when prefix
        caching already advanced past the image start (the image's embedding
        is still required, so no decode can proceed this step).
        """
        if num_computed_tokens + shift_computed_tokens < start_pos:
            return start_pos - (num_computed_tokens + shift_computed_tokens)
        return 0
