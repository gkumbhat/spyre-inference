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
        self.image_waits: list[float] = []

    def record_step(self, sched_s: float, drain_s: float, flush_s: float, loads: int) -> None:
        self.steps += 1
        self.sched_s += sched_s
        self.drain_s += drain_s
        self.flush_s += flush_s
        self.max_sched_s = max(self.max_sched_s, sched_s)
        self.loads += loads

    def record_image_wait(self, wait_s: float) -> None:
        """Dispatch to completion for one image: how long its request waited on it."""
        self.image_waits.append(wait_s)

    def record_dispatch(self, items: int, nbytes: int) -> None:
        self.dispatch_events += 1
        self.dispatch_items += items
        self.dispatch_bytes += nbytes

    def maybe_log(self, inflight: int) -> None:
        now = time.monotonic()
        dt = now - self._last
        if dt < self._interval:
            return
        n = max(self.steps, 1)
        pe_overhead_ms = 1e3 * (self.drain_s + self.flush_s) / n
        waits = self.image_waits
        logger.info(
            "parallel-encoder PROFILE[sched] %.1fs, %d steps: "
            "schedule mean=%.3fms max=%.3fms | PE-overhead mean=%.3fms "
            "(drain=%.3fms flush=%.3fms) | dispatch %d ev / %d items / "
            "%.1f MB (%.2f MB/s) | region-loads=%d | images in flight=%d | "
            "image wait n=%d mean=%.2fs max=%.2fs",
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
            inflight,
            len(waits),
            sum(waits) / len(waits) if waits else 0.0,
            max(waits, default=0.0),
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
        # Dispatch time per in-flight image; filled only while profiling.
        self._dispatched_at: dict[str, float] = {}
        # Per-image wall-clock stages (dispatched, written, drained); profiling only.
        self._image_stages: dict[str, dict[str, float]] = {}
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

    def _num_partial_prefills(self) -> int:
        """Leave out requests parked on an image the encoder has not finished: they do
        no prefill work until it lands, and counting them would stop every new request
        (text-only included) at the default cap of one partial prefill."""
        return sum(
            request.num_computed_tokens < request.num_prompt_tokens
            and not self._waits_on_encoder(request)
            for request in self.running
        )

    def _waits_on_encoder(self, request: Request) -> bool:
        for feature in request.mm_features or ():
            pos = feature.mm_position
            if pos.offset + pos.length > request.num_computed_tokens:
                return feature.identifier in self._encoding_inflight
        return False

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
        output.encoder_region_loads = self._pending_loads_this_step
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
        self._prefetch_waiting()
        self._flush_dispatch()
        if prof.enabled:
            t3 = time.perf_counter()
            prof.record_step(
                sched_s=t3 - t0,
                drain_s=t1 - t0,
                flush_s=t3 - t2,
                loads=len(self._pending_loads_this_step),
            )
            prof.maybe_log(len(self._encoding_inflight))
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
            if (sent := self._dispatched_at.pop(done.mm_hash, None)) is not None:
                self._prof.record_image_wait(time.perf_counter() - sent)
            if (stages := self._image_stages.get(done.mm_hash)) is not None:
                stages["written"] = done.written_at or 0.0
                stages["drained"] = time.time()

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
        # Look ahead to every image not yet passed, not just those in this step's
        # window: Spyre caps a step at 512 tokens, so an image after a long text
        # prompt would otherwise reach the encoder only once that text has
        # prefilled, and its encode could not overlap its own request's prefill.
        hi_all = len(mm_features)

        # Dispatch every not-cached image (like vLLM's inline loop), so a
        # multi-image request encodes its images concurrently rather than one
        # blocking round-trip at a time. Decode can advance no further than the
        # first in-window image whose embedding is not yet ready (in-flight or
        # newly dispatched) — tracked as block_pos. Images past the window are only
        # dispatched: they never truncate decode and are not loaded until reached.
        block_pos: int | None = None

        for i in range(lo, hi_all):
            mm_feature = mm_features[i]
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier
            in_window = i < hi

            if item_identifier in scheduled_this_call:
                continue
            if not in_window and (
                item_identifier in self._encoding_inflight
                or self._region_alloc.is_allocated(item_identifier)
            ):
                # Already dispatched by an earlier look-ahead; nothing to do until
                # the window reaches it.
                continue

            # No chunking of a single mm item's decoder tokens: if the window
            # would only partially cover this item, decode must stop before it.
            if (
                in_window
                and self.scheduler_config.disable_chunked_mm_input
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
            if cached and not in_window:
                # Encoded and reachable by another request; registered as a referrer
                # above. Loaded when this request's window reaches it.
                continue
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
                        self._log_image_stages(item_identifier, request)
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
            # The step's encoder compute budget meters encoder work on the decoder
            # cards this step; a look-ahead image runs off them, so only cache space
            # limits it.
            compute_budget = encoder_compute_budget if in_window else num_encoder_embeds
            if not needs_reencode and not self.encoder_cache_manager.can_allocate(
                request, i, compute_budget, num_embeds_to_schedule
            ):
                # Encoder budget / cache space exhausted. This image (and any after
                # it) can't be dispatched now; decode stops here if it is in reach.
                if in_window and block_pos is None:
                    block_pos = start_pos
                break

            self._dispatch_item(request, i, allocate=not needs_reencode)
            scheduled_this_call.add(item_identifier)
            num_embeds_to_schedule += num_encoder_embeds
            if in_window:
                encoder_compute_budget -= num_encoder_embeds

            # Just dispatched -> not ready this step; decode stops before it.
            if in_window and block_pos is None:
                block_pos = start_pos

        if block_pos is not None:
            num_new_tokens = self._truncate_before(
                block_pos, num_computed_tokens, shift_computed_tokens
            )
        return [], num_new_tokens, encoder_compute_budget, [], []

    def _dispatch_item(self, request: Request, i: int, *, allocate: bool) -> None:
        """Reserve region blocks for ``request``'s image ``i`` and queue it for the
        encoder. The caller has already checked cache space with ``can_allocate``."""
        mm_feature = request.mm_features[i]
        mm_hash = mm_feature.identifier
        # can_allocate() may have evicted freeable entries; reclaim their region
        # blocks now so the free-list stays in sync this pass.
        self._reconcile_region_frees()
        block_ids = self._region_alloc.alloc(mm_hash, mm_feature.mm_position.get_num_embeds())
        if allocate:
            self.encoder_cache_manager.allocate(request, i)
        self._encoding_inflight.add(mm_hash)
        if self._prof.enabled:
            self._dispatched_at[mm_hash] = time.perf_counter()
            self._image_stages[mm_hash] = {"dispatched": time.time()}
        self._dispatch_this_step.append(
            EncodeItem(
                mm_hash=mm_hash,
                modality=mm_feature.modality,
                block_ids=block_ids,
                data=mm_feature.data,  # ty: ignore[invalid-argument-type]
            )
        )

    def _log_image_stages(self, mm_hash: str, request: Request) -> None:
        """One line per image when its region load is emitted: encode (dispatch to
        written), notify (written to drained), and how long the ready image waited for
        its request's window (drained to load)."""
        stages = self._image_stages.pop(mm_hash, None)
        if stages is None or "drained" not in stages:
            return
        now = time.time()
        logger.info(
            "parallel-encoder PROFILE[image] %s req=%s encode=%.3fs notify=%.3fs "
            "wait_for_window=%.3fs since_arrival=%.3fs computed=%d",
            mm_hash[:12],
            request.request_id[:12],
            stages["written"] - stages["dispatched"] if stages["written"] else -1.0,
            stages["drained"] - stages["written"] if stages["written"] else -1.0,
            now - stages["drained"],
            now - request.arrival_time,
            request.num_computed_tokens,
        )

    def _prefetch_waiting(self) -> None:
        """Dispatch the images of requests still waiting for admission.

        Spyre admits one partial prefill at a time by default, so a waiting request
        would otherwise reach the encoder only once every request ahead of it has
        prefilled. Its image encodes meanwhile and is loaded when its window reaches
        it. Arrival order; stops at the in-flight cap or when cache space runs out.
        """
        for request in self.waiting:
            for i, feature in enumerate(request.mm_features or ()):
                if len(self._encoding_inflight) >= self._max_inflight_encode_images:
                    return
                mm_hash = feature.identifier
                if mm_hash in self._encoding_inflight or self._region_alloc.is_allocated(mm_hash):
                    # Dispatched or encoded already, possibly for another request:
                    # register this one as a referrer so eviction leaves it alone.
                    self.encoder_cache_manager.check_and_update_cache(request, i)
                    continue
                embeds = feature.mm_position.get_num_embeds()
                if not self.encoder_cache_manager.can_allocate(request, i, embeds, 0):
                    return
                self._dispatch_item(request, i, allocate=True)

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
