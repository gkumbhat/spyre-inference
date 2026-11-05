# As-Built Spec: In-Engine Parallel Encoder Executor for Multimodal Models

> Companion to `spec.md` (the original finalized requirements). This document
> records the design **as actually implemented and tuned**, including the
> performance learnings that changed several original decisions (async-scheduler
> base class, contiguous transport, decoupled encoder capacity, revised
> correctness bar). Where this conflicts with `spec.md`, this document is
> authoritative for the current code.

## 1. Context & Background
- **Problem:** Multimodal models have two parts — a vision encoder and an LLM
  decoder. They either run on the same GPU (a burst of encoder work steals
  compute from decode, spiking inter-token latency), or they are split across
  two separate vLLM instances ("disaggregated encoder" / EPD). The
  separate-instance path exists in-tree (`vllm/distributed/ec_transfer/`,
  `docs/features/disagg_encoder.md`) but is hard to deploy: it needs an external
  orchestrator, cross-instance transport, and producer/consumer connector
  wiring.
- **Goal:** Get the compute-isolation benefit of disaggregation **inside a
  single vLLM engine**, with no orchestrator, by adding an executor that spawns
  one extra process for the encoder and overlaps encoding with decoding.
- **User Persona:** End user running vLLM multimodal models (single-engine
  deployment) under **high multimodal concurrency / multi-image** load, where an
  inline encoder contends with decode.

## 2. Core Goal & Objectives
- **Primary Goal:** A new executor starts one additional Python process running
  only the vision encoder (vision tower + multimodal projector) on a dedicated
  device, in parallel with the decoder on the TP workers. A companion scheduler
  detects multimodal requests, dispatches encoding to that process
  **asynchronously** (continuing to decode other requests), and once encoding
  completes shares the projected embeddings to the decoder workers via a
  shared-memory region; prefill/decode then proceed.
- **Business Value:** Lower, more stable ITL and better throughput for
  multimodal inference **when the encoder is a contention bottleneck** (high
  multimodal concurrency), because the decoder GPUs no longer stall on encoder
  compute.
- **Regime caveat (learned):** At **low multimodal concurrency** the per-request
  encode round-trip (dispatch → encode → transport → resume) is not amortized
  by concurrent decode, so the inline path can be as fast or faster. This
  feature targets the encoder-contended regime.

## 3. As-Built Design Decisions

| Area | Decision |
|---|---|
| Executor | `ParallelEncoderExecutor(MultiprocExecutor)`. Creates the shared region + coordination queues in `_init_executor` (before workers spawn), spawns one encoder process in `_post_init_executor`, and monitors it. Selected via **dotted import path** on `--distributed-executor-backend`. |
| Scheduler | `ParallelEncoderScheduler(AsyncScheduler)` — **must** subclass `AsyncScheduler`, not `Scheduler`, or async scheduling is silently disabled (large ITL/throughput regression). Selected via `--scheduler-cls`. |
| Enable + tuning | Via `additional_config["parallel_encoder"]`: `enabled`, `encoder_device` (`cuda:<idx outside TP set>` or `cpu`), `encoder_cache_tokens` (encoder working-set size), `max_dispatch_batch_embeds`. No new CLI flags / arg-parser changes. |
| Encode/decode coordination | **Async, non-blocking**, over plain `multiprocessing` queues (dispatch + completion). Completions are drained at the top of `schedule()` (guarded: skipped when nothing is in flight). Not the `ec_connector` role machinery. |
| Embedding transport | `ECSharedRegion` (`/dev/shm` mmap), `cudaHostRegister`-pinned. **Contiguous** per-image block ranges (`RegionBlockAllocator`) so each image transfers as a **single** D2H (write) / H2D (read) copy — not thousands of per-token copies. |
| Multi-image dispatch | Dispatch **every** not-cached image in the decode window in one scheduler pass (mirrors vLLM's inline loop), so a request's images encode **concurrently**. Decode is truncated to stop before the first not-ready image. |
| Encoder capacity | **Decoupled** from `max_num_batched_tokens`. The region / `EncoderCacheManager` working set is sized for pipeline depth (default ~16 in-flight images; override via `encoder_cache_tokens`). Analog of a dedicated EPD encoder instance's `max_num_seqs`; critical for TTFT under multi-image load. |
| Encoder batching | The encoder process does continuous batching across dispatches/requests up to `max_dispatch_batch_embeds`, reusing `group_and_batch_mm_kwargs`. |
| Model split | Encoder process loads input processing + vision tower + projector (`mm_config.mm_encoder_only=True`), outputs final projected mm embeddings. Decoder workers merge them at placeholder positions (`embed_input_ids`) and run the LLM + sampling. **Current limitation:** the decoder still loads the *full* model (vision tower resident but unused); language-only decode is a follow-up. |
| Text-only requests | Bypass the encoder process and region entirely; identical to the `mp` executor. |
| Prefix / encoder cache | Shared-region entries allocated/refcounted/freed by `mm_hash` via `EncoderCacheManager` + `free_encoder_mm_hashes`; identical images reuse the same region bytes; region does not leak (coalescing free-range allocator). |
| Robustness | Encoder-process death is **fatal**: sentinel monitor → clear error + graceful engine shutdown (same as a dead TP worker). |
| Target model | `Mistral3ForConditionalGeneration` / Pixtral vision tower (e.g. a Mistral-Small-3.1 / Ministral-3 vision checkpoint). Serve with `--tokenizer-mode mistral` for consolidated Mistral checkpoints so `[IMG]` expands. |

## 4. Critical Implementation Learnings (why the above)

Each fixed a measured regression during bring-up:

1. **Subclass `AsyncScheduler`, not `Scheduler`.** A custom `scheduler_cls` that
   subclasses plain `Scheduler` loses the run-ahead output-placeholder machinery
   → async scheduling effectively off → the decode GPU idles during every
   schedule pass. Dominant ITL/throughput regression; invisible except via the
   (unconditional) custom-scheduler warning.
2. **Contiguous transport.** The region is a per-token block grid. Scattered
   per-token `swap_blocks_batch` copies were one to two orders of magnitude slower
   per image (per-descriptor overhead + host-memory contention) than a contiguous
   range moved as a single copy.
3. **Dispatch all of a request's in-window images at once.** Dispatching one per
   scheduler pass serialized a 4-image request into 4 sequential encode
   round-trips (~4× encode latency).
4. **Decouple encoder working-set size from `max_num_batched_tokens`.** Tied to
   the decode budget, the region held only ~2–3 large images → the encoder could
   not run ahead → high TTFT/E2E. Sizing it for ~16 in-flight images (like EPD's
   encoder `max_num_seqs`) is the TTFT/E2E fix.
5. **Transport touches host memory.** A GPU encoder writes D2H → region → H2D;
   concurrent write+read contend on host bandwidth. GPU→GPU (CUDA IPC / NCCL)
   would remove this (future work).

## 5. Scope Boundaries

### In Scope (implemented)
- [x] Executor + scheduler selected via dotted import paths; enabled/tuned via
      `additional_config`.
- [x] Encoder process loads vision-only (never the LM weights).
- [x] Encoder runs without TP, on a dedicated GPU (or CPU).
- [x] Decoder uses TP workers on GPU as today.
- [x] Prefix caching: `mm_hash` refcounted region reuse + decoder KV unchanged.
- [x] Reuse vLLM built-ins (`ECSharedRegion`, `EncoderCacheManager`,
      `group_and_batch_mm_kwargs`, `embed_multimodal`/`embed_input_ids`).
- [x] Async, non-blocking scheduling; concurrent multi-image encode.
- [x] Decoupled, configurable encoder working-set (pipeline depth).

### Non-Goals (Out of Scope)
- Accelerators other than CPU and GPU.
- Multi-node / NIXL / RDMA connectors.
- Multiple encoder replicas or encoder-side TP/PP.
- Bounded-queue backpressure / admission control beyond fatal-crash handling.
- GPU→GPU (CUDA IPC / NCCL) transport (future; would remove the host bounce).
- Language-only decoder load (future; recovers VRAM/KV vs the current full load).
- The cross-instance `ec_connector` EPD path (this is the in-engine alternative).

## 6. Components & Files (as-built)
- `vllm/config/parallel_encoder.py` — `ParallelEncoderConfig`.
- `vllm/v1/executor/parallel_encoder/`
  - `executor.py` — `ParallelEncoderExecutor` (region + queues + encoder proc + crash monitor).
  - `encoder_proc.py` — encoder child process (vision-only load, continuous-batching loop, region write).
  - `channel.py` — `EncoderChannel`, `EncodeItem`/`EncodeBatch`/`EncodeDone`.
  - `region.py` — region sizing (`resolve_encoder_cache_size`, `build_region`) + `RegionBlockAllocator` (contiguous ranges, coalescing).
  - `region_io.py` — `RegionWriter`/`RegionReader` (single-copy contiguous fast path; scattered fallback).
  - `profiling.py` — gated (`PARALLEL_ENCODER_PROFILE`) probes.
- `vllm/v1/core/sched/parallel_encoder_scheduler.py` — `ParallelEncoderScheduler`.
- Edits: `vllm/v1/core/sched/output.py` (`encoder_region_loads` field),
  `vllm/v1/engine/core.py` (attach channel to scheduler),
  `vllm/v1/worker/gpu/model_runner.py` + `vllm/v1/worker/gpu_model_runner.py`
  (region read before `_gather_mm_embeddings`; V2 is the active path).

## 7. Configuration (how to run)
```
vllm serve <mistral3-vision-checkpoint> --tokenizer-mode mistral \
  --distributed-executor-backend \
      vllm.v1.executor.parallel_encoder.executor.ParallelEncoderExecutor \
  --scheduler-cls \
      vllm.v1.core.sched.parallel_encoder_scheduler.ParallelEncoderScheduler \
  --additional-config '{"parallel_encoder":{"enabled":true,
      "encoder_device":"cuda:1","encoder_cache_tokens":48400}}'
```
`encoder_cache_tokens` ≈ `target_in_flight_images × tokens_per_image`; omit for
the auto default (~16 images). Confirm the base is correct via the startup log:
`ParallelEncoderScheduler active: base=AsyncScheduler, async_scheduling=True`.

## 8. Acceptance Criteria

**Functional**
- [ ] AC-1: Launch starts exactly one encoder process + the decoder TP workers.
- [ ] AC-2: The encoder process holds only vision tower + projector (no LM
      weights) — verifiable via loaded modules / process memory.
- [ ] AC-3: Decoder runs with `tensor_parallel_size > 1` unchanged; encoder uses
      no TP on a device disjoint from the decoder TP set.
- [ ] AC-4 (**revised**): **Logprob-close correctness.** With greedy sampling,
      generation stays logprob-close to the inline `mp` baseline
      (`check_logprobs_close`): at every mismatch position the top-logprob token
      sets match. Bitwise token-exactness is **not** achievable in bf16 across
      separate GPUs (vision-tower rounding amplified by greedy decoding).
      Optional embedding-parity diagnostic via `PARALLEL_ENCODER_VERIFY_ATOL`.
- [ ] AC-5: **Text-only no-regression** — text-only requests bypass the encoder
      and match the `mp` executor token-exactly.
- [ ] AC-6: **Prefix / encoder cache** — repeated identical image encoded once
      and reused; region freed after referencing requests finish; no leak;
      decoder KV prefix caching unaffected.
- [ ] AC-7: **Async overlap** — under concurrent load, one request's encode
      overlaps other requests' decode (encoder busy, decode not stalled).
- [ ] AC-8: **Fatal-crash handling** — encoder death → clear error + graceful
      shutdown (no hang on encoder-cache miss).

**Performance** (author-run; compared against **both** TP=2 inline **and** EPD on
the **same GPUs**, at high multimodal concurrency / multi-image)
- [ ] AC-9: ITL/TPOT and throughput competitive with EPD (same hardware, same
      eager encoder).
- [ ] AC-10: TTFT/E2E competitive with EPD; no regression vs inline TP=2 in the
      encoder-contended regime.
- [ ] AC-11: Low-concurrency behavior documented (inline may win there — expected).

## 9. Known Divergences & Follow-ups
- **Eager commit before KV alloc:** encoder dispatch/allocation happens inside
  `_try_schedule_encoder_inputs`, before `allocate_slots` confirms the request is
  schedulable (vLLM defers). Reconciles correctly but can waste encode under
  heavy preemption. Fix: defer commit / roll back on KV-alloc failure.
- **Per-step encoder compute budget not threaded across requests** (we return an
  empty `encoder_inputs_to_schedule`, so the base never commits our budget). The
  real bound is region cache space; acceptable for async, but not the intended
  semantics.
- **Decoder loads the full model** (vision unused) vs EPD's language-only decode.
- **Host-memory transport bounce** (see learning #5).
- **Single eager encoder** is a throughput ceiling (shared with EPD); replicas
  are out of scope.

## 10. Verification & Testing
- **Correctness (CI):** pytest asserting AC-4 (logprob-close vs inline `mp`) and
  AC-5 (text-only parity) on a fixed image+text set via the chat API
  (`tests/v1/e2e/test_parallel_encoder.py`). Unit tests for `RegionBlockAllocator`
  (contiguity, fragmentation fallback, coalescing) in
  `tests/v1/executor/test_parallel_encoder_region.py`.
- **Cache-lifecycle / crash tests:** AC-6 (reuse + freeing, no leak) and AC-8.
- **Performance (PR):** serving benchmark on the Mistral3 model at high
  concurrency / multi-image, reporting P50/P99 TPOT + ITL, TTFT, E2E, and
  throughput for inline TP=2 vs EPD vs parallel-encoder; include the
  `encoder_cache_tokens` used.
- Per AGENTS.md: human review of every line, model evals for output-affecting
  changes, and test commands + results in the PR description. Pure code-agent
  PRs are not allowed.
