# Gemma 4 vision tower: execution map

Where each part of the Gemma 4 multimodal (`Gemma4ForConditionalGeneration`) image
path runs — **host vs Spyre**, **dtype**, and **eager vs compiled** — as implemented
by `spyre_inference/multimodal/gemma4_vision.py`.

!!! note "Status"
    The pipeline below runs end-to-end, and the encoder matches a stock CPU fp32
    reference to **cosine 0.9959** (2 layers, real 26B-A4B vision config) after two
    correctness fixes described under [What the comparison
    caught](#what-the-comparison-caught): host-side weight padding and an fp32 rope
    frequency recompute. End-to-end on a real ChartQA image it now describes the chart
    and its title accurately (before those fixes it hallucinated unrelated content),
    so the vision path is functionally correct; a full-length generation confirming an
    exact benchmark answer has not been run yet.

## dtype

The whole model runs in **bfloat16**, not the platform's usual float16:
`TorchSpyrePlatform._default_dtype` selects it for a Gemma 4 config carrying a
`vision_config`/`audio_config`, because Gemma 4 overflows fp16's range (`inf` → NaN
end-to-end). The choice is recorded on the `ModelConfig` so it survives
`VllmConfig.with_hf_config`, which re-enters the hook with the bare text config when
the decoder is built.

Both dtypes are 2 bytes, so every 64-element stick-alignment constant in the plugin is
unaffected by the choice.

Two places deliberately leave bf16:

| Where | dtype | Why |
|---|---|---|
| `Gemma4VisionPooler` output | **fp32** | Stock behaviour: its `sqrt(hidden_size)` scaling overflows fp16. Stays fp32 through the `standardize` affine. |
| rope cos/sin table construction | **fp32 → bf16** | Trig is computed exactly on the host; only the bf16 result reaches the device. |

Notably **absent**: any fp32 promotion inside RMSNorm. torch-spyre does not support
dtype promotion (see `custom_ops/rms_norm.py`), and an on-device bf16→fp32→bf16 round
trip also leaves the result in a stick-tiling state a later eager elementwise op
cannot broadcast against. Both `Gemma4RMSNorm` (patched) and `_padded_rms_norm`
therefore reduce natively in bf16 — the same trade `SpyreRMSNorm` /
`SpyreGemmaRMSNorm` already accept, with the same expectation of small numerical
differences from upstream.

## Host vs Spyre, step by step

In `Gemma4ForConditionalGeneration._process_image_input` order, for one image
(2520 patches → right-padded to 2560 → pooled to 280 soft tokens):

| # | Step | Device | dtype | Notes |
|---|---|---|---|---|
| 1 | `SpyreModelWrapper.embed_multimodal` input conversion | → **Spyre** | bf16 | Float multimodal inputs (`pixel_values`) moved and cast; integer `pixel_position_ids` stay on host. |
| 2 | `patch_embedder.input_proj` (768→1152) + `2*(x-0.5)` scaling | **Spyre** | bf16 | Plain `nn.Linear`; no conv, so no im2col problem. |
| 3 | `patch_embedder._position_embeddings` (`F.embedding` XY gather + `-1` padding mask) | **host** | bf16 | Patched. Integer-gather doctrine, as for Pixtral's position lookup. Result uploaded and summed on Spyre. |
| 4 | rope cos/sin table (two-axis, padded layout) | **host** | fp32 → bf16 | Built once per image, then uploaded. |
| 5 | key-validity attention mask | **host** | bool | Uploaded lazily inside `padded_sdpa`, cached on the mask object (it is O(L²)). |
| 6 | one-time weight surgery: head-dim padding 72→128, MLP padding 4304→4352 | **host → Spyre** | bf16 | On first forward per layer; scratch tensors are built on host then moved. Idempotent. |
| 7 | **per encoder layer × 27** — see below | **Spyre** | bf16 | The whole transformer body. |
| 8 | `pooler` (masked_fill, `one_hot`, weighted matmul, `sqrt(hidden)` scale) | **host** | → **fp32** | Patched. `aten::masked_fill_` has no Spyre kernel; the rest is integer geometry. Reduces 2520→280, so the small side crosses back. |
| 9 | `pooled[valid_mask]` boolean row select | **host** | fp32 | `aten::index.Tensor_out` has no Spyre kernel — the reason step 8 returns host tensors. |
| 10 | `standardize`: `(x - std_bias) * std_scale` | **host** | fp32 | `place_vision_tail_on_cpu` moves the two buffers to host to match. |
| 11 | `cat(...).to(model_dtype)` | **host** | bf16 | |
| 12 | `embed_vision` (`Gemma4MultimodalEmbedder`: unscaled RMSNorm + Linear 1152→text hidden) | **host** | bf16 | Module moved to host. Cheap end of the tower, and its output feeds a host-side merge anyway. |
| 13 | `SpyreModelWrapper.embed_input_ids` merge into text embeddings | **host** → Spyre | bf16 | Image rows scattered on host (`_index_put_impl_` is unimplemented on Spyre), then the merged embedding table is uploaded for the decoder. |

This host/Spyre split matches the one hf-adapters#495 documents for the same tower:
*"integer position lookup, spatial pooling, and the text-space projector stay on CPU"*.

### Inside one encoder layer (all Spyre, all bf16)

| Op | Notes |
|---|---|
| `input_layernorm` (`Gemma4RMSNorm`, patched) | bf16 reduction, `rsqrt` instead of `torch.pow(x, -0.5)` (no `pow` lowering in this fused context). |
| `q_proj` / `k_proj` | Padded 72→128 with a quarter-interleave that groups each rope axis's halves, so one rotation covers both axes. |
| `v_proj` | Padded 72→128 by plain end-padding (no rope, so no interleave). |
| `q_norm` / `k_norm` / `v_norm` (`_padded_rms_norm`) | Variance rescaled by `padded/orig` so the zero lanes don't deflate the denominator. `v_norm` is unscaled (stock). |
| rope (`_apply_rope`) | `x*cos + cat([x2, x1])*sin`, i.e. `rotate_half` by **slicing**, not Pixtral's matmul. Legal here because the padded head_dim makes each half exactly one 64-element stick; Pixtral's 64-wide heads give 32-wide halves and need the matmul instead. |
| attention (`padded_sdpa`, reused from `multimodal/pixtral.py`) | Patch axis padded to the 64 stick, padded keys masked to `-inf`, padded queries cropped. `scale=1.0` (stock Gemma 4 uses an explicit unit scale, not `head_dim**-0.5`). |
| `o_proj` | Input-padded to match the padded per-head width. |
| `post_attention_layernorm` → residual | Sandwich norm: normalise the attention output *before* adding the residual. |
| `pre_feedforward_layernorm` → `mlp` → `post_feedforward_layernorm` → residual | MLP is GELU-tanh gated; intermediate padded 4304→4352 by zero-extending gate/up outputs and down-projection inputs (verified numerically transparent). |

## Eager vs compiled

| Portion | Mode |
|---|---|
| Vision tower (steps 2–7) | **Eager** at the Python level — no `torch.compile` anywhere in `gemma4_vision.py`. |
| Host portions (steps 3–5, 8–13) | **Eager** plain PyTorch on CPU. |
| Text decoder | **Compiled** — the run is `STOCK_TORCH_COMPILE`, one transformer block per graph. |
| Decoder attention kernels | **Pre-recorded during warmup** (`SpyreAttnBucketer.variants()`), not compiled lazily in the serving path. |

Two things are easy to get wrong here:

- **vLLM does not compile the multimodal encoder even in a compiled run.**
  `compilation_config.compile_mm_encoder` is `False`, so `STOCK_TORCH_COMPILE` applies
  to the decoder only. The vision tower would be eager regardless of what this module
  does.
- **"Eager" does not mean "uninterpreted".** torch-spyre dispatches each individual
  aten op on a Spyre tensor through its own per-op Inductor kernel, which is why
  `InductorError`s appear with no `torch.compile` anywhere in the traceback. So the
  vision tower is many small separately-compiled kernels rather than one graph — and
  the boundaries between them are where several of this tower's layout failures came
  from, because adjacent kernels can disagree about a tensor's device tiling.

Wrapping the per-layer forward in one `torch.compile(fullgraph=True)` graph was tried
(it is what hf-adapters#495 does, since that stack compiles a parameter-explicit
executor per structural layer class). It resolved one layout error class and
introduced another (`coarse_tile: hint_id … appears in both group 0 and group 1`), so
the eager path is what is implemented here. If the remaining numerical divergence
turns out to be a per-kernel tiling disagreement, revisiting this is the natural next
lever.

## Comparison with the hf-adapters reference

Compared against `hf_adapters/hf_gemma4_vision.py` and the Gemma 4 VLM section of
ARCHITECTURE.md on `torch-spyre/hf-adapters#495`.

### Matches

The host/Spyre boundary agrees on every point the reference states: dense encoder
blocks on Spyre; integer XY position lookup, spatial pooling, and the text-space
`embed_vision` projector on host; the 2520→2560 right-pad so the score matmul has no
ragged final stick; padded key columns masked once; fp32 output standardization on
host with the pooler. The numeric adaptations agree too — head_dim 64/72 → 128, the
Q/K channel and norm-weight rearrangement that lets one rotation serve both rope axes,
padded Q/K/V RMSNorm preserving the native-width denominator, the explicit attention
scale of 1.0, and the 4304→4352 MLP padding. E2B's clipping bounds are preserved for
free here, since we call the stock `Gemma4ClippableLinear` module rather than
reimplementing it.

### Where we do *less* on the host (keep)

| | Reference | Here |
|---|---|---|
| Patch projection `input_proj` | **host** (`pixel_values.to("cpu")` before `patch_embedder`) | **Spyre** |

Only the integer position gather needs the host; the projection itself is a plain
GEMM. Worth keeping.

### Where we still do *more* on the host than necessary

1. **The attention mask is O(L²) here and O(L) in the reference.** `_build_attention_mask`
   there returns `[B, 1, 1, padded_len]` and lets SDPA broadcast it over queries — about
   5 KB. We reuse Pixtral's `_padded_attn_mask`, which materialises `[B, 1, L, L]`:
   at L=2560 in bf16 that is **~13 MB assembled on host and uploaded per image**, to
   carry information that is one bit per key. This is the clearest remaining win —
   it costs host time, H2D bandwidth, and device memory. It needs a `padded_sdpa`
   variant (or a Gemma-4-local SDPA) that accepts a key-only mask; the shared helper
   currently reshapes the mask to `[seq, seq]`.
2. **Padded rows get a real rotation rather than the identity.** The reference appends
   identity rope matrices for the pad region; we clamp `position_ids == -1` to 0, so
   pad rows are rotated as if at position 0. Harmless today (those keys are masked and
   those query rows are cropped), but it is extra work and a latent trap if the mask
   ever regresses.

### Differences that are *not* host/device, but matter

3. **RMSNorm reduction precision.** The reference's `_padded_rms_norm` computes in fp32
   and casts back; ours reduces natively in bf16, because a full-width fp32 round trip
   on device leaves a tiling state the next eager op cannot broadcast against. Note
   ARCHITECTURE.md describes a middle ground we have not tried: promote only the
   mean/variance **reduction** to fp32 while keeping the affine multiply in bf16 (the
   `[…, 1]` variance is tiny, so it may not trip the layout problem the full-width
   tensor did). That is the first lever if more encoder accuracy is needed.
4. **Compiled blocks vs eager ops.** The reference compiles each block as a
   parameter-explicit executor shared across structurally identical layers — a large
   cold-start win, and it gives each graph one consistent layout. We run eager, so
   torch-spyre compiles each op separately and adjacent kernels can disagree about
   tiling. Revisiting this is the other lever for the residual accuracy gap.
5. **The reference `.clone()`s each block's output** before feeding the next layer,
   which breaks layout aliasing between layers. We do not; worth trying if per-layer
   error accumulation shows up.

### What the comparison caught

The reference does all of its weight surgery in `prepare_for_spyre(model)` **before the
model is ever moved to the device**. We cannot copy that structure, because vLLM moves
the model before our patches run — and doing the same padding on device-resident
weights turned out to produce silently wrong weights: encoder cosine **0.50** versus
**0.9996** for the identical surgery done host-side. Individual slice-assignments are
fine on device, so only the composite is affected and nothing raised. Reading the
reference's ordering is what prompted checking it.

Fixing that, plus recomputing the rope frequencies in fp32 (`model.to(bfloat16)` had
downcast `rotary_emb.inv_freq`, worth up to 0.10 absolute on the cos/sin table), moved
the whole encoder from cosine **0.252 → 0.9959** against the stock CPU reference.

## Constraints asserted loudly

- `num_key_value_heads != num_attention_heads` (vision GQA) → `NotImplementedError`.
- A scaled `v_norm` → `NotImplementedError` (stock Gemma 4 leaves it unscaled).
- A batch mixing different valid-patch counts per row would need `padded_sdpa`
  extended to a per-row mask; the current shared mask is correct for the
  uniform-padding case it was built against.
