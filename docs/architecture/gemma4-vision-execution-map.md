# Gemma 4 vision tower: execution map

Where each part of the Gemma 4 multimodal (`Gemma4ForConditionalGeneration`) image
path runs — **host vs Spyre**, **dtype**, and **eager vs compiled** — as implemented
by `spyre_inference/multimodal/gemma4_vision.py`.

!!! warning "Status: executes end-to-end, not yet numerically correct"
    The pipeline below runs to completion and produces fluent text, but the encoder's
    device output currently diverges from a CPU reference (**cosine ≈ 0.25**, finite
    values), so generations are effectively blind to the image. The same code is
    numerically exact on CPU in fp32 (cosine 1.000000) and fine on CPU in bf16
    (0.999680), which localises the divergence to the device lowering rather than to
    this placement or to precision. Treat this document as a map of the *implemented*
    placement, not of a validated-correct one.

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

## Constraints asserted loudly

- `num_key_value_heads != num_attention_heads` (vision GQA) → `NotImplementedError`.
- A scaled `v_norm` → `NotImplementedError` (stock Gemma 4 leaves it unscaled).
- A batch mixing different valid-patch counts per row would need `padded_sdpa`
  extended to a per-row mask; the current shared mask is correct for the
  uniform-padding case it was built against.
