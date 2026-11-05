"""Time the CPU vision tower alone on max-size images, sweeping batch size and threads.

Loads the model as the encoder process does (no Spyre plugin, mm_encoder_only) and times
embed_multimodal per call.
Run with: VLLM_PLUGINS= TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_TARGET_DEVICE=cpu python -I ...
Usage: encoder_microbench.py MODEL THREADS[,THREADS...] BATCH[,BATCH...]
"""

import sys
import time

import torch
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.utils import group_and_batch_mm_kwargs
from vllm.v1.worker.cpu_model_runner import _set_torch_accelerator_to_noop
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

model_name = sys.argv[1]
threads = [int(t) for t in sys.argv[2].split(",")]
batches = [int(b) for b in sys.argv[3].split(",")]

_set_torch_accelerator_to_noop()
cfg = EngineArgs(
    model=model_name,
    config_format="mistral",
    tokenizer_mode="mistral",
    load_format="mistral",
    max_model_len=32768,
    dtype="bfloat16",
    enforce_eager=True,
    limit_mm_per_prompt={"image": max(batches)},
    mm_encoder_only=True,
).create_engine_config()

with set_current_vllm_config(cfg):
    init_worker_distributed_environment(cfg, 0, "tcp://127.0.0.1:29541", 0, "gloo")
    model = get_model(vllm_config=cfg)

# The inputs vLLM profiles the encoder with: max-size images (Pixtral resizes to its
# 1540px cap, so this is the same item a 2048x2048 request becomes).
budget = MultiModalBudget(cfg, MULTIMODAL_REGISTRY)
items = budget.get_dummy_encoder_profile_inputs("image", max(batches))
rows = budget.mm_max_toks_per_item["image"]
print(f"one max-size image -> {rows} embedding rows", flush=True)


@torch.inference_mode()
def encode(batch_items) -> None:
    with set_current_vllm_config(cfg):
        for _m, _n, kwargs in group_and_batch_mm_kwargs(batch_items, device="cpu"):
            model.embed_multimodal(**kwargs)


for n_threads in threads:
    torch.set_num_threads(n_threads)
    encode(items[:1])  # warm the per-shape kernels
    for batch in batches:
        t0 = time.perf_counter()
        encode(items[:batch])
        dt = time.perf_counter() - t0
        print(
            f"threads={n_threads:3d} batch={batch}: {dt:6.2f}s per call, "
            f"{dt / batch:6.2f}s per image",
            flush=True,
        )
