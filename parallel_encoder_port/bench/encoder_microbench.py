"""Time the CPU vision tower alone: one 2048x2048 image, batch and thread sweeps.

Loads the model as the encoder process does (no Spyre plugin, mm_encoder_only), builds
the real processor input for the image, and times embed_multimodal per call.
Run with: VLLM_PLUGINS= TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_TARGET_DEVICE=cpu python -I ...
Usage: encoder_microbench.py MODEL THREADS[,THREADS...] BATCH[,BATCH...]
"""

import sys
import time

import numpy as np
import torch
from PIL import Image
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY
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

rng = np.random.default_rng(0)
image = Image.fromarray(rng.integers(0, 256, (2048, 2048, 3), dtype=np.uint8))
processor = MULTIMODAL_REGISTRY.create_processor(cfg.model_config)
inputs = processor.apply(
    prompt="[IMG]", mm_data={"image": [image] * max(batches)}, hf_processor_mm_kwargs={}
)
items = [("image", item) for item in inputs["mm_kwargs"]["image"]]
rows = inputs["mm_placeholders"]["image"][0].get_num_embeds()
print(f"one 2048x2048 image -> {rows} embedding rows", flush=True)


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
