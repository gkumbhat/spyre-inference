"""Load the model the way the encoder process does and count what it keeps (AC-2)."""

import sys

from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.cpu_model_runner import _set_torch_accelerator_to_noop
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

model = sys.argv[1]
_set_torch_accelerator_to_noop()
cfg = EngineArgs(
    model=model,
    config_format="mistral",
    tokenizer_mode="mistral",
    load_format="mistral",
    max_model_len=1024,
    dtype="bfloat16",
    enforce_eager=True,
    limit_mm_per_prompt={"image": 2},
    mm_encoder_only=True,
).create_engine_config()

with set_current_vllm_config(cfg):
    init_worker_distributed_environment(cfg, 0, "tcp://127.0.0.1:29533", 0, "gloo")
    m = get_model(vllm_config=cfg)

real = {n: p for n, p in m.named_parameters() if p.device.type != "meta"}
meta = [n for n, p in m.named_parameters() if p.device.type == "meta"]
groups: dict[str, int] = {}
for n, p in real.items():
    groups[n.split(".")[0]] = groups.get(n.split(".")[0], 0) + p.numel()
print("resident params by top-level module:", {k: f"{v / 1e6:.1f}M" for k, v in groups.items()})
print(f"resident total {sum(p.numel() for p in real.values()) / 1e6:.1f}M, meta params {len(meta)}")
print("language_model resident:", any(n.startswith("language_model") for n in real))
print("language_model type:", type(getattr(m, "language_model", None)).__name__)
