"""AC-8: kill the encoder process while an image request is in flight.

The engine must fail that request with a clear error and shut down, not hang.
Usage: python crash.py MODEL [eager|compiled]
"""

import os
import signal
import sys
import threading

import psutil

sys.path.insert(0, os.path.dirname(__file__))
from smoke import PROMPTS  # noqa: E402

MODEL = sys.argv[1]
DECODER = sys.argv[2] if len(sys.argv) > 2 else "compiled"
HANG_LIMIT_S = 300


def encoder_pid() -> int | None:
    for proc in psutil.process_iter():
        try:
            if proc.environ().get("SPYRE_PARALLEL_ENCODER_PROC") == "1":
                return proc.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def main() -> int:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        config_format="mistral",
        tokenizer_mode="mistral",
        load_format="mistral",
        max_model_len=1024,
        max_num_seqs=4,
        dtype="float16",
        enforce_eager=DECODER == "eager",
        limit_mm_per_prompt={"image": 2},
        distributed_executor_backend=(
            "spyre_inference.v1.executor.parallel_encoder.executor.ParallelEncoderExecutor"
        ),
        additional_config={"parallel_encoder": {"enabled": True}},
    )
    params = SamplingParams(max_tokens=24, temperature=0.0)
    (warm,) = llm.chat([PROMPTS["text_only"]], params)
    print(f"[crash] warm text-only ok: {warm.outputs[0].text[:40]!r}", flush=True)

    pid = encoder_pid()
    if pid is None:
        print("[crash] FAIL: encoder process not found", flush=True)
        return 2
    print(f"[crash] encoder pid {pid}; killing it with SIGKILL in 1s", flush=True)
    threading.Timer(1.0, os.kill, args=(pid, signal.SIGKILL)).start()

    done = threading.Event()
    outcome: dict = {}

    def run() -> None:
        try:
            llm.chat([PROMPTS["two_images"]] * 4, params)
            outcome["result"] = "completed"
        except Exception as e:  # noqa: BLE001
            outcome["result"] = f"{type(e).__name__}: {e}"
        done.set()

    threading.Thread(target=run, daemon=True).start()
    if not done.wait(HANG_LIMIT_S):
        print(f"[crash] FAIL: engine still blocked after {HANG_LIMIT_S}s (hang)", flush=True)
        os._exit(3)
    print(f"[crash] outcome: {outcome['result']}", flush=True)
    if outcome["result"] == "completed":
        print("[crash] FAIL: requests completed although the encoder was killed", flush=True)
        return 4
    print("[crash] PASS: clear error, no hang", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
