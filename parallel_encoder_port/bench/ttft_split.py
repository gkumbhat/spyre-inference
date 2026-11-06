"""Mean per-request latency phases from vLLM /metrics scrapes, one per arm.

Usage: python ttft_split.py NAME=metrics.txt [NAME=metrics.txt ...]
queue = queued -> scheduled; prefill = scheduled -> first token (includes any wait for an
image); decode = first -> last token.
"""

import re
import sys

PHASES = {
    "queue": "vllm:request_queue_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "ttft": "vllm:time_to_first_token_seconds",
    "decode": "vllm:request_decode_time_seconds",
}


def mean(text: str, metric: str) -> float | None:
    sums = re.findall(rf"^{re.escape(metric)}_sum\{{[^}}]*\}} ([0-9.e+-]+)", text, re.M)
    counts = re.findall(rf"^{re.escape(metric)}_count\{{[^}}]*\}} ([0-9.e+-]+)", text, re.M)
    if not sums or not counts or float(counts[0]) == 0:
        return None
    return sum(map(float, sums)) / sum(map(float, counts))


for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    with open(path) as f:
        text = f.read()
    parts = {k: mean(text, m) for k, m in PHASES.items()}
    print(
        name, " ".join(f"{k}={v:.2f}s" if v is not None else f"{k}=n/a" for k, v in parts.items())
    )
