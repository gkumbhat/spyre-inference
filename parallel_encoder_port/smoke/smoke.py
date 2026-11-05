"""Parallel-encoder smoke run: same prompts, inline vs parallel encoder, one mode per process.

Usage: python smoke.py {inline|parallel} MODEL OUT.json [TP] [eager|compiled]

The last argument sets the decoder mode; a CPU encoder always runs eager.
"""

import base64
import io
import json
import os
import sys

from PIL import Image

MAX_MODEL_LEN = 1024
MAX_TOKENS = 24
NUM_LOGPROBS = 5


def image_uri(size: int = 176, seed: int = 0) -> str:
    image = Image.new("RGB", (size, size))
    px = image.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = ((x * 7 + seed) % 256, (y * 5 + seed) % 256, ((x + y) * 3 + seed) % 256)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def conversation(text: str, *uris: str):
    return [
        {
            "role": "user",
            "content": [
                *({"type": "image_url", "image_url": {"url": u}} for u in uris),
                {"type": "text", "text": text},
            ],
        }
    ]


PROMPTS = {
    "text_only": conversation("Name three primary colors."),
    "one_image": conversation("Describe this image in one short sentence.", image_uri(seed=0)),
    "repeat_image": conversation("What colors dominate this image?", image_uri(seed=0)),
    "two_images": conversation(
        "How do these two images differ?", image_uri(seed=0), image_uri(seed=97)
    ),
}


def main() -> None:
    from vllm import LLM, SamplingParams

    MODE, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    TP = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    DECODER = sys.argv[5] if len(sys.argv) > 5 else "eager"

    kwargs = dict(
        model=MODEL,
        config_format="mistral",
        tokenizer_mode="mistral",
        load_format="mistral",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=4,
        dtype="float16",
        enforce_eager=DECODER == "eager",
        tensor_parallel_size=TP,
        limit_mm_per_prompt={"image": 2},
    )
    if MODE == "parallel":
        kwargs.update(
            distributed_executor_backend=(
                "spyre_inference.v1.executor.parallel_encoder.executor.ParallelEncoderExecutor"
            ),
            additional_config={"parallel_encoder": {"enabled": True}},
        )

    llm = LLM(**kwargs)
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, logprobs=NUM_LOGPROBS)

    results = {}
    for name, conv in PROMPTS.items():
        (out,) = llm.chat([conv], params)
        comp = out.outputs[0]
        results[name] = {
            "text": comp.text,
            "token_ids": list(comp.token_ids),
            "logprobs": [
                {str(tok): lp.logprob for tok, lp in step.items()} for step in (comp.logprobs or [])
            ],
        }
        print(f"[{MODE}] {name}: {comp.text!r}", flush=True)

    # Batched: every multimodal prompt at once, so encodes overlap decode.
    batch = llm.chat(list(PROMPTS.values()), params)
    results["_batched"] = {
        "texts": [o.outputs[0].text for o in batch],
        "token_ids": [list(o.outputs[0].token_ids) for o in batch],
    }

    with open(OUT, "w") as f:
        json.dump(
            {
                "mode": MODE,
                "model": MODEL,
                "tp": TP,
                "decoder": DECODER,
                "pid": os.getpid(),
                "results": results,
            },
            f,
            indent=1,
        )
    print(f"[{MODE}] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
