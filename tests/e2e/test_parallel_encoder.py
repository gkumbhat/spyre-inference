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

"""The parallel encoder against the inline tower, on the same compiled decoder.

Text-only output must match token for token: those requests never reach the encoder.
Image output must be logprob-close, as in upstream's check_logprobs_close: the CPU tower
computes in bf16 while the inline one runs fp16 on the card, so a near-tie can flip,
but each side's token must be in the other's top-k where they first differ.
"""

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_device_count
from test_multimodal import MODEL, _conversation, _synthetic_image_data_uri

MAX_MODEL_LEN = 512
MAX_TOKENS = 16
NUM_LOGPROBS = 5

PARALLEL_ENCODER = {
    "distributed_executor_backend": (
        "spyre_inference.v1.executor.parallel_encoder.executor.ParallelEncoderExecutor"
    ),
    "additional_config": {"parallel_encoder": {"enabled": True}},
}


def _generate(conversations, **llm_kwargs):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        config_format="mistral",
        tokenizer_mode="mistral",
        load_format="mistral",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=len(conversations),
        compilation_config={"compile_sizes": [1, MAX_MODEL_LEN]},
        dtype="float16",
        limit_mm_per_prompt={"image": 2},
        **llm_kwargs,
    )
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, logprobs=NUM_LOGPROBS)
    outputs = [o.outputs[0] for o in llm.chat(conversations, params)]
    return [(list(o.token_ids), o.logprobs) for o in outputs]


def _assert_logprobs_close(inline, parallel, name):
    (ids_a, lps_a), (ids_b, lps_b) = inline, parallel
    for i, (a, b) in enumerate(zip(ids_a, ids_b)):
        if a != b:
            assert a in lps_b[i] and b in lps_a[i], (
                f"{name}: first difference at token {i} is not a near-tie: inline {a} "
                f"is not in parallel's top-{NUM_LOGPROBS}, or parallel {b} not in inline's"
            )
            return


@pytest.mark.multimodal
@pytest.mark.uses_subprocess
@pytest.mark.timeout(7200)
def test_parallel_encoder_matches_inline(monkeypatch):
    if spyre_device_count() == 0:
        pytest.skip("Spyre device not available")
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    conversations = {
        "text_only": _conversation(),
        "one_image": _conversation(_synthetic_image_data_uri(seed=0)),
        "two_images": _conversation(
            _synthetic_image_data_uri(seed=0), _synthetic_image_data_uri(seed=97)
        ),
    }
    inline = _generate(list(conversations.values()))
    parallel = _generate(list(conversations.values()), **PARALLEL_ENCODER)

    by_name = dict(zip(conversations, zip(inline, parallel)))
    assert by_name["text_only"][0][0] == by_name["text_only"][1][0], (
        "text-only output changed although it never reaches the encoder"
    )
    for name in ("one_image", "two_images"):
        _assert_logprobs_close(*by_name[name], name)
