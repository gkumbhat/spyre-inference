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

"""When images reach the encoder, for the serve benchmark's request shape.

random-mm sends the text first and the image last, and Spyre caps the step at 512 tokens.
These replay four such requests through the real schedule() loop, with the encoder
answering at once, and record the step each image is dispatched in.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched import scheduler as base_scheduler_module
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from spyre_inference.v1.core import parallel_encoder_scheduler as pe_scheduler_module
from spyre_inference.v1.executor.parallel_encoder import channel_slot
from spyre_inference.v1.executor.parallel_encoder.channel import EncodeDone, EncoderChannel

TEXT, IMAGE, NREQ = 8192, 3080, 4
BLOCK_SIZE = 64
NUM_BLOCKS = 4096
MAX_MODEL_LEN = 16384


class _FakeQueue:
    def __init__(self) -> None:
        self.items: list = []

    def put(self, item) -> None:
        self.items.append(item)

    def get_nowait(self):
        import queue

        if not self.items:
            raise queue.Empty
        return self.items.pop(0)


def _scheduler(monkeypatch, max_num_batched_tokens: int):
    def budget(*_args, **_kwargs):
        return SimpleNamespace(
            encoder_compute_budget=max(max_num_batched_tokens, IMAGE),
            encoder_cache_size=max(max_num_batched_tokens, IMAGE),
            mm_max_toks_per_item={"image": IMAGE},
        )

    monkeypatch.setattr(base_scheduler_module, "MultiModalBudget", budget)
    monkeypatch.setattr(pe_scheduler_module, "MultiModalBudget", budget)
    monkeypatch.setattr(ModelConfig, "supports_multimodal_inputs", property(lambda self: True))
    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=MAX_MODEL_LEN,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    cache_config = CacheConfig(block_size=BLOCK_SIZE, enable_prefix_caching=False)
    cache_config.num_gpu_blocks = NUM_BLOCKS
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=NREQ,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=MAX_MODEL_LEN,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
        ),
    )
    # The platform caps max_num_batched_tokens at 512; restore the value under test.
    vllm_config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    vllm_config.additional_config = {
        "parallel_encoder": {"enabled": True, "encoder_cache_tokens": 16 * IMAGE}
    }
    dispatch_q, completion_q = _FakeQueue(), _FakeQueue()
    channel_slot.publish(EncoderChannel(dispatch_q, completion_q))
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=8, head_size=128, dtype=torch.float16
    )
    scheduler = pe_scheduler_module.ParallelEncoderScheduler(
        vllm_config=vllm_config,
        kv_cache_config=KVCacheConfig(
            num_blocks=NUM_BLOCKS,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
        ),
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
    )
    scheduler.max_num_partial_prefills = NREQ
    return SimpleNamespace(scheduler=scheduler, dispatch=dispatch_q, completion=completion_q)


def _request(request_id: str) -> Request:
    feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        mm_position=PlaceholderRange(offset=TEXT, length=IMAGE),
        identifier=f"img-{request_id}",
        modality="image",
    )
    params = SamplingParams(max_tokens=8)
    params.update_from_generation_config({}, 0)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(1, TEXT + IMAGE + 1)),
        sampling_params=params,
        pooling_params=None,
        mm_features=[feature],
    )


def replay(monkeypatch, max_num_batched_tokens: int) -> dict[str, int]:
    """Step at which each request's image is dispatched to the encoder."""
    h = _scheduler(monkeypatch, max_num_batched_tokens)
    for i in range(NREQ):
        h.scheduler.add_request(_request(f"r{i}"))
    dispatched: dict[str, int] = {}
    for step in range(1, 400):
        out = h.scheduler.schedule()
        for req_id, n in out.num_scheduled_tokens.items():
            h.scheduler.requests[req_id].num_computed_tokens += n
        for batch in h.dispatch.items:
            for item in batch.items:
                dispatched.setdefault(item.mm_hash, step)
                h.completion.put(EncodeDone(mm_hash=item.mm_hash))
        h.dispatch.items.clear()
        if len(dispatched) == NREQ:
            break
    return dispatched


def test_images_reach_the_encoder_on_arrival(monkeypatch):
    """At 512 tokens a step, an image after an 8k prompt is in no step's window until the
    text has prefilled, and Spyre admits one prefill at a time. Every image must still be
    dispatched in the first step, so all four encode while the text prefills."""
    steps = replay(monkeypatch, 512)
    assert steps == {f"img-r{i}": 1 for i in range(NREQ)}


def test_look_ahead_respects_the_in_flight_cap(monkeypatch):
    h = _scheduler(monkeypatch, 512)
    h.scheduler._max_inflight_encode_images = 2
    for i in range(NREQ):
        h.scheduler.add_request(_request(f"r{i}"))
    h.scheduler.schedule()
    dispatched = [item.mm_hash for batch in h.dispatch.items for item in batch.items]
    assert dispatched == ["img-r0", "img-r1"]


def test_look_ahead_does_not_truncate_or_load_early(monkeypatch):
    """A dispatched image past the window neither stops the text prefill before it nor
    is loaded by the workers until the window reaches it."""
    h = _scheduler(monkeypatch, 512)
    h.scheduler.add_request(_request("r0"))
    out = h.scheduler.schedule()
    assert out.num_scheduled_tokens["r0"] == 512
    assert out.encoder_region_loads == {}
    h.completion.put(EncodeDone(mm_hash="img-r0"))
    h.scheduler.requests["r0"].num_computed_tokens += 512
    out = h.scheduler.schedule()
    assert out.num_scheduled_tokens["r0"] == 512
    assert out.encoder_region_loads == {}, "loaded before the window reached the image"
