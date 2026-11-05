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

"""ParallelEncoderScheduler bookkeeping, driven through the real upstream ``schedule()``.

The encoder process is replaced by in-memory queues the test answers by hand, and
``MultiModalBudget`` by a stub, so no model or processor loads.
"""

from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched import scheduler as base_scheduler_module
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from spyre_inference.v1.core import parallel_encoder_scheduler as pe_scheduler_module
from spyre_inference.v1.executor.parallel_encoder import channel_slot
from spyre_inference.v1.executor.parallel_encoder.channel import EncodeDone, EncoderChannel

BLOCK_SIZE = 64
NUM_BLOCKS = 512
IMAGE_TOKENS = 64
PROMPT_LEN = 200
IMAGE_OFFSET = 20


class _FakeQueue:
    def __init__(self) -> None:
        self.items: list = []

    def put(self, item) -> None:
        self.items.append(item)

    def get_nowait(self):
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)


def _budget(*_args, **_kwargs):
    return SimpleNamespace(
        encoder_compute_budget=4 * IMAGE_TOKENS,
        encoder_cache_size=4 * IMAGE_TOKENS,
        mm_max_toks_per_item={"image": IMAGE_TOKENS},
    )


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(base_scheduler_module, "MultiModalBudget", _budget)
    monkeypatch.setattr(pe_scheduler_module, "MultiModalBudget", _budget)
    monkeypatch.setattr(ModelConfig, "supports_multimodal_inputs", property(lambda self: True))

    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B", max_model_len=2048, dtype=torch.float16, trust_remote_code=True
    )
    cache_config = CacheConfig(block_size=BLOCK_SIZE, enable_prefix_caching=False)
    cache_config.num_gpu_blocks = NUM_BLOCKS
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=512,
            max_model_len=2048,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
        ),
    )
    # After validation: the platform rejects the parallel encoder on a text model.
    vllm_config.additional_config = {
        "parallel_encoder": {"enabled": True, "encoder_cache_tokens": 3 * IMAGE_TOKENS}
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
    return SimpleNamespace(scheduler=scheduler, dispatch=dispatch_q, completion=completion_q)


def _request(request_id: str, image_hash: str) -> Request:
    feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        mm_position=PlaceholderRange(offset=IMAGE_OFFSET, length=IMAGE_TOKENS),
        identifier=image_hash,
        modality="image",
    )
    params = SamplingParams(max_tokens=8)
    params.update_from_generation_config({}, 0)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(1, PROMPT_LEN + 1)),
        sampling_params=params,
        pooling_params=None,
        mm_features=[feature],
    )


def _dispatched_hashes(h) -> list[str]:
    hashes = [item.mm_hash for batch in h.dispatch.items for item in batch.items]
    h.dispatch.items.clear()
    return hashes


def _step(h):
    out = h.scheduler.schedule()
    # No model runs: advance computed tokens the way update_from_output would for
    # a prefill chunk, without sampling.
    for req_id, n in out.num_scheduled_tokens.items():
        h.scheduler.requests[req_id].num_computed_tokens += n
    return out


def test_image_dispatched_once_and_decode_stops_before_it(harness):
    harness.scheduler.add_request(_request("r0", "img-a"))
    out = _step(harness)
    assert _dispatched_hashes(harness) == ["img-a"]
    assert out.num_scheduled_tokens["r0"] == IMAGE_OFFSET
    assert out.scheduled_encoder_inputs == {}
    assert out.encoder_region_loads == {}

    out = _step(harness)
    assert _dispatched_hashes(harness) == [], "an in-flight image must not be re-dispatched"
    assert out.num_scheduled_tokens.get("r0", 0) == 0


def test_completion_emits_one_region_load_then_decode_continues(harness):
    harness.scheduler.add_request(_request("r0", "img-a"))
    _step(harness)
    _dispatched_hashes(harness)

    harness.completion.put(EncodeDone(mm_hash="img-a"))
    out = _step(harness)
    assert list(out.encoder_region_loads) == ["img-a"]
    assert len(out.encoder_region_loads["img-a"]) == IMAGE_TOKENS
    # Past the image; the platform may hold back a prefill-lookahead tail.
    assert harness.scheduler.requests["r0"].num_computed_tokens >= IMAGE_OFFSET + IMAGE_TOKENS

    out = _step(harness)
    assert out.encoder_region_loads == {}, "a loaded image is not loaded again"


def test_repeated_image_is_encoded_once(harness):
    s = harness.scheduler
    s.add_request(_request("r0", "img-a"))
    _step(harness)
    harness.completion.put(EncodeDone(mm_hash="img-a"))
    _step(harness)

    s.add_request(_request("r1", "img-a"))
    out = _step(harness)
    assert _dispatched_hashes(harness) == ["img-a"], "only the first request dispatched"
    assert s.requests["r1"].num_computed_tokens >= IMAGE_OFFSET + IMAGE_TOKENS
    assert out.encoder_region_loads == {}, "the workers already hold img-a"
    assert s._region_alloc.num_free() == s._region_alloc.num_blocks - IMAGE_TOKENS


def test_finished_images_return_their_blocks_when_evicted(harness):
    """The working set holds three images. Once img-a's requests finish, three new
    images evict it; its region blocks must come back or the third cannot fit."""
    s = harness.scheduler
    s.add_request(_request("r0", "img-a"))
    _step(harness)
    harness.completion.put(EncodeDone(mm_hash="img-a"))
    _step(harness)
    s.finish_requests(["r0"], RequestStatus.FINISHED_ABORTED)
    _dispatched_hashes(harness)

    for i, image in enumerate(("img-b", "img-c", "img-d")):
        s.add_request(_request(f"n{i}", image))
    freed: list[str] = []
    for _ in range(3):
        freed += _step(harness).free_encoder_mm_hashes
    assert sorted(_dispatched_hashes(harness)) == ["img-b", "img-c", "img-d"]
    assert "img-a" in freed
    assert not s._region_alloc.is_allocated("img-a")
    assert s._region_alloc.num_free() == 0

    for image in ("img-b", "img-c", "img-d"):
        harness.completion.put(EncodeDone(mm_hash=image))
    _step(harness)
    s.finish_requests(["n0", "n1", "n2"], RequestStatus.FINISHED_ABORTED)
    s.add_request(_request("z", "img-z"))
    for _ in range(2):
        _step(harness)
    assert _dispatched_hashes(harness) == ["img-z"]
    # Finished images stay cached until space is needed: img-z took one's blocks.
    live = [h for h in ("img-b", "img-c", "img-d", "img-z") if s._region_alloc.is_allocated(h)]
    assert len(live) == 3 and "img-z" in live
    assert s._region_alloc.num_free() == 0


def test_new_requests_wait_when_too_many_images_are_in_flight(harness):
    s = harness.scheduler
    s._max_inflight_encode_images = 1
    s.add_request(_request("r0", "img-a"))
    s.add_request(_request("r1", "img-b"))
    out = _step(harness)
    assert _dispatched_hashes(harness) == ["img-a"]
    assert "r1" not in out.num_scheduled_tokens
    assert s.requests["r1"].num_computed_tokens == 0


def test_parked_request_does_not_block_admission(harness):
    """Spyre admits one partial prefill at a time by default. A request parked on its
    image does no prefill work, so the next request must still be admitted."""
    s = harness.scheduler
    assert s.max_num_partial_prefills == 1
    s.add_request(_request("r0", "img-a"))
    _step(harness)
    s.add_request(_request("text", "img-b"))
    out = _step(harness)
    assert out.num_scheduled_tokens.get("text", 0) > 0
    assert sorted(_dispatched_hashes(harness)) == ["img-a", "img-b"]
