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

"""CPU-only tests for the parallel-encoder region allocator, transport and config."""

import uuid

import pytest
import torch
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import ECSharedRegion

from spyre_inference.config.parallel_encoder import ParallelEncoderConfig
from spyre_inference.v1.executor.parallel_encoder import channel_slot
from spyre_inference.v1.executor.parallel_encoder.channel import EncoderChannel
from spyre_inference.v1.executor.parallel_encoder.region import (
    RegionBlockAllocator,
    is_contiguous,
)
from spyre_inference.v1.executor.parallel_encoder.region_io import RegionReader, RegionWriter


def test_alloc_and_free_roundtrip():
    alloc = RegionBlockAllocator(num_blocks=8)
    assert alloc.num_free() == 8

    ids_a = alloc.alloc("hash_a", 3)
    assert len(ids_a) == 3
    assert len(set(ids_a)) == 3  # distinct blocks
    assert alloc.num_free() == 5
    assert alloc.is_allocated("hash_a")
    assert alloc.block_ids("hash_a") == ids_a

    ids_b = alloc.alloc("hash_b", 5)
    assert alloc.num_free() == 0
    assert set(ids_a).isdisjoint(ids_b)

    alloc.free("hash_a")
    assert alloc.num_free() == 3
    assert not alloc.is_allocated("hash_a")


def test_alloc_is_idempotent_for_same_hash():
    """Repeated identical image reuses the same region blocks (prefix cache)."""
    alloc = RegionBlockAllocator(num_blocks=8)
    ids_first = alloc.alloc("dup", 4)
    ids_again = alloc.alloc("dup", 4)
    assert ids_first == ids_again
    # A second allocation must NOT consume more blocks.
    assert alloc.num_free() == 4


def test_free_is_idempotent():
    alloc = RegionBlockAllocator(num_blocks=4)
    alloc.alloc("x", 2)
    alloc.free("x")
    alloc.free("x")  # no error, no double-return
    assert alloc.num_free() == 4


def test_alloc_over_capacity_raises():
    alloc = RegionBlockAllocator(num_blocks=4)
    alloc.alloc("x", 3)
    with pytest.raises(MemoryError):
        alloc.alloc("y", 2)


def test_reset_restores_all_blocks():
    alloc = RegionBlockAllocator(num_blocks=6)
    alloc.alloc("a", 2)
    alloc.alloc("b", 3)
    alloc.reset()
    assert alloc.num_free() == 6
    assert not alloc.is_allocated("a")


def test_no_block_reused_while_live():
    """Freed blocks may be reused, but live allocations never overlap."""
    alloc = RegionBlockAllocator(num_blocks=10)
    live: dict[str, list[int]] = {}
    for i in range(3):
        live[f"h{i}"] = alloc.alloc(f"h{i}", 2)
    all_live = [b for ids in live.values() for b in ids]
    assert len(all_live) == len(set(all_live))  # no overlap among live


def test_alloc_is_contiguous_in_common_case():
    """Each image gets a contiguous range -> single-copy transport."""
    alloc = RegionBlockAllocator(num_blocks=16)
    ids_a = alloc.alloc("a", 4)
    ids_b = alloc.alloc("b", 4)
    assert is_contiguous(ids_a)
    assert is_contiguous(ids_b)
    assert ids_a == list(range(ids_a[0], ids_a[0] + 4))
    assert set(ids_a).isdisjoint(ids_b)


def test_fragmentation_falls_back_to_scattered():
    """When no single free range fits, allocation still succeeds (scattered)."""
    alloc = RegionBlockAllocator(num_blocks=6)
    alloc.alloc("a", 2)  # [0,1]
    alloc.alloc("b", 2)  # [2,3]
    alloc.alloc("c", 2)  # [4,5]
    alloc.free("a")  # free [0,1]
    alloc.free("c")  # free [4,5]  -> two disjoint free ranges of size 2
    # Need 4 blocks: total free is 4 but no contiguous range of 4 exists.
    ids = alloc.alloc("d", 4)
    assert len(ids) == 4
    assert len(set(ids)) == 4
    assert not is_contiguous(ids)  # scattered fallback
    assert alloc.num_free() == 0


def test_free_coalesces_adjacent_ranges():
    """Freeing adjacent ranges restores a single large contiguous range."""
    alloc = RegionBlockAllocator(num_blocks=8)
    alloc.alloc("a", 4)  # [0,1,2,3]
    alloc.alloc("b", 4)  # [4,5,6,7]
    alloc.free("a")
    alloc.free("b")
    # Coalesced back to one range -> a full-size contiguous alloc succeeds.
    ids = alloc.alloc("c", 8)
    assert is_contiguous(ids)
    assert ids == list(range(8))


def test_config_defaults_to_cpu():
    class _Cfg:
        additional_config = {"parallel_encoder": {"enabled": True}}

    cfg = ParallelEncoderConfig.from_vllm_config(_Cfg())
    assert cfg is not None and cfg.enabled
    assert cfg.encoder_device == "cpu"
    assert cfg.encoder_device_index() is None
    assert not cfg.compiled
    assert ParallelEncoderConfig(encoder_device="spyre:2").compiled
    assert cfg.torch_dtype is torch.bfloat16
    assert ParallelEncoderConfig.is_enabled(_Cfg())


def test_config_absent_or_disabled():
    class _Absent:
        additional_config = {}

    class _Disabled:
        additional_config = {"parallel_encoder": {"enabled": False}}

    assert ParallelEncoderConfig.from_vllm_config(_Absent()) is None
    assert not ParallelEncoderConfig.is_enabled(_Absent())
    assert not ParallelEncoderConfig.is_enabled(_Disabled())


@pytest.mark.parametrize(("device", "index"), [("spyre:3", 3), ("spyre", 0)])
def test_config_spyre_device_index(device, index):
    assert ParallelEncoderConfig(encoder_device=device).encoder_device_index() == index


@pytest.mark.parametrize("device", ["cuda:1", "spyre:x", "gpu"])
def test_config_rejects_bad_device(device):
    with pytest.raises(ValueError, match="encoder_device"):
        ParallelEncoderConfig(encoder_device=device)


def test_config_rejects_bad_dtype():
    with pytest.raises(ValueError, match="encoder_dtype"):
        ParallelEncoderConfig(encoder_dtype="int8")


@pytest.fixture
def region():
    hidden, dtype = 24, torch.float16
    r = ECSharedRegion(
        engine_id=f"test_{uuid.uuid4().hex}",
        num_blocks=16,
        block_size_bytes=hidden * dtype.itemsize,
    )
    yield r, hidden, dtype
    r.cleanup()


def test_region_roundtrip_contiguous_and_scattered(region):
    r, hidden, dtype = region
    a = torch.randn(4, hidden, dtype=torch.bfloat16).to(dtype)
    b = torch.randn(3, hidden).to(dtype)
    RegionWriter(r).write_batch([("a", [2, 3, 4, 5], a), ("b", [0, 9, 14], b)])

    cache: dict[str, torch.Tensor] = {}
    RegionReader(r, dtype).load(cache, {"a": [2, 3, 4, 5], "b": [0, 9, 14]})
    torch.testing.assert_close(cache["a"], a, rtol=0, atol=0)
    torch.testing.assert_close(cache["b"], b, rtol=0, atol=0)


def test_region_read_survives_block_reuse(region):
    """A loaded entry is a copy, so a later image written to the same blocks leaves it."""
    r, hidden, dtype = region
    first = torch.ones(2, hidden, dtype=dtype)
    writer, reader = RegionWriter(r), RegionReader(r, dtype)
    writer.write_batch([("first", [0, 1], first)])
    cache: dict[str, torch.Tensor] = {}
    reader.load(cache, {"first": [0, 1]})
    writer.write_batch([("second", [0, 1], torch.zeros(2, hidden, dtype=dtype))])
    torch.testing.assert_close(cache["first"], first, rtol=0, atol=0)


def test_region_load_skips_cached(region):
    r, hidden, dtype = region
    sentinel = torch.full((2, hidden), 7, dtype=dtype)
    cache = {"x": sentinel}
    RegionReader(r, dtype).load(cache, {"x": [0, 1]})
    assert cache["x"] is sentinel


def test_region_write_rejects_oversized(region):
    r, hidden, dtype = region
    with pytest.raises(AssertionError, match="allocated"):
        RegionWriter(r).write_batch([("big", [0], torch.zeros(2, hidden, dtype=dtype))])


def test_channel_slot_hands_over_once():
    channel = EncoderChannel(dispatch_q=None, completion_q=None)
    channel_slot.publish(channel)
    assert channel_slot.take() is channel
    with pytest.raises(RuntimeError, match="ParallelEncoderExecutor"):
        channel_slot.take()
