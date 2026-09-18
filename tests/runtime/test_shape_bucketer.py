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

"""Unit tests for SpyreShapeBucketer."""

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from spyre_inference.v1.worker.spyre_shape_bucketer import (
    SpyreShapeBucketer,
    default_encoder_len_buckets,
    logits_row_buckets,
    next_bucket,
)


@pytest.fixture()
def mock_vllm_config():
    """Create a minimal VllmConfig mock with compile_sizes."""
    config = MagicMock()
    config.compilation_config.compile_sizes = [1, 2, 4, 8, 16]
    return config


@pytest.fixture()
def bucketer(mock_vllm_config):
    return SpyreShapeBucketer(mock_vllm_config)


class TestFindBucket:
    def test_exact_match(self, bucketer):
        assert bucketer.find_bucket(8) == 8

    def test_rounds_up_to_next_bucket(self, bucketer):
        assert bucketer.find_bucket(3) == 4
        assert bucketer.find_bucket(5) == 8
        assert bucketer.find_bucket(9) == 16

    def test_smallest_token_count(self, bucketer):
        assert bucketer.find_bucket(1) == 1

    def test_exceeds_max_returns_none(self, bucketer):
        assert bucketer.find_bucket(17) is None
        assert bucketer.find_bucket(100) is None

    def test_zero_tokens(self, bucketer):
        assert bucketer.find_bucket(0) == 1


class TestDispatch:
    def test_returns_descriptor_with_padding(self, bucketer):
        desc = bucketer.dispatch(5)
        assert desc is not None
        assert desc.actual_num_tokens == 5
        assert desc.padded_num_tokens == 8

    def test_exact_match_no_padding(self, bucketer):
        desc = bucketer.dispatch(4)
        assert desc is not None
        assert desc.actual_num_tokens == 4
        assert desc.padded_num_tokens == 4

    def test_exceeds_max_returns_none(self, bucketer):
        assert bucketer.dispatch(20) is None

    def test_descriptor_is_frozen(self, bucketer):
        desc = bucketer.dispatch(3)
        with pytest.raises(FrozenInstanceError):
            desc.actual_num_tokens = 10


class TestBucketerState:
    def test_initial_state_not_warmed_up(self, bucketer):
        assert not bucketer.is_warmed_up

    def test_mark_warmed_up(self, bucketer):
        bucketer.mark_warmed_up()
        assert bucketer.is_warmed_up

    def test_bucket_sizes_sorted(self, bucketer):
        assert bucketer.bucket_sizes == [1, 2, 4, 8, 16]

    def test_max_bucket_size(self, bucketer):
        assert bucketer.max_bucket_size == 16


class TestEdgeCases:
    def test_empty_compile_sizes(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = []
        b = SpyreShapeBucketer(config)
        assert b.bucket_sizes == []
        assert b.max_bucket_size == 0
        assert b.find_bucket(1) is None
        assert b.dispatch(1) is None

    def test_single_bucket(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = [8]
        b = SpyreShapeBucketer(config)
        assert b.find_bucket(1) == 8
        assert b.find_bucket(8) == 8
        assert b.find_bucket(9) is None

    def test_unsorted_input_gets_sorted(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = [16, 2, 8, 1, 4]
        b = SpyreShapeBucketer(config)
        assert b.bucket_sizes == [1, 2, 4, 8, 16]


def _pooling_vllm_config(
    *,
    max_num_seqs: int = 4,
    max_model_len: int = 128,
    max_num_batched_tokens: int = 512,
    runner_type: str = "pooling",
) -> MagicMock:
    config = MagicMock()
    config.model_config.runner_type = runner_type
    config.model_config.max_model_len = max_model_len
    config.scheduler_config.max_num_seqs = max_num_seqs
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    config.compilation_config.compile_sizes = []
    return config


class TestForPooling:
    """Encoder attention is varlen flash; pooling gets the same 1D bucketer as decode."""

    def test_skips_non_pooling(self):
        assert SpyreShapeBucketer.for_pooling(_pooling_vllm_config(runner_type="generate")) is None

    def test_none_when_no_compile_sizes(self):
        cfg = _pooling_vllm_config()
        cfg.compilation_config.compile_sizes = []
        assert SpyreShapeBucketer.for_pooling(cfg) is None

    def test_uses_1d_compile_sizes(self):
        cfg = _pooling_vllm_config()
        cfg.compilation_config.compile_sizes = [64, 128]
        b = SpyreShapeBucketer.for_pooling(cfg)
        assert b is not None
        assert b.bucket_sizes == [64, 128]


class TestEncoderLenLadder:
    """``default_encoder_len_buckets``/``next_bucket`` are still shared with the
    pooler's own bucketed-row trimming (``spyre_pooler.py``) and with
    ``platform.py``'s default ``compile_sizes`` generation."""

    def test_next_bucket_picks_smallest_fit(self):
        assert next_bucket(30, [64, 128, 256]) == 64
        assert next_bucket(64, [64, 128, 256]) == 64
        assert next_bucket(65, [64, 128, 256]) == 128

    def test_next_bucket_overflow_stick_aligns(self):
        assert next_bucket(3000, [64, 128]) == 3008  # 3000 → 47*64 = 3008

    def test_default_encoder_len_buckets_from_max_model_len(self):
        assert default_encoder_len_buckets(512) == [64, 128, 256, 512]
        assert default_encoder_len_buckets(2048) == [64, 128, 256, 512, 1024, 2048]
        assert default_encoder_len_buckets(100) == [64]
        assert default_encoder_len_buckets(768) == [64, 128, 256, 512, 768]

    def test_default_encoder_len_buckets_honours_a_higher_floor(self):
        """The body raises the floor to keep warmup short; the pooler keeps the stick."""
        assert default_encoder_len_buckets(2048, floor=256) == [256, 512, 1024, 2048]
        assert default_encoder_len_buckets(512, floor=256) == [256, 512]
        # A model shorter than the floor still gets one usable bucket.
        assert default_encoder_len_buckets(100, floor=256) == [64]


class TestLogitsRowBuckets:
    def test_clips_prefill_bucket_to_max_num_reqs(self):
        # The 512-token prefill bucket samples at most max_num_seqs rows.
        assert logits_row_buckets([1, 2, 4, 8, 512], max_num_reqs=8) == [1, 2, 4, 8]

    def test_keeps_a_non_power_of_two_max(self):
        assert logits_row_buckets([1, 2, 4, 6, 512], max_num_reqs=6) == [1, 2, 4, 6]

    def test_ignores_non_positive_sizes(self):
        assert logits_row_buckets([0, -1, 4], max_num_reqs=8) == [4]
