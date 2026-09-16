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

"""Tests for `spyre_inference/custom_ops/vit_attn.py`. CPU-only: correctness of
the padding/masking math, checked against plain unpadded SDPA."""

from __future__ import annotations

import einops
import torch
import torch.nn.functional as F

from spyre_inference.custom_ops.vit_attn import _padded_apply_sdpa, register


def _reference_apply_sdpa(q, k, v, scale=None, enable_gqa=False):
    """Upstream's unpadded `apply_sdpa`, reproduced so this test has no vLLM
    version dependency."""
    q, k, v = (einops.rearrange(x, "b s h d -> b h s d") for x in (q, k, v))
    out = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, scale=scale, enable_gqa=enable_gqa
    )
    return einops.rearrange(out, "b h s d -> b s h d")


def _random_qkv(batch, seq_q, seq_kv, heads, head_size, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, seq_q, heads, head_size, generator=g)
    k = torch.randn(batch, seq_kv, heads, head_size, generator=g)
    v = torch.randn(batch, seq_kv, heads, head_size, generator=g)
    return q, k, v


class TestPaddedApplySdpaMatchesReference:
    def test_clip_vit_b32_shape(self):
        # B=1, H=12, seq=50 (7x7 patches + CLS), D=64 -- the shape that crashed.
        q, k, v = _random_qkv(batch=1, seq_q=50, seq_kv=50, heads=12, head_size=64)
        expected = _reference_apply_sdpa(q, k, v)
        actual = _padded_apply_sdpa(q, k, v)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_head_dim_also_needs_padding(self):
        q, k, v = _random_qkv(batch=2, seq_q=37, seq_kv=37, heads=4, head_size=48)
        expected = _reference_apply_sdpa(q, k, v)
        actual = _padded_apply_sdpa(q, k, v)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_already_aligned_shape_takes_the_no_pad_branch(self):
        q, k, v = _random_qkv(batch=1, seq_q=64, seq_kv=64, heads=8, head_size=64)
        expected = _reference_apply_sdpa(q, k, v)
        actual = _padded_apply_sdpa(q, k, v)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_cross_attention_shape_with_different_seq_q_and_seq_kv(self):
        q, k, v = _random_qkv(batch=1, seq_q=17, seq_kv=50, heads=4, head_size=32)
        expected = _reference_apply_sdpa(q, k, v)
        actual = _padded_apply_sdpa(q, k, v)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_enable_gqa_is_passed_through(self):
        # 8 query heads grouped over 2 kv heads.
        q = torch.randn(1, 50, 8, 64)
        k = torch.randn(1, 50, 2, 64)
        v = torch.randn(1, 50, 2, 64)
        expected = _reference_apply_sdpa(q, k, v, enable_gqa=True)
        actual = _padded_apply_sdpa(q, k, v, enable_gqa=True)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


class TestRegister:
    def test_register_is_idempotent(self):
        import vllm.v1.attention.ops.vit_attn_wrappers as vit_attn_wrappers

        register()
        patched = vit_attn_wrappers.apply_sdpa
        register()

        assert vit_attn_wrappers.apply_sdpa is patched
