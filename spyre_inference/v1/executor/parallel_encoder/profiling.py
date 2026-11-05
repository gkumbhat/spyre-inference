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

# SPDX-License-Identifier: Apache-2.0

"""Shared, gated profiling helpers for the parallel encoder.

All parallel-encoder profiling is off unless ``PARALLEL_ENCODER_PROFILE`` is
set (to a non-empty value). Kept in one place so the scheduler, encoder
process, and decoder worker report under a consistent switch and log format.
"""

import os

# Single switch for all parallel-encoder profiling probes.
PROFILE_ENABLED: bool = bool(os.environ.get("PARALLEL_ENCODER_PROFILE"))


def mm_item_nbytes(item) -> int:
    """Sum the tensor bytes in a ``MultiModalKwargsItem``.

    This is (roughly) what the mp-queue feeder thread must pickle when the
    scheduler dispatches an image to the encoder process — the quantity behind
    the GIL-contention hypothesis. Walks nested tensors without copying.
    """
    import torch

    total = 0
    # item is a UserDict[str, MultiModalFieldElem]; elem.data is NestedTensors.
    stack = [elem.data for elem in item.values()]
    while stack:
        x = stack.pop()
        if isinstance(x, torch.Tensor):
            total += x.element_size() * x.nelement()
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
        elif isinstance(x, dict):
            stack.extend(x.values())
    return total
