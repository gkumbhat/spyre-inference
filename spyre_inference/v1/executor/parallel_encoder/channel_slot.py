# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0
"""Hands the executor's ``EncoderChannel`` to the scheduler.

EngineCore builds the executor, then the scheduler, in the same process, and has no
hook between them. Spyre rejects DP>1, so a process holds at most one engine.
"""

from spyre_inference.v1.executor.parallel_encoder.channel import EncoderChannel

_channel: EncoderChannel | None = None


def publish(channel: EncoderChannel) -> None:
    global _channel
    _channel = channel


def take() -> EncoderChannel:
    global _channel
    if _channel is None:
        raise RuntimeError(
            "ParallelEncoderScheduler found no encoder channel: run with "
            "--distributed-executor-backend "
            "spyre_inference.v1.executor.parallel_encoder.executor.ParallelEncoderExecutor"
        )
    channel, _channel = _channel, None
    return channel
