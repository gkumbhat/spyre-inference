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

"""Coordination channel between the scheduler and the encoder process.

Plain multiprocessing queues carry dispatch requests (scheduler -> encoder)
and completion signals (encoder -> scheduler). Embedding *bytes* travel out
of band through the shared region; only small metadata flows on these queues.
"""

from dataclasses import dataclass, field

from vllm.multimodal.inputs import MultiModalKwargsItem


@dataclass
class EncodeItem:
    """One multimodal item to encode.

    Attributes:
        mm_hash: Stable identifier of the mm item (``mm_feature.identifier``).
        modality: Modality string (e.g. ``"image"``).
        block_ids: Region block IDs the encoder must write the embedding to,
            one block per output embedding token. Chosen by the scheduler.
        data: The preprocessed multimodal kwargs for this item.
    """

    mm_hash: str
    modality: str
    block_ids: list[int]
    data: MultiModalKwargsItem


@dataclass
class EncodeBatch:
    """A batch of items dispatched together (batched across requests)."""

    items: list[EncodeItem] = field(default_factory=list)


@dataclass
class EncodeDone:
    """Completion signal for a finished mm_hash.

    ``ok=False`` carries an error string; the scheduler treats an encode
    failure the same as a fatal encoder fault.
    """

    mm_hash: str
    ok: bool = True
    error: str | None = None
    # Wall clock when the embedding landed in the region; set only while profiling.
    written_at: float | None = None


class EncoderChannel:
    """Handle bundling the two coordination queues.

    Created by the executor; a reference is handed to the scheduler via
    ``ParallelEncoderScheduler.attach_encoder_channel``. The encoder process
    receives the raw queue objects (they are picklable across the spawn).
    """

    def __init__(self, dispatch_q, completion_q) -> None:
        self.dispatch_q = dispatch_q
        self.completion_q = completion_q

    def dispatch(self, batch: EncodeBatch) -> None:
        self.dispatch_q.put(batch)

    def drain_completions(self) -> list[EncodeDone]:
        """Non-blocking drain of all currently-available completions."""
        done: list[EncodeDone] = []
        while True:
            try:
                done.append(self.completion_q.get_nowait())
            except Exception:
                # queue.Empty (and any transient get error) ends the drain.
                break
        return done
