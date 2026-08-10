# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import numpy as np
import pytest
import torch

from verl.experimental.agent_loop.opsd_memory_agent_loop import align_response_topk_targets
from verl.experimental.agent_loop.trajectory_memory import (
    HashingTextEmbedder,
    TrajectoryMemory,
    TrajectoryRecord,
)


def test_hashing_embedder_prefers_lexically_related_text():
    embedder = HashingTextEmbedder(dimension=512)
    query = np.asarray(embedder.encode("solve the quadratic equation by factoring"))
    related = np.asarray(embedder.encode("factor a quadratic equation and solve its roots"))
    unrelated = np.asarray(embedder.encode("prepare tomato pasta in a saucepan"))

    assert np.dot(query, related) > np.dot(query, unrelated)


def test_memory_is_keyed_by_request_id_and_evicts_oldest_record():
    memory = TrajectoryMemory(capacity=2)
    memory.upsert(TrajectoryRecord("r1", "p1", "t1", "s1", [1.0, 0.0]))
    memory.upsert(TrajectoryRecord("r2", "p2", "t2", "s2", [0.0, 1.0]))
    memory.upsert(TrajectoryRecord("r1", "p1-new", "t1-new", "s1-new", [1.0, 0.0]))
    memory.upsert(TrajectoryRecord("r3", "p3", "t3", "s3", [0.5, 0.5]))

    assert memory.get("r2") is None
    assert memory.get("r1").trajectory == "t1-new"
    record, score = memory.search([1.0, 0.0], exclude_request_id="r1")
    assert record.request_id == "r3"
    assert score == pytest.approx(2**-0.5)


def test_memory_state_round_trip_and_dimension_validation():
    memory = TrajectoryMemory(capacity=3)
    memory.upsert(TrajectoryRecord("r1", "p1", "t1", "s1", [3.0, 4.0]))

    restored = TrajectoryMemory(capacity=1)
    restored.load_state_dict(memory.state_dict())

    assert restored.capacity == 3
    np.testing.assert_allclose(restored.get("r1").embedding, [0.6, 0.8])
    with pytest.raises(ValueError, match="dimension"):
        restored.search([1.0, 0.0, 0.0])


def test_align_response_topk_targets_uses_causal_lm_positions():
    teacher_ids, teacher_logprobs = align_response_topk_targets(
        prompt_length=3,
        topk_ids=[[10, 11], [20, 21]],
        topk_logprobs=[[-0.1, -2.0], [-0.2, -1.8]],
        pad_token_id=0,
    )

    assert teacher_ids.dtype == torch.int32
    assert teacher_ids.tolist() == [
        [0, 0],
        [0, 0],
        [10, 11],
        [20, 21],
        [0, 0],
    ]
    torch.testing.assert_close(
        teacher_logprobs,
        torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [-0.1, -2.0],
                [-0.2, -1.8],
                [0.0, 0.0],
            ]
        ),
    )


def test_align_response_topk_targets_rejects_ragged_distributions():
    with pytest.raises(ValueError, match="Inconsistent top-k width"):
        align_response_topk_targets(
            prompt_length=1,
            topk_ids=[[1, 2], [3]],
            topk_logprobs=[[-0.1, -0.2], [-0.3]],
            pad_token_id=0,
        )
