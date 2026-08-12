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

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.experimental.agent_loop.opsd_memory_agent_loop import (
    OPSDMemoryAgentLoop,
    align_response_topk_targets,
    compute_paired_generation_budget,
    decode_complete_generated_text,
)
from verl.experimental.agent_loop.trajectory_memory import (
    HashingTextEmbedder,
    TrajectoryMemory,
    TrajectoryRecord,
    append_memory_record,
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


def test_append_memory_record_writes_readable_jsonl_without_embedding(tmp_path):
    output_path = tmp_path / "memory" / "trajectories.jsonl"
    record = TrajectoryRecord(
        request_id="request-1",
        prompt="current problem",
        trajectory="reasoning trajectory",
        summary="reusable summary",
        embedding=[0.6, 0.8],
        metadata={"retrieved_request_id": "request-0", "retrieval_score": 0.75},
    )

    append_memory_record(output_path, record, memory_size=2)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == {
        "request_id": "request-1",
        "prompt": "current problem",
        "trajectory": "reasoning trajectory",
        "summary": "reusable summary",
        "metadata": {"retrieved_request_id": "request-0", "retrieval_score": 0.75},
        "memory_size": 2,
    }


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


def test_decode_complete_generated_text_drops_only_incomplete_trailing_tokens():
    class ByteBoundaryTokenizer:
        @staticmethod
        def decode(token_ids, skip_special_tokens=True):
            del skip_special_tokens
            pieces = {1: "complete ", 2: "text", 3: "\ufffd", 4: "   "}
            return "".join(pieces[token_id] for token_id in token_ids)

    assert decode_complete_generated_text(ByteBoundaryTokenizer(), [1, 2, 3, 4]) == "complete text"
    assert decode_complete_generated_text(ByteBoundaryTokenizer(), [1, 3, 2]) == "complete \ufffdtext"


def test_prompt_b_length_does_not_reduce_prompt_a_generation_budget():
    generation_limit, prompt_b_budget = compute_paired_generation_budget(
        prompt_a_length=128,
        prompt_a_max_total_tokens=1024,
        response_length=1024,
        physical_context_length=4096,
    )

    assert generation_limit == 896
    assert prompt_b_budget == 3199
    assert prompt_b_budget >= 2048
    assert 2048 + generation_limit + 1 <= 4096


@pytest.mark.asyncio
async def test_prompt_b_can_exceed_generic_prompt_cap_without_truncation():
    class WhitespaceTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            del add_special_tokens
            return text.split()

        @staticmethod
        def decode(token_ids, skip_special_tokens=True):
            del skip_special_tokens
            return " ".join(token_ids)

    cap_prompt_length_values = []

    async def apply_chat_template(messages, cap_prompt_length=True):
        cap_prompt_length_values.append(cap_prompt_length)
        return messages[0]["content"].split()

    loop = SimpleNamespace(
        tokenizer=WhitespaceTokenizer(),
        apply_chat_template=apply_chat_template,
    )
    prompt_a = " ".join(["a"] * 128)
    memory_trajectory = " ".join(["trajectory"] * 1800)
    memory_summary = " ".join(["summary"] * 120)

    _, prompt_b_ids = await OPSDMemoryAgentLoop._render_prompt_with_budget(
        loop,
        template="{prompt_a} {memory_trajectory} {memory_summary}",
        fields={
            "prompt_a": prompt_a,
            "memory_trajectory": memory_trajectory,
            "memory_summary": memory_summary,
        },
        trim_order=("memory_trajectory", "memory_summary"),
        max_prompt_tokens=3199,
    )

    assert len(prompt_b_ids) == 2048
    assert cap_prompt_length_values == [False]


def test_physical_context_must_preserve_prompt_a_generation_budget():
    with pytest.raises(ValueError, match="cannot preserve prompt A"):
        compute_paired_generation_budget(
            prompt_a_length=128,
            prompt_a_max_total_tokens=1024,
            response_length=1024,
            physical_context_length=900,
        )
