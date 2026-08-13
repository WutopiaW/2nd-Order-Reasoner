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

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.experimental.agent_loop.opsd_memory_base import (
    OPSDMemoryAgentLoopBase,
    OPSDMemoryContext,
    OPSDPromptPair,
    OPSDTrajectoryTargetAccumulator,
    OPSDTurnOutput,
)
from verl.experimental.agent_loop.trajectory_memory import (
    HashingTextEmbedder,
    TrajectoryMemory,
    TrajectoryRecord,
    append_memory_record,
)


def test_opsd_base_leaves_run_to_the_concrete_agent_loop():
    assert "run" not in OPSDMemoryAgentLoopBase.__dict__
    assert inspect.isabstract(OPSDMemoryAgentLoopBase)


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


def _turn(token_ids, *, first_target_id):
    length = len(token_ids)
    pair = OPSDPromptPair(prompt_a_ids=[1], prompt_b_ids=[1, 2], max_new_tokens=length)
    return OPSDTurnOutput(
        prompt_pair=pair,
        token_ids=token_ids,
        source_logprobs=[-0.3] * length,
        fused_logprobs=[-0.2] * length,
        source_topk_ids=[[first_target_id + i, 90] for i in range(length)],
        source_topk_logprobs=[[-0.3, -2.0] for _ in range(length)],
        fused_topk_ids=[[first_target_id + i, 91] for i in range(length)],
        fused_topk_logprobs=[[-0.2, -1.8] for _ in range(length)],
        stop_reason="stop",
        num_preempted=0,
    )


def test_multiturn_targets_skip_tool_tokens_and_use_causal_lm_positions():
    accumulator = OPSDTrajectoryTargetAccumulator(prompt_ids=[1, 2, 3], target_topk=2, pad_token_id=0)
    accumulator.record_model_turn(_turn([10, 11], first_target_id=10), response_start=0)
    accumulator.record_model_turn(_turn([20], first_target_id=20), response_start=4)

    targets = accumulator.finalize(
        response_ids=[10, 11, 70, 71, 20],
        response_mask=[1, 1, 0, 0, 1],
    )

    assert targets.source_topk_ids.dtype == torch.int32
    assert targets.source_topk_ids.tolist() == [
        [0, 0],
        [0, 0],
        [10, 90],
        [11, 90],
        [0, 0],
        [0, 0],
        [20, 90],
        [0, 0],
    ]
    assert targets.response_logprobs == pytest.approx([-0.2, -0.2, 0.0, 0.0, -0.2])
    torch.testing.assert_close(targets.fused_topk_logprobs[4:6], torch.zeros((2, 2)))


def test_multiturn_targets_fail_when_a_model_position_has_no_pds_target():
    accumulator = OPSDTrajectoryTargetAccumulator(prompt_ids=[1], target_topk=2, pad_token_id=0)
    accumulator.record_model_turn(_turn([10], first_target_id=10), response_start=0)

    with pytest.raises(ValueError, match="do not cover exactly"):
        accumulator.finalize(response_ids=[10, 11], response_mask=[1, 1])


@pytest.mark.asyncio
async def test_prompt_pair_uses_explicit_max_new_tokens_without_prompt_a_budget_math():
    loop = SimpleNamespace(memory_prompt_max_length=2048)
    prompt_a_ids = list(range(3000))

    pair = await OPSDMemoryAgentLoopBase.initialize_prompt_pair(
        loop,
        prompt_a_ids=prompt_a_ids,
        memory_context=OPSDMemoryContext(request_id="request-1", prompt_a_text="problem"),
        max_new_tokens=777,
    )

    assert pair.prompt_a_ids == prompt_a_ids
    assert pair.prompt_b_ids == prompt_a_ids
    assert pair.max_new_tokens == 777


@pytest.mark.asyncio
async def test_initial_prompt_a_is_checked_after_chat_template_without_silent_truncation():
    cap_prompt_length_values = []

    async def apply_chat_template(messages, cap_prompt_length=True, **kwargs):
        del messages, kwargs
        cap_prompt_length_values.append(cap_prompt_length)
        return [1, 2, 3, 4, 5]

    loop = SimpleNamespace(
        apply_chat_template=apply_chat_template,
        rollout_config=SimpleNamespace(prompt_length=4),
    )
    with pytest.raises(ValueError, match="after applying the chat template"):
        await OPSDMemoryAgentLoopBase.initialize_prompt_a(loop, [{"role": "user", "content": "x"}])

    assert cap_prompt_length_values == [False]


@pytest.mark.asyncio
async def test_prompt_b_is_trimmed_to_explicit_memory_prompt_cap():
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

    _, prompt_b_ids = await OPSDMemoryAgentLoopBase._render_prompt_with_budget(
        loop,
        template="{prompt_a} {memory_trajectory} {memory_summary}",
        fields={
            "prompt_a": prompt_a,
            "memory_trajectory": memory_trajectory,
            "memory_summary": memory_summary,
        },
        trim_order=("memory_trajectory", "memory_summary"),
        max_prompt_tokens=512,
    )

    assert len(prompt_b_ids) <= 512
    assert len(prompt_b_ids) > 128
    assert cap_prompt_length_values and all(value is False for value in cap_prompt_length_values)
