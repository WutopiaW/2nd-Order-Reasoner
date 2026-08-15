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

from types import SimpleNamespace

import pytest

from verl.experimental.agent_loop.opsd_memory_base import OPSDMemoryAgentLoopBase, OPSDMemoryContext
from verl.experimental.math_memory_agent.agent_loop import MathOPSDMemoryAgentLoop, normalize_math_ground_truth


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("42", "42"),
        (42, "42"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"Work gives $\boxed{7}$.", "7"),
    ],
)
def test_normalize_math_ground_truth(value, expected):
    assert normalize_math_ground_truth(value) == expected


@pytest.mark.parametrize("value", [None, "", "   ", True, ["1", "2"]])
def test_normalize_math_ground_truth_rejects_unsupported_values(value):
    with pytest.raises((TypeError, ValueError)):
        normalize_math_ground_truth(value)


def test_summary_template_depends_on_verifier_outcome():
    loop = object.__new__(MathOPSDMemoryAgentLoop)
    loop.success_summary_template = "success summary"
    loop.failure_summary_template = "failure summary"

    assert loop._summary_template_for_outcome(True) == "success summary"
    assert loop._summary_template_for_outcome(False) == "failure summary"


@pytest.mark.asyncio
async def test_summary_uses_qwen_no_thinking_without_posthoc_extraction(monkeypatch):
    loop = object.__new__(MathOPSDMemoryAgentLoop)
    original_kwargs = {"custom_flag": "kept"}
    loop.apply_chat_template_kwargs = original_kwargs

    async def fake_base_summarize(self, **kwargs):
        assert self.apply_chat_template_kwargs == {
            "custom_flag": "kept",
            "enable_thinking": False,
        }
        return "<think>unexpected but preserved</think>summary"

    monkeypatch.setattr(OPSDMemoryAgentLoopBase, "_summarize", fake_base_summarize)
    summary = await loop._summarize(
        request_id="request-1",
        prompt_a="problem",
        trajectory="trajectory",
        priority=0,
    )

    assert summary == "<think>unexpected but preserved</think>summary"
    assert loop.apply_chat_template_kwargs is original_kwargs


@pytest.mark.asyncio
async def test_finalize_memory_preserves_raw_trajectory_and_summary():
    captured = {}

    class FakeUpsert:
        async def remote(self, **kwargs):
            captured.update(kwargs)

    loop = object.__new__(MathOPSDMemoryAgentLoop)
    loop.memory = SimpleNamespace(upsert=FakeUpsert())

    async def fake_summarize(**kwargs):
        return "  <think>summary reasoning</think>final summary  "

    loop._summarize = fake_summarize
    context = OPSDMemoryContext(request_id="request-1", prompt_a_text="problem")
    summary = await loop.finalize_memory(
        memory_context=context,
        trajectory="  <think>solution reasoning</think>\\boxed{1}  ",
    )

    assert captured["trajectory"] == "<think>solution reasoning</think>\\boxed{1}"
    assert captured["summary"] == "<think>summary reasoning</think>final summary"
    assert summary == captured["summary"]


@pytest.mark.asyncio
async def test_memory_retrieval_preserves_raw_thinking_blocks():
    class FakeSearch:
        async def remote(self, *args, **kwargs):
            return {
                "score": 0.9,
                "record": {
                    "request_id": "previous-request",
                    "prompt": "previous problem",
                    "trajectory": "<think>previous reasoning</think>\\boxed{2}",
                    "summary": "<think>unexpected summary thinking</think>lesson",
                },
            }

    loop = object.__new__(MathOPSDMemoryAgentLoop)
    loop.memory = SimpleNamespace(search=FakeSearch())
    context = await loop.initialize_memory_context(
        [1, 2],
        request_id="request-1",
        query_text="current problem",
    )

    assert context.memory_trajectory == "<think>previous reasoning</think>\\boxed{2}"
    assert context.memory_summary == "<think>unexpected summary thinking</think>lesson"
