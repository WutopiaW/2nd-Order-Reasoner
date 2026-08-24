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

from verl.experimental.agent_loop.opsd_memory_base import (
    OPSDMemoryAgentLoopBase,
    OPSDMemoryContext,
    OPSDPromptPair,
)
from verl.experimental.math_memory_agent.agent_loop import (
    FAILURE_SUMMARY_TEMPLATE,
    MATH_PROMPT_B_TEMPLATE,
    SUCCESS_SUMMARY_TEMPLATE,
    TRUNCATED_SUMMARY_TEMPLATE,
    MathOPSDMemoryAgentLoop,
    normalize_math_ground_truth,
)


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


def test_reference_solution_is_loaded_from_extra_info():
    kwargs = {"extra_info": {"solution": "  reference reasoning  "}}

    assert (
        MathOPSDMemoryAgentLoop._reference_solution_from_sample(kwargs, required=True)
        == "reference reasoning"
    )


@pytest.mark.parametrize("extra_info", [None, {}, {"solution": None}, {"solution": "   "}])
def test_reference_solution_fails_closed_for_training(extra_info):
    with pytest.raises(ValueError, match="extra_info"):
        MathOPSDMemoryAgentLoop._reference_solution_from_sample(
            {"extra_info": extra_info},
            required=True,
        )


def test_reference_solution_is_optional_for_validation():
    assert MathOPSDMemoryAgentLoop._reference_solution_from_sample({}, required=False) == ""


def test_math_templates_accept_reference_solution():
    fields = {
        "prompt_a": "problem",
        "trajectory": "attempt",
        "reference_solution": "reference",
        "memory_prompt": "previous problem",
        "memory_summary": "previous summary",
        "memory_trajectory": "previous trajectory",
    }

    for template in (SUCCESS_SUMMARY_TEMPLATE, FAILURE_SUMMARY_TEMPLATE, TRUNCATED_SUMMARY_TEMPLATE):
        rendered = template.format(**fields)
        assert "Reference solution:\nreference" in rendered
    prompt_b = MATH_PROMPT_B_TEMPLATE.format(**fields)
    assert "=== Current Problem Begin ===\nproblem\n=== Current Problem End ===" in prompt_b
    assert "=== Reference Solution Begin ===\nreference\n=== Reference Solution End ===" in prompt_b
    assert "=== Retrieved Problem Begin ===\nprevious problem\n=== Retrieved Problem End ===" in prompt_b
    assert "=== Retrieved Summary Begin ===\nprevious summary\n=== Retrieved Summary End ===" in prompt_b
    assert "previous trajectory" not in prompt_b
    assert r"Put the final answer within \boxed{}." in prompt_b


@pytest.mark.asyncio
async def test_prompt_b_can_be_forced_with_reference_solution_before_memory_warmup():
    class WhitespaceTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            del add_special_tokens
            return text.split()

        @staticmethod
        def decode(token_ids, skip_special_tokens=True):
            del skip_special_tokens
            return " ".join(token_ids)

    async def apply_chat_template(messages, cap_prompt_length=True):
        assert cap_prompt_length is False
        return messages[0]["content"].split()

    loop = SimpleNamespace(
        tokenizer=WhitespaceTokenizer(),
        apply_chat_template=apply_chat_template,
        prompt_b_template=MATH_PROMPT_B_TEMPLATE,
        memory_prompt_max_length=512,
    )
    pair = await OPSDMemoryAgentLoopBase.initialize_prompt_pair(
        loop,
        prompt_a_ids=["current", "problem"],
        memory_context=OPSDMemoryContext(request_id="request-1", prompt_a_text="current problem"),
        max_new_tokens=32,
        prompt_b_extra_fields={"reference_solution": "gold reasoning and answer"},
        force_prompt_b=True,
    )

    assert pair.prompt_b_text is not None
    assert "gold reasoning and answer" in pair.prompt_b_text
    assert pair.prompt_b_ids != pair.prompt_a_ids


def test_solution_conditioned_prompt_b_remains_a_training_sample():
    prompt_pair = OPSDPromptPair(
        prompt_a_ids=[1, 2],
        prompt_b_ids=[3, 4, 5],
        max_new_tokens=32,
        prompt_b_text="problem plus reference solution",
    )

    assert MathOPSDMemoryAgentLoop._should_distill_prompt_b(prompt_pair, validate=False)
    assert not MathOPSDMemoryAgentLoop._should_distill_prompt_b(prompt_pair, validate=True)


def test_summary_template_depends_on_verifier_outcome():
    loop = object.__new__(MathOPSDMemoryAgentLoop)
    loop.success_summary_template = "success summary"
    loop.failure_summary_template = "failure summary"
    loop.truncated_summary_template = "truncated summary"

    assert loop._summary_template_for_outcome(True) == "success summary"
    assert loop._summary_template_for_outcome(False) == "failure summary"
    assert loop._summary_template_for_outcome(True, response_truncated=True) == "truncated summary"
    assert loop._summary_template_for_outcome(False, response_truncated=True) == "truncated summary"
    assert loop._summary_outcome(answer_correct=True, response_truncated=False) == "correct"
    assert loop._summary_outcome(answer_correct=False, response_truncated=False) == "incorrect"
    assert loop._summary_outcome(answer_correct=True, response_truncated=True) == "truncated"


@pytest.mark.parametrize(
    ("stop_reason", "token_count", "response_limit", "expected"),
    [
        ("length", 3, 8, True),
        ("stop", 9, 8, True),
        ("stop", 8, 8, False),
        (None, 3, 8, False),
    ],
)
def test_response_truncation_covers_backend_and_local_caps(stop_reason, token_count, response_limit, expected):
    turn_output = SimpleNamespace(stop_reason=stop_reason, token_ids=list(range(token_count)))

    assert (
        MathOPSDMemoryAgentLoop._response_was_truncated(turn_output, response_limit=response_limit) is expected
    )


def test_problem_text_uses_original_user_message_without_template_markers():
    messages = [
        {"role": "system", "content": "Answer carefully."},
        {"role": "user", "content": "  What is 1 + 1?  "},
    ]

    assert MathOPSDMemoryAgentLoop._problem_text_from_messages(messages) == "What is 1 + 1?"


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "system", "content": "No user message."}],
        [{"role": "user", "content": "   "}],
        [{"role": "user", "content": ["not", "text"]}],
    ],
)
def test_problem_text_rejects_missing_or_non_text_user_message(messages):
    with pytest.raises(ValueError, match="user message"):
        MathOPSDMemoryAgentLoop._problem_text_from_messages(messages)


def test_paired_trajectory_messages_preserve_thinking_and_actual_prompt_b():
    raw_prompt = [
        {"role": "system", "content": "Solve carefully."},
        {"role": "user", "content": "What is 1 + 1?"},
    ]
    pair = OPSDPromptPair(
        prompt_a_ids=[1, 2],
        prompt_b_ids=[3, 4],
        max_new_tokens=32,
        prompt_b_text="Use this trimmed memory, then solve 1 + 1.",
    )

    trajectory_a, trajectory_b = MathOPSDMemoryAgentLoop._build_paired_trajectory_messages(
        raw_prompt,
        pair,
        "<think>One plus one is two.</think>\\boxed{2}",
    )

    assert trajectory_a == [
        {"role": "system", "content": "Solve carefully."},
        {"role": "user", "content": "What is 1 + 1?"},
        {"role": "assistant", "content": "<think>One plus one is two.</think>\\boxed{2}"},
    ]
    assert trajectory_b == [
        {"role": "user", "content": "Use this trimmed memory, then solve 1 + 1."},
        {"role": "assistant", "content": "<think>One plus one is two.</think>\\boxed{2}"},
    ]
    assert "<|im_end|>" not in trajectory_a[-1]["content"]
    assert "<|im_end|>" not in trajectory_b[-1]["content"]
    assert raw_prompt[-1]["content"] == "What is 1 + 1?"


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
        assert kwargs["trajectory"] == "<think>reasoning remains in the summary input</think>answer"
        assert kwargs["summary_extra_fields"] == {"reference_solution": "reference reasoning"}
        return "<think>unexpected but preserved</think>summary"

    monkeypatch.setattr(OPSDMemoryAgentLoopBase, "_summarize", fake_base_summarize)
    summary = await loop._summarize(
        request_id="request-1",
        prompt_a="problem",
        trajectory="<think>reasoning remains in the summary input</think>answer",
        reference_solution="reference reasoning",
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
        trajectory_a=[
            {"role": "user", "content": "problem"},
            {"role": "assistant", "content": "<think>A reasoning</think>\\boxed{1}"},
        ],
        trajectory_b=[
            {"role": "user", "content": "memory plus problem"},
            {"role": "assistant", "content": "<think>B reasoning</think>\\boxed{1}"},
        ],
        ground_truth="1",
        reference_solution="reference reasoning and answer",
        verifier_score=1.0,
        answer_correct=True,
        response_truncated=False,
        summary_outcome="correct",
        rollout_stop_reason="stop",
    )

    assert captured["trajectory"] == "<think>solution reasoning</think>\\boxed{1}"
    assert captured["summary"] == "<think>summary reasoning</think>final summary"
    assert captured["trajectory_a"][-1]["content"] == "<think>A reasoning</think>\\boxed{1}"
    assert captured["trajectory_b"][-1]["content"] == "<think>B reasoning</think>\\boxed{1}"
    assert captured["ground_truth"] == "1"
    assert captured["metadata"]["math_verifier_score"] == 1.0
    assert captured["metadata"]["math_answer_correct"] is True
    assert captured["metadata"]["math_response_truncated"] is False
    assert captured["metadata"]["math_summary_outcome"] == "correct"
    assert captured["metadata"]["rollout_stop_reason"] == "stop"
    assert summary == captured["summary"]


@pytest.mark.asyncio
async def test_memory_retrieval_removes_trajectory_thinking_before_prompt_b():
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

    assert context.memory_trajectory == "\\boxed{2}"
    assert context.memory_summary == "<think>unexpected summary thinking</think>lesson"
