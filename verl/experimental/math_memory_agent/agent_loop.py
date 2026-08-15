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
"""Experimental math memory AgentLoop with outcome-conditioned summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import partial
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.opsd_memory_base import (
    OPSDMemoryAgentLoopBase,
    OPSDMemoryContext,
    OPSDPromptPair,
    extract_formal_response,
)
from verl.utils.profiler import simple_timer
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

SUCCESS_SUMMARY_TEMPLATE = """The completed math trajectory has been verified as correct.
Tell the solver clearly: "You answered correctly."
Then summarize the reusable successful experience: the key reasoning strategy, decisive intermediate insights,
and checks that made the solution reliable. Be concise and do not merely repeat the trajectory.

Problem:
{prompt_a}

Correct trajectory:
{trajectory}
"""

FAILURE_SUMMARY_TEMPLATE = """The completed math trajectory has been verified as incorrect.
Tell the solver clearly: "Your answer is incorrect."
Then summarize the reusable lessons from the failure: identify likely reasoning, calculation, or verification
mistakes and explain what should be checked or changed next time. Do not present an uncertain step as correct.
Be concise and do not merely repeat the trajectory.

Problem:
{prompt_a}

Incorrect trajectory:
{trajectory}
"""


def normalize_math_ground_truth(value: Any) -> str:
    """Normalize one dataset ground truth to the unboxed form expected by Math-Verify."""
    if value is None:
        raise ValueError("MathOPSDMemoryAgentLoop requires reward_model.ground_truth.")
    if isinstance(value, bool):
        raise TypeError("reward_model.ground_truth must be a math answer, not a boolean.")
    if not isinstance(value, (str, int, float)):
        item = getattr(value, "item", None)
        if item is None:
            raise TypeError(
                "reward_model.ground_truth must be a string or scalar number, "
                f"got {type(value).__name__}."
            )
        value = item()
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise TypeError(
                "reward_model.ground_truth must be a string or scalar number, "
                f"got {type(value).__name__}."
            )

    ground_truth = str(value).strip()
    if not ground_truth:
        raise ValueError("reward_model.ground_truth must not be empty.")

    boxed = last_boxed_only_string(ground_truth)
    if boxed is not None:
        try:
            ground_truth = remove_boxed(boxed).strip()
        except (AssertionError, IndexError):
            # Let Math-Verify fail closed on malformed LaTeX rather than
            # silently changing a ground truth that could not be unboxed.
            pass
    return ground_truth


class MathOPSDMemoryAgentLoop(OPSDMemoryAgentLoopBase):
    """One-shot math OPSD loop whose memory summary depends on answer correctness."""

    def __init__(
        self,
        *args,
        success_summary_template: str = SUCCESS_SUMMARY_TEMPLATE,
        failure_summary_template: str = FAILURE_SUMMARY_TEMPLATE,
        math_verifier_timeout: float = 30.0,
        correctness_threshold: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.success_summary_template = str(success_summary_template)
        self.failure_summary_template = str(failure_summary_template)
        self.math_verifier_timeout = float(math_verifier_timeout)
        self.correctness_threshold = float(correctness_threshold)
        if not self.success_summary_template.strip() or not self.failure_summary_template.strip():
            raise ValueError("Both math summary templates must be non-empty.")
        if self.math_verifier_timeout <= 0:
            raise ValueError(f"math_verifier_timeout must be positive, got {self.math_verifier_timeout}.")
        if not 0 < self.correctness_threshold <= 1:
            raise ValueError(
                f"correctness_threshold must be in (0, 1], got {self.correctness_threshold}."
            )

    @staticmethod
    def _ground_truth_from_sample(kwargs: Mapping[str, Any]) -> str:
        reward_model = kwargs.get("reward_model")
        if not isinstance(reward_model, Mapping):
            raise ValueError(
                "MathOPSDMemoryAgentLoop requires each dataset row to contain a "
                "reward_model mapping with a ground_truth field."
            )
        return normalize_math_ground_truth(reward_model.get("ground_truth"))

    async def _verify_trajectory(self, trajectory: str, ground_truth: str) -> float:
        # The shared scorer runs Math-Verify in a bounded subprocess. Calling it
        # from an executor keeps the AgentLoop event loop responsive meanwhile.
        from verl.utils.reward_score.math_verify import compute_score

        score = await self.loop.run_in_executor(
            None,
            partial(
                compute_score,
                model_output=trajectory,
                ground_truth=ground_truth,
                timeout_score=0.0,
                timeout=self.math_verifier_timeout,
            ),
        )
        return float(score)

    def _summary_template_for_outcome(self, answer_correct: bool) -> str:
        return self.success_summary_template if answer_correct else self.failure_summary_template

    @staticmethod
    def _problem_text_from_messages(messages: Sequence[Mapping[str, Any]]) -> str:
        """Return the original current math problem without chat-template markers."""
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("MathOPSDMemoryAgentLoop requires a non-empty text user message.")
            return content.strip()
        raise ValueError("MathOPSDMemoryAgentLoop requires raw_prompt to contain a user message.")

    @staticmethod
    def _build_paired_trajectory_messages(
        raw_prompt: Sequence[Mapping[str, Any]],
        prompt_pair: OPSDPromptPair,
        assistant_content: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build auditable A/B message histories from the actual rollout contexts."""
        trajectory_a = [deepcopy(dict(message)) for message in raw_prompt]
        if prompt_pair.prompt_b_text is None:
            trajectory_b = deepcopy(trajectory_a)
        else:
            trajectory_b = [{"role": "user", "content": prompt_pair.prompt_b_text}]

        assistant_message = {"role": "assistant", "content": str(assistant_content)}
        trajectory_a.append(deepcopy(assistant_message))
        trajectory_b.append(deepcopy(assistant_message))
        return trajectory_a, trajectory_b

    async def initialize_memory_context(
        self,
        prompt_a_ids: Sequence[int],
        *,
        request_id: str | None = None,
        query_text: str | None = None,
    ) -> OPSDMemoryContext:
        """Retrieve memory while excluding prior private reasoning from prompt B."""
        request_id = request_id or uuid4().hex
        prompt_a_text = query_text or self.tokenizer.decode(list(prompt_a_ids), skip_special_tokens=True)
        retrieved = await self.memory.search.remote(prompt_a_text, exclude_request_id=request_id)
        if retrieved is None:
            return OPSDMemoryContext(request_id=request_id, prompt_a_text=prompt_a_text)
        record = retrieved["record"]
        return OPSDMemoryContext(
            request_id=request_id,
            prompt_a_text=prompt_a_text,
            retrieved_request_id=record["request_id"],
            retrieval_score=retrieved["score"],
            memory_prompt=record["prompt"],
            memory_summary=str(record["summary"]).strip(),
            memory_trajectory=extract_formal_response(record["trajectory"]),
        )

    async def _summarize(
        self,
        *,
        request_id: str,
        prompt_a: str,
        trajectory: str,
        priority: int,
    ) -> str:
        """Generate the memory summary with Qwen thinking disabled."""
        previous_template_kwargs = self.apply_chat_template_kwargs
        self.apply_chat_template_kwargs = {
            **dict(previous_template_kwargs),
            "enable_thinking": False,
        }
        try:
            return await super()._summarize(
                request_id=request_id,
                prompt_a=prompt_a,
                trajectory=trajectory,
                priority=priority,
            )
        finally:
            self.apply_chat_template_kwargs = previous_template_kwargs

    async def finalize_memory(
        self,
        *,
        memory_context: OPSDMemoryContext,
        trajectory: str,
        trajectory_a: list[dict[str, Any]],
        trajectory_b: list[dict[str, Any]],
        ground_truth: str,
        verifier_score: float,
        answer_correct: bool,
        priority: int = 0,
        validate: bool = False,
    ) -> str:
        """Write raw trajectory/summary text without extracting formal responses."""
        generated_summary = await self._summarize(
            request_id=memory_context.request_id,
            prompt_a=memory_context.prompt_a_text,
            trajectory=trajectory,
            priority=int(priority),
        )
        memory_trajectory = str(trajectory).strip()
        memory_summary = str(generated_summary).strip()
        if not validate:
            retrieved_memory = None
            if memory_context.has_memory:
                retrieved_memory = {
                    "request_id": memory_context.retrieved_request_id,
                    "retrieval_score": memory_context.retrieval_score,
                    "prompt": memory_context.memory_prompt,
                    "trajectory": memory_context.memory_trajectory,
                    "summary": memory_context.memory_summary,
                }
            await self.memory.upsert.remote(
                request_id=memory_context.request_id,
                prompt=memory_context.prompt_a_text,
                trajectory=memory_trajectory,
                summary=memory_summary,
                trajectory_a=trajectory_a,
                trajectory_b=trajectory_b,
                ground_truth=ground_truth,
                metadata={
                    "retrieved_request_id": memory_context.retrieved_request_id,
                    "retrieval_score": memory_context.retrieval_score,
                    "retrieved_memory": retrieved_memory,
                    "math_verifier_score": verifier_score,
                    "math_answer_correct": answer_correct,
                },
            )
        return memory_summary

    async def run(
        self,
        sampling_params: dict[str, Any],
        priority: int = 0,
        __validate__: bool = False,
        **kwargs,
    ) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        ground_truth = self._ground_truth_from_sample(kwargs)
        multi_modal_data = await self.process_multi_modal_info(messages)
        if multi_modal_data:
            raise NotImplementedError("MathOPSDMemoryAgentLoop currently supports text-only prompts.")

        prompt_a_ids = await self.initialize_prompt_a(messages)
        problem_text = self._problem_text_from_messages(messages)
        memory_context = await self.initialize_memory_context(prompt_a_ids, query_text=problem_text)

        metrics: dict[str, float] = {}
        with simple_timer("generate_sequences", metrics):
            turn_output = await self.generate_paired(
                prompt_a_ids=prompt_a_ids,
                memory_context=memory_context,
                sampling_params=sampling_params,
                priority=int(priority),
            )

        response_ids = turn_output.token_ids[: self.rollout_config.response_length]
        if not response_ids:
            raise RuntimeError("Math OPSD rollout produced no model tokens to verify or distill.")
        response_mask = [1] * len(response_ids)
        accumulator = self.new_target_accumulator(prompt_a_ids)
        accumulator.record_model_turn(turn_output, response_start=0)
        targets = accumulator.finalize(response_ids=response_ids, response_mask=response_mask)

        trajectory = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        trajectory_a, trajectory_b = self._build_paired_trajectory_messages(
            messages,
            turn_output.prompt_pair,
            trajectory,
        )
        with simple_timer("math_verify", metrics):
            verifier_score = await self._verify_trajectory(trajectory, ground_truth)
        answer_correct = verifier_score >= self.correctness_threshold

        # AgentLoopManager creates one loop instance per trajectory. Selecting
        # the template here keeps the reusable OPSD finalization and memory
        # write-back path unchanged while making this recipe outcome-aware.
        previous_summary_template = self.summary_template
        self.summary_template = self._summary_template_for_outcome(answer_correct)
        try:
            summary = await self.finalize_memory(
                memory_context=memory_context,
                trajectory=trajectory,
                trajectory_a=trajectory_a,
                trajectory_b=trajectory_b,
                ground_truth=ground_truth,
                verifier_score=verifier_score,
                answer_correct=answer_correct,
                priority=int(priority),
                validate=__validate__,
            )
        finally:
            self.summary_template = previous_summary_template

        extra_fields = dict(accumulator.extra_fields)
        extra_fields.update(
            {
                "teacher_ids": targets.fused_topk_ids,
                "teacher_logprobs": targets.fused_topk_logprobs,
                "memory_request_id": memory_context.request_id,
                "retrieved_memory_request_id": memory_context.retrieved_request_id,
                "memory_retrieval_score": memory_context.retrieval_score,
                "trajectory_summary": summary,
                "math_verifier_score": verifier_score,
                "math_answer_correct": answer_correct,
                "math_verifier_seconds": metrics["math_verify"],
                "turn_scores": [],
                "tool_rewards": [],
            }
        )
        return AgentLoopOutput(
            prompt_ids=prompt_a_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=targets.response_logprobs,
            source_topk_ids=targets.source_topk_ids,
            source_topk_logprobs=targets.source_topk_logprobs,
            fused_topk_ids=targets.fused_topk_ids,
            fused_topk_logprobs=targets.fused_topk_logprobs,
            multi_modal_data={},
            num_turns=2,
            metrics=AgentLoopMetrics(
                generate_sequences=metrics["generate_sequences"],
                tool_calls=0.0,
                compute_score=metrics["math_verify"],
                num_preempted=accumulator.num_preempted,
            ),
            extra_fields=extra_fields,
        )


__all__ = [
    "FAILURE_SUMMARY_TEMPLATE",
    "MathOPSDMemoryAgentLoop",
    "SUCCESS_SUMMARY_TEMPLATE",
    "normalize_math_ground_truth",
]
