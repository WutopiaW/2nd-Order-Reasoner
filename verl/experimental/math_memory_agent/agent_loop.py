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

MATH_PROMPT_B_TEMPLATE = r"""You are solving the current math problem. A potentially relevant experience
summarized from a different problem and a reference solution for the current problem are provided as privileged
context.

Current problem:
=== Current Problem Begin ===
{prompt_a}
=== Current Problem End ===

Reference solution for the current problem:
=== Reference Solution Begin ===
{reference_solution}
=== Reference Solution End ===

Retrieved related problem:
=== Retrieved Problem Begin ===
{memory_prompt}
=== Retrieved Problem End ===

Summary of the retrieved experience:
=== Retrieved Summary Begin ===
{memory_summary}
=== Retrieved Summary End ===

Instructions:

1. Determine internally which general strategies or warnings from the retrieved experience apply to the current
   problem.
2. The retrieved experience may be only partially relevant and is not guaranteed to be correct. Verify every
   transferred insight.
3. Any statement that the previous attempt was correct, incorrect, or truncated applies only to the retrieved
   problem.
4. Do not infer or reuse the retrieved problem's final answer.
5. The reference solution belongs to the current problem. Use it as training-time guidance, but verify its steps
   and do not merely copy it.
6. Solve the current problem independently and completely using your own reasoning.
7. Explore alternative approaches, check intermediate results, and backtrack or reconsider when necessary.
8. Do not shorten the solution merely because the retrieved summary or reference solution is concise.
9. Put the final answer within \boxed{{}}.

Now solve the current problem."""

SUCCESS_SUMMARY_TEMPLATE = """The completed math trajectory has been verified as correct.
Tell the solver clearly: "You answered correctly."
Compare the completed trajectory with the reference solution, then summarize the reusable successful experience:
the key reasoning strategy, decisive intermediate insights, and checks that made the solution reliable. Mention
meaningful alternative reasoning when the two solutions differ. Be concise and do not merely repeat either text.

Problem:
{prompt_a}

Correct trajectory:
{trajectory}

Reference solution:
{reference_solution}
"""

FAILURE_SUMMARY_TEMPLATE = """The completed math trajectory has been verified as incorrect.
Tell the solver clearly: "Your answer is incorrect."
Compare the attempted trajectory with the reference solution. Identify the first meaningful divergence, including
reasoning, calculation, or verification mistakes, and explain what should be checked or changed next time.
Summarize the correct reusable strategy without merely copying the reference solution. Do not present an uncertain
step as correct. Be concise.

Problem:
{prompt_a}

Incorrect trajectory:
{trajectory}

Reference solution:
{reference_solution}
"""

TRUNCATED_SUMMARY_TEMPLATE = """The math response below was truncated because it reached the generation
length limit and may be incomplete.
Do not characterize the answer as correct or incorrect.
Compare only the completed portion with the reference solution. Summarize any reusable partial progress: the
approach taken, useful intermediate insights, and the point where the solution became incomplete. Explain what
steps or checks are still needed to finish the solution reliably. Do not claim that reference-only steps were
already produced by the solver. Do not invent missing reasoning or a final answer. Be concise.

Problem:
{prompt_a}

Truncated trajectory:
{trajectory}

Reference solution:
{reference_solution}
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
    """One-shot math OPSD loop with outcome-aware memory summaries."""

    def __init__(
        self,
        *args,
        success_summary_template: str = SUCCESS_SUMMARY_TEMPLATE,
        failure_summary_template: str = FAILURE_SUMMARY_TEMPLATE,
        truncated_summary_template: str = TRUNCATED_SUMMARY_TEMPLATE,
        math_verifier_timeout: float = 30.0,
        correctness_threshold: float = 1.0,
        **kwargs,
    ):
        kwargs.setdefault("prompt_b_template", MATH_PROMPT_B_TEMPLATE)
        super().__init__(*args, **kwargs)
        self.success_summary_template = str(success_summary_template)
        self.failure_summary_template = str(failure_summary_template)
        self.truncated_summary_template = str(truncated_summary_template)
        self.math_verifier_timeout = float(math_verifier_timeout)
        self.correctness_threshold = float(correctness_threshold)
        if not all(
            template.strip()
            for template in (
                self.success_summary_template,
                self.failure_summary_template,
                self.truncated_summary_template,
            )
        ):
            raise ValueError("The success, failure, and truncated math summary templates must be non-empty.")
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

    @staticmethod
    def _reference_solution_from_sample(kwargs: Mapping[str, Any], *, required: bool) -> str:
        extra_info = kwargs.get("extra_info")
        if not isinstance(extra_info, Mapping):
            if required:
                raise ValueError(
                    "MathOPSDMemoryAgentLoop requires each training row to contain an "
                    "extra_info mapping with a solution field."
                )
            return ""
        solution = extra_info.get("solution")
        if not isinstance(solution, str) or not solution.strip():
            if required:
                raise ValueError(
                    "MathOPSDMemoryAgentLoop requires each training row to contain a non-empty "
                    "extra_info.solution string."
                )
            return ""
        return solution.strip()

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

    @staticmethod
    def _summary_outcome(*, answer_correct: bool, response_truncated: bool) -> str:
        if response_truncated:
            return "truncated"
        return "correct" if answer_correct else "incorrect"

    def _summary_template_for_outcome(
        self,
        answer_correct: bool,
        *,
        response_truncated: bool = False,
    ) -> str:
        if response_truncated:
            return self.truncated_summary_template
        return self.success_summary_template if answer_correct else self.failure_summary_template

    @staticmethod
    def _response_was_truncated(turn_output: OPSDTurnOutput, *, response_limit: int) -> bool:
        """Detect backend length stops and the loop's own final response cap."""
        return turn_output.stop_reason == "length" or len(turn_output.token_ids) > int(response_limit)

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

    @staticmethod
    def _should_distill_prompt_b(prompt_pair: OPSDPromptPair, *, validate: bool) -> bool:
        """Return B only when it is a distinct training context."""
        return not validate and prompt_pair.prompt_b_text is not None

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
        reference_solution: str,
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
                summary_extra_fields={"reference_solution": reference_solution},
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
        reference_solution: str,
        verifier_score: float,
        answer_correct: bool,
        response_truncated: bool,
        summary_outcome: str,
        rollout_stop_reason: str | None,
        priority: int = 0,
        validate: bool = False,
    ) -> str:
        """Write raw trajectory/summary text without extracting formal responses."""
        generated_summary = await self._summarize(
            request_id=memory_context.request_id,
            prompt_a=memory_context.prompt_a_text,
            trajectory=trajectory,
            reference_solution=reference_solution,
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
                    "math_response_truncated": response_truncated,
                    "math_summary_outcome": summary_outcome,
                    "rollout_stop_reason": rollout_stop_reason,
                },
            )
        return memory_summary

    async def run(
        self,
        sampling_params: dict[str, Any],
        priority: int = 0,
        __validate__: bool = False,
        **kwargs,
    ) -> AgentLoopOutput | list[AgentLoopOutput]:
        messages = list(kwargs["raw_prompt"])
        ground_truth = self._ground_truth_from_sample(kwargs)
        reference_solution = self._reference_solution_from_sample(kwargs, required=not __validate__)
        prompt_b_reference_solution = "" if __validate__ else reference_solution
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
                prompt_b_extra_fields={"reference_solution": prompt_b_reference_solution},
                force_prompt_b=bool(prompt_b_reference_solution),
            )

        response_limit = int(self.rollout_config.response_length)
        response_truncated = self._response_was_truncated(turn_output, response_limit=response_limit)
        response_ids = turn_output.token_ids[:response_limit]
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
        training_response_mask = response_mask if (__validate__ or answer_correct) else [0] * len(response_mask)
        summary_outcome = self._summary_outcome(
            answer_correct=answer_correct,
            response_truncated=response_truncated,
        )

        # AgentLoopManager creates one loop instance per trajectory. Selecting
        # the template here keeps the reusable OPSD finalization and memory
        # write-back path unchanged while making this recipe outcome-aware.
        previous_summary_template = self.summary_template
        self.summary_template = self._summary_template_for_outcome(
            answer_correct,
            response_truncated=response_truncated,
        )
        try:
            summary = await self.finalize_memory(
                memory_context=memory_context,
                trajectory=trajectory,
                trajectory_a=trajectory_a,
                trajectory_b=trajectory_b,
                ground_truth=ground_truth,
                reference_solution=reference_solution,
                verifier_score=verifier_score,
                answer_correct=answer_correct,
                response_truncated=response_truncated,
                summary_outcome=summary_outcome,
                rollout_stop_reason=turn_output.stop_reason,
                priority=int(priority),
                validate=__validate__,
            )
        finally:
            self.summary_template = previous_summary_template

        shared_extra_fields = {
            "memory_request_id": memory_context.request_id,
            "retrieved_memory_request_id": memory_context.retrieved_request_id,
            "memory_retrieval_score": memory_context.retrieval_score,
            "trajectory_summary": summary,
            "math_verifier_score": verifier_score,
            "math_answer_correct": answer_correct,
            "math_response_truncated": response_truncated,
            "math_summary_outcome": summary_outcome,
            "rollout_stop_reason": turn_output.stop_reason,
            "math_verifier_seconds": metrics["math_verify"],
            "turn_scores": [],
            "tool_rewards": [],
        }
        extra_fields_a = dict(accumulator.extra_fields)
        extra_fields_a.update(
            {
                "teacher_ids": targets.fused_topk_ids,
                "teacher_logprobs": targets.fused_topk_logprobs,
                "distillation_prompt_variant": "A",
                **shared_extra_fields,
            }
        )
        output_a = AgentLoopOutput(
            prompt_ids=prompt_a_ids,
            response_ids=response_ids,
            response_mask=training_response_mask,
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
            extra_fields=extra_fields_a,
        )

        # Validation keeps A as the sole output so reward and reporting semantics
        # remain unchanged. A missing prompt-B text means B reused A exactly, so
        # returning it again would only double this sample's training weight.
        if not self._should_distill_prompt_b(turn_output.prompt_pair, validate=__validate__):
            return output_a

        accumulator_b = self.new_target_accumulator(turn_output.prompt_pair.prompt_b_ids)
        accumulator_b.record_model_turn(turn_output, response_start=0)
        targets_b = accumulator_b.finalize(response_ids=response_ids, response_mask=response_mask)
        extra_fields_b = dict(accumulator_b.extra_fields)
        extra_fields_b.update(
            {
                "teacher_ids": targets_b.fused_topk_ids,
                "teacher_logprobs": targets_b.fused_topk_logprobs,
                "distillation_prompt_variant": "B",
                **shared_extra_fields,
            }
        )
        output_b = AgentLoopOutput(
            prompt_ids=turn_output.prompt_pair.prompt_b_ids,
            response_ids=response_ids,
            response_mask=training_response_mask,
            response_logprobs=targets_b.response_logprobs,
            fused_topk_ids=targets_b.fused_topk_ids,
            fused_topk_logprobs=targets_b.fused_topk_logprobs,
            multi_modal_data={},
            num_turns=2,
            metrics=AgentLoopMetrics(),
            extra_fields=extra_fields_b,
        )

        # V1 treats the last item as the session's final output for reward,
        # advantage, and reporting. Keep A last to preserve the existing
        # non-privileged evaluation semantics while training on both contexts.
        return [output_b, output_a]


__all__ = [
    "FAILURE_SUMMARY_TEMPLATE",
    "MATH_PROMPT_B_TEMPLATE",
    "MathOPSDMemoryAgentLoop",
    "SUCCESS_SUMMARY_TEMPLATE",
    "TRUNCATED_SUMMARY_TEMPLATE",
    "normalize_math_ground_truth",
]
