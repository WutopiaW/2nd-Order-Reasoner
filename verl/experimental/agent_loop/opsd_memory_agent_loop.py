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
"""Single-turn OPSD memory AgentLoop built from the reusable base primitives."""

from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.opsd_memory_base import (
    DEFAULT_PROMPT_B_TEMPLATE,
    DEFAULT_SUMMARY_TEMPLATE,
    OPSDMemoryAgentLoopBase,
    OPSDTrajectoryTargetAccumulator,
    OPSDTrajectoryTargets,
    OPSDTurnOutput,
)
from verl.utils.profiler import simple_timer


class SingleTurnOPSDMemoryAgentLoop(OPSDMemoryAgentLoopBase):
    """Math-style one-shot specialization of :class:`OPSDMemoryAgentLoopBase`."""

    async def run(
        self,
        sampling_params: dict[str, Any],
        priority: int = 0,
        __validate__: bool = False,
        **kwargs,
    ) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_multi_modal_info(messages)
        if multi_modal_data:
            raise NotImplementedError("SingleTurnOPSDMemoryAgentLoop currently supports text-only prompts.")

        # Render once without AgentLoopBase's silent text truncation, then fail
        # closed against rollout.prompt_length inside initialize_prompt_a().
        prompt_a_ids = await self.initialize_prompt_a(messages)
        memory_context = await self.initialize_memory_context(prompt_a_ids)

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
            raise RuntimeError("Single-turn OPSD rollout produced no model tokens to distill.")
        response_mask = [1] * len(response_ids)
        accumulator = self.new_target_accumulator(prompt_a_ids)
        accumulator.record_model_turn(turn_output, response_start=0)
        targets = accumulator.finalize(
            response_ids=response_ids,
            response_mask=response_mask,
        )

        trajectory = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        summary = await self.finalize_memory(
            memory_context=memory_context,
            trajectory=trajectory,
            priority=int(priority),
            validate=__validate__,
        )
        extra_fields = dict(accumulator.extra_fields)
        extra_fields.update(
            {
                # Existing distillation code consumes the fused distribution
                # through these teacher aliases.
                "teacher_ids": targets.fused_topk_ids,
                "teacher_logprobs": targets.fused_topk_logprobs,
                "memory_request_id": memory_context.request_id,
                "retrieved_memory_request_id": memory_context.retrieved_request_id,
                "memory_retrieval_score": memory_context.retrieval_score,
                "trajectory_summary": summary,
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
                compute_score=0.0,
                num_preempted=accumulator.num_preempted,
            ),
            extra_fields=extra_fields,
        )


__all__ = [
    "DEFAULT_PROMPT_B_TEMPLATE",
    "DEFAULT_SUMMARY_TEMPLATE",
    "OPSDMemoryAgentLoopBase",
    "OPSDTrajectoryTargetAccumulator",
    "OPSDTrajectoryTargets",
    "OPSDTurnOutput",
    "SingleTurnOPSDMemoryAgentLoop",
]
