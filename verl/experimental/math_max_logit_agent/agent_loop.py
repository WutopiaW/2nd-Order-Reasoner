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
"""Math memory AgentLoop that distills max-fused raw logits into prompt A."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.opsd_memory_base import OPSDMemoryContext, OPSDPromptPair
from verl.experimental.math_memory_agent.agent_loop import MathOPSDMemoryAgentLoop
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput


@dataclass(frozen=True)
class MaxLogitTurnOutput:
    prompt_pair: OPSDPromptPair
    token_ids: list[int]
    fused_topk_ids: list[list[int]]
    fused_topk_logits: list[list[float]]
    stop_reason: str | None
    num_preempted: int | None
    extra_fields: dict[str, Any]


class MathMaxLogitDistillationAgentLoop(MathOPSDMemoryAgentLoop):
    """Reuse math memory/verification while training only A on max-fused logits."""

    async def generate_paired(
        self,
        *,
        prompt_a_ids: Sequence[int],
        memory_context: OPSDMemoryContext,
        sampling_params: dict[str, Any],
        priority: int = 0,
    ) -> MaxLogitTurnOutput:
        max_new_tokens = sampling_params.get(
            "max_new_tokens",
            sampling_params.get("max_tokens", self.rollout_config.response_length),
        )
        pair = await self.initialize_prompt_pair(
            prompt_a_ids=prompt_a_ids,
            memory_context=memory_context,
            max_new_tokens=max_new_tokens,
        )
        sample_group = f"opsd-max-{memory_context.request_id}-{uuid4().hex}"
        base_params = dict(sampling_params)
        base_params.pop("max_tokens", None)
        base_params["max_new_tokens"] = pair.max_new_tokens
        base_params["logprobs"] = False

        def paired_params(weight: float, *, return_targets: bool) -> dict[str, Any]:
            params = dict(base_params)
            custom_params = dict(params.get("custom_params") or {})
            custom_params.update(
                {
                    "__pds_sample_group": sample_group,
                    "__pds_fuse_method": "max_logits",
                    "__pds_fuse_weight": weight,
                }
            )
            for key in ("__pds_return_prob_trajectory", "__pds_return_top_k"):
                custom_params.pop(key, None)
            if return_targets:
                custom_params["__pds_return_top_k_logits"] = self.target_topk
            else:
                custom_params.pop("__pds_return_top_k_logits", None)
            params["custom_params"] = custom_params
            return params

        outputs = await self.server_manager.generate_group(
            request_id=memory_context.request_id,
            requests=[
                {
                    "prompt_ids": pair.prompt_a_ids,
                    "sampling_params": paired_params(self.prompt_a_fuse_weight, return_targets=True),
                    "priority": int(priority),
                },
                {
                    "prompt_ids": pair.prompt_b_ids,
                    "sampling_params": paired_params(self.prompt_b_fuse_weight, return_targets=False),
                    "priority": int(priority),
                },
            ],
        )
        return self.process_max_logit_outputs(outputs, prompt_pair=pair)

    def process_max_logit_outputs(
        self,
        outputs: Sequence[TokenOutput],
        *,
        prompt_pair: OPSDPromptPair,
    ) -> MaxLogitTurnOutput:
        if len(outputs) != 2:
            raise RuntimeError(f"mix-sglang returned {len(outputs)} outputs for a two-member PDS group.")
        output_a, output_b = outputs
        if output_a.stop_reason == "aborted" or output_b.stop_reason == "aborted":
            raise RuntimeError("A synchronized max-logit PDS group was aborted; both members must restart.")
        if output_a.token_ids != output_b.token_ids:
            raise RuntimeError(
                "mix-sglang max-logit PDS returned divergent trajectories: "
                f"{output_a.token_ids=} != {output_b.token_ids=}."
            )
        required = ("fused_topk_ids", "fused_topk_logits")
        missing = [key for key in required if output_a.extra_fields.get(key) is None]
        if missing:
            raise RuntimeError(f"mix-sglang did not return max-fused top-k logit fields: {missing}.")
        response_length = len(output_a.token_ids)
        for key in required:
            rows = output_a.extra_fields[key]
            if len(rows) != response_length:
                raise RuntimeError(f"mix-sglang field {key!r} has {len(rows)} rows, expected {response_length}.")
            for position, row in enumerate(rows):
                if len(row) != self.target_topk:
                    raise RuntimeError(
                        f"mix-sglang {key}[{position}] has width {len(row)}, expected {self.target_topk}."
                    )
        length = min(response_length, prompt_pair.max_new_tokens)
        return MaxLogitTurnOutput(
            prompt_pair=prompt_pair,
            token_ids=list(output_a.token_ids[:length]),
            fused_topk_ids=list(output_a.extra_fields["fused_topk_ids"][:length]),
            fused_topk_logits=list(output_a.extra_fields["fused_topk_logits"][:length]),
            stop_reason=output_a.stop_reason,
            num_preempted=output_a.num_preempted,
            extra_fields={
                key: value
                for key, value in output_a.extra_fields.items()
                if key not in {"fused_topk_ids", "fused_topk_logits"}
            },
        )

    def _align_teacher_targets(
        self,
        *,
        prompt_ids: Sequence[int],
        response_ids: Sequence[int],
        turn_output: MaxLogitTurnOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(response_ids) > len(turn_output.token_ids):
            raise ValueError("Response ids extend beyond the max-logit target trajectory.")
        sequence_length = len(prompt_ids) + len(response_ids)
        shape = (sequence_length, self.target_topk)
        teacher_ids = torch.full(
            shape,
            self.tokenizer.pad_token_id or 0,
            dtype=torch.int32,
        )
        teacher_logits = torch.zeros(shape, dtype=torch.float32)
        target_positions = [len(prompt_ids) - 1 + position for position in range(len(response_ids))]
        teacher_ids[target_positions] = torch.tensor(
            turn_output.fused_topk_ids[: len(response_ids)],
            dtype=torch.int32,
        )
        teacher_logits[target_positions] = torch.tensor(
            turn_output.fused_topk_logits[: len(response_ids)],
            dtype=torch.float32,
        )
        return teacher_ids, teacher_logits

    async def run(
        self,
        sampling_params: dict[str, Any],
        priority: int = 0,
        __validate__: bool = False,
        **kwargs,
    ) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        ground_truth = self._ground_truth_from_sample(kwargs)
        reference_solution = self._reference_solution_from_sample(kwargs, required=not __validate__)
        multi_modal_data = await self.process_multi_modal_info(messages)
        if multi_modal_data:
            raise NotImplementedError("MathMaxLogitDistillationAgentLoop supports text-only prompts.")

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

        response_limit = int(self.rollout_config.response_length)
        response_truncated = self._response_was_truncated(turn_output, response_limit=response_limit)
        response_ids = turn_output.token_ids[:response_limit]
        if not response_ids:
            raise RuntimeError("Math max-logit rollout produced no model tokens to verify or distill.")
        response_mask = [1] * len(response_ids)
        teacher_ids, teacher_logits = self._align_teacher_targets(
            prompt_ids=prompt_a_ids,
            response_ids=response_ids,
            turn_output=turn_output,
        )

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

        extra_fields = dict(turn_output.extra_fields)
        extra_fields.update(
            {
                "teacher_ids": teacher_ids,
                "teacher_logits": teacher_logits,
                "distillation_prompt_variant": "A",
                "distillation_target_kind": "max_fused_raw_logits",
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
        )
        return AgentLoopOutput(
            prompt_ids=prompt_a_ids,
            response_ids=response_ids,
            response_mask=training_response_mask,
            multi_modal_data={},
            num_turns=2,
            metrics=AgentLoopMetrics(
                generate_sequences=metrics["generate_sequences"],
                tool_calls=0.0,
                compute_score=metrics["math_verify"],
                num_preempted=turn_output.num_preempted if turn_output.num_preempted is not None else -1,
            ),
            extra_fields=extra_fields,
        )


__all__ = ["MathMaxLogitDistillationAgentLoop", "MaxLogitTurnOutput"]
