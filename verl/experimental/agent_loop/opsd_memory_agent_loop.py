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
"""Memory-augmented mix-sglang AgentLoop for OPSD."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import ray
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.trajectory_memory import TrajectoryMemoryActor
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput


DEFAULT_PROMPT_B_TEMPLATE = """A previous attempt on a semantically related problem is provided below.

Previous summary:
{memory_summary}

Previous trajectory:
{memory_trajectory}

Current problem:
{prompt_a}

Use the previous experience only when it helps. Solve the current problem independently."""

DEFAULT_SUMMARY_TEMPLATE = """Summarize the reusable reasoning experience in the trajectory below.
Focus on the approach, useful intermediate insights, and mistakes to avoid. Be concise.

Problem:
{prompt_a}

Trajectory:
{trajectory}
"""


def align_response_topk_targets(
    *,
    prompt_length: int,
    topk_ids: list[list[int]],
    topk_logprobs: list[list[float]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align generation-step distributions with full-sequence model logits.

    The distribution that sampled response token ``t`` belongs at sequence
    position ``prompt_length - 1 + t`` because causal LM logits are shifted by
    one position relative to labels.
    """
    if prompt_length <= 0:
        raise ValueError(f"prompt_length must be positive, got {prompt_length}.")
    if len(topk_ids) != len(topk_logprobs):
        raise ValueError(
            f"top-k ids/logprobs length mismatch: {len(topk_ids)} != {len(topk_logprobs)}."
        )
    response_length = len(topk_ids)
    if response_length == 0:
        raise ValueError("Cannot build OPSD targets for an empty response.")
    topk = len(topk_ids[0])
    if topk == 0:
        raise ValueError("top-k target width must be positive.")
    for position, (ids, logprobs) in enumerate(zip(topk_ids, topk_logprobs, strict=True)):
        if len(ids) != topk or len(logprobs) != topk:
            raise ValueError(
                f"Inconsistent top-k width at response position {position}: "
                f"{len(ids)=}, {len(logprobs)=}, expected {topk}."
            )

    sequence_length = prompt_length + response_length
    target_ids = torch.full((sequence_length, topk), pad_token_id, dtype=torch.int32)
    target_logprobs = torch.zeros((sequence_length, topk), dtype=torch.float32)
    start = prompt_length - 1
    end = start + response_length
    target_ids[start:end] = torch.tensor(topk_ids, dtype=torch.int32)
    target_logprobs[start:end] = torch.tensor(topk_logprobs, dtype=torch.float32)
    return target_ids, target_logprobs


class OPSDMemoryAgentLoop(AgentLoopBase):
    """Retrieve memory, run paired PDS sampling, and emit rollout targets."""

    def __init__(
        self,
        *args,
        target_topk: int = 32,
        prompt_b_template: str = DEFAULT_PROMPT_B_TEMPLATE,
        summary_template: str = DEFAULT_SUMMARY_TEMPLATE,
        summary_max_tokens: int = 256,
        min_response_tokens: int = 32,
        memory_capacity: int = 100_000,
        memory_actor_name: str | None = None,
        memory_embedder: dict[str, Any] | None = None,
        memory_output_path: str | None = None,
        memory_seed_path: str | None = None,
        prompt_a_fuse_weight: float = 1.0,
        prompt_b_fuse_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        target_topk = int(target_topk)
        summary_max_tokens = int(summary_max_tokens)
        min_response_tokens = int(min_response_tokens)
        memory_capacity = int(memory_capacity)
        if target_topk <= 0:
            raise ValueError(f"target_topk must be positive, got {target_topk}.")
        if summary_max_tokens <= 0:
            raise ValueError(f"summary_max_tokens must be positive, got {summary_max_tokens}.")
        if min_response_tokens <= 0 or min_response_tokens > self.rollout_config.response_length:
            raise ValueError(
                "min_response_tokens must be in [1, rollout.response_length], "
                f"got {min_response_tokens} and {self.rollout_config.response_length}."
            )
        self.target_topk = target_topk
        self.prompt_b_template = prompt_b_template
        self.summary_template = summary_template
        self.summary_max_tokens = summary_max_tokens
        self.min_response_tokens = min_response_tokens
        self.prompt_a_fuse_weight = float(prompt_a_fuse_weight)
        self.prompt_b_fuse_weight = float(prompt_b_fuse_weight)
        if self.prompt_a_fuse_weight < 0 or self.prompt_b_fuse_weight < 0:
            raise ValueError("PDS fuse weights must be non-negative.")
        if self.prompt_a_fuse_weight + self.prompt_b_fuse_weight <= 0:
            raise ValueError("At least one PDS fuse weight must be positive.")

        distillation_config = self.config.get("distillation")
        if not distillation_config or not distillation_config.get("enabled", False):
            raise ValueError("OPSDMemoryAgentLoop requires distillation.enabled=true.")
        if distillation_config.get("target_source", "teacher") != "rollout":
            raise ValueError("OPSDMemoryAgentLoop requires distillation.target_source=rollout.")
        loss_config = distillation_config.get("distillation_loss", {})
        if loss_config.get("loss_mode") != "forward_kl_topk":
            raise ValueError("OPSDMemoryAgentLoop requires distillation loss_mode=forward_kl_topk.")
        if loss_config.get("use_policy_gradient", False):
            raise ValueError("OPSDMemoryAgentLoop requires use_policy_gradient=false for direct distribution matching.")
        configured_topk_value = loss_config.get("topk")
        if configured_topk_value is None:
            raise ValueError("OPSDMemoryAgentLoop requires distillation.distillation_loss.topk.")
        configured_topk = int(configured_topk_value)
        if configured_topk != target_topk:
            raise ValueError(
                "Agent-loop target_topk must match distillation.distillation_loss.topk, "
                f"got {target_topk} and {configured_topk}."
            )

        if memory_actor_name is None:
            job_id = ray.get_runtime_context().get_job_id()
            memory_actor_name = f"opsd_trajectory_memory_{job_id}"
        self.memory = TrajectoryMemoryActor.options(
            name=memory_actor_name,
            get_if_exists=True,
            lifetime="detached",
        ).remote(
            capacity=memory_capacity,
            embedder=memory_embedder,
            output_path=memory_output_path,
            seed_path=memory_seed_path,
        )

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
            raise NotImplementedError("OPSDMemoryAgentLoop currently supports text-only prompts.")

        prompt_a_ids = await self.apply_chat_template(messages)
        prompt_a_text = self.tokenizer.decode(prompt_a_ids, skip_special_tokens=True)
        request_id = uuid4().hex

        retrieved = await self.memory.search.remote(prompt_a_text, exclude_request_id=request_id)
        if retrieved is None:
            memory_request_id = None
            memory_score = None
            memory_summary = ""
            memory_trajectory = ""
        else:
            memory_record = retrieved["record"]
            memory_request_id = memory_record["request_id"]
            memory_score = retrieved["score"]
            memory_summary = memory_record["summary"]
            memory_trajectory = memory_record["trajectory"]

        if retrieved is None:
            # Bootstrap an empty memory without introducing an arbitrary,
            # memory-free instruction distribution.
            prompt_b_ids = list(prompt_a_ids)
        else:
            _, prompt_b_ids = await self._render_prompt_with_budget(
                template=self.prompt_b_template,
                fields={
                    "prompt_a": prompt_a_text,
                    "memory_summary": memory_summary,
                    "memory_trajectory": memory_trajectory,
                },
                trim_order=("memory_trajectory", "memory_summary"),
                max_prompt_tokens=self._context_limit() - self.min_response_tokens - 1,
            )

        generation_limit = min(
            self.rollout_config.response_length,
            self.rollout_config.prompt_length + self.rollout_config.response_length - max(
                len(prompt_a_ids), len(prompt_b_ids)
            ),
        )
        if self.rollout_config.max_model_len is not None:
            generation_limit = min(
                generation_limit,
                self.rollout_config.max_model_len - max(len(prompt_a_ids), len(prompt_b_ids)) - 1,
            )
        if generation_limit <= 0:
            raise ValueError(
                "No shared generation budget remains for the OPSD prompt pair: "
                f"{len(prompt_a_ids)=}, {len(prompt_b_ids)=}."
            )

        sample_group = f"opsd-{request_id}"
        base_params = dict(sampling_params)
        base_params["max_new_tokens"] = generation_limit
        # Dedicated PDS probabilities describe the deferred shared sample;
        # ordinary sampler logprobs are both redundant and potentially stale.
        base_params["logprobs"] = False

        def paired_params(weight: float) -> dict[str, Any]:
            params = dict(base_params)
            custom_params = dict(params.get("custom_params") or {})
            custom_params.update(
                {
                    "__pds_sample_group": sample_group,
                    "__pds_fuse_method": "avg_probs",
                    "__pds_fuse_weight": weight,
                    "__pds_return_prob_trajectory": True,
                    "__pds_return_top_k": self.target_topk,
                }
            )
            params["custom_params"] = custom_params
            return params

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            outputs = await self.server_manager.generate_group(
                request_id=request_id,
                requests=[
                    {
                        "prompt_ids": prompt_a_ids,
                        "sampling_params": paired_params(self.prompt_a_fuse_weight),
                        "priority": int(priority),
                    },
                    {
                        "prompt_ids": prompt_b_ids,
                        "sampling_params": paired_params(self.prompt_b_fuse_weight),
                        "priority": int(priority),
                    },
                ],
            )
        output_a, output_b = outputs
        if output_a.stop_reason == "aborted" or output_b.stop_reason == "aborted":
            raise RuntimeError(
                "A synchronized mix-sglang request group was aborted. Partial-rollout resume "
                "is not supported because both PDS members must restart together."
            )
        if output_a.token_ids != output_b.token_ids:
            raise RuntimeError(
                "mix-sglang PDS returned divergent trajectories for one sample group: "
                f"{output_a.token_ids=} != {output_b.token_ids=}."
            )
        self._validate_mix_output(output_a)
        self._validate_mix_output(output_b)
        for key in ("fused_log_probs", "fused_topk_ids", "fused_topk_logprobs"):
            if output_a.extra_fields[key] != output_b.extra_fields[key]:
                raise RuntimeError(f"mix-sglang returned inconsistent {key!r} across one PDS sample group.")

        response_ids = output_a.token_ids[: self.rollout_config.response_length]
        source_topk_ids = output_a.extra_fields["source_topk_ids"][: len(response_ids)]
        source_topk_logprobs = output_a.extra_fields["source_topk_logprobs"][: len(response_ids)]
        fused_topk_ids = output_a.extra_fields["fused_topk_ids"][: len(response_ids)]
        fused_topk_logprobs = output_a.extra_fields["fused_topk_logprobs"][: len(response_ids)]
        source_ids, source_logprobs = align_response_topk_targets(
            prompt_length=len(prompt_a_ids),
            topk_ids=source_topk_ids,
            topk_logprobs=source_topk_logprobs,
            pad_token_id=self.tokenizer.pad_token_id or 0,
        )
        teacher_ids, teacher_logprobs = align_response_topk_targets(
            prompt_length=len(prompt_a_ids),
            topk_ids=fused_topk_ids,
            topk_logprobs=fused_topk_logprobs,
            pad_token_id=self.tokenizer.pad_token_id or 0,
        )

        trajectory = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        summary = await self._summarize(
            request_id=request_id,
            prompt_a=prompt_a_text,
            trajectory=trajectory,
            priority=int(priority),
        )
        if not __validate__:
            await self.memory.upsert.remote(
                request_id=request_id,
                prompt=prompt_a_text,
                trajectory=trajectory,
                summary=summary,
                metadata={
                    "retrieved_request_id": memory_request_id,
                    "retrieval_score": memory_score,
                },
            )

        response_logprobs = output_a.log_probs[: len(response_ids)] if output_a.log_probs is not None else None
        extra_fields = dict(output_a.extra_fields)
        for key in (
            "source_topk_ids",
            "source_topk_logprobs",
            "fused_topk_ids",
            "fused_topk_logprobs",
        ):
            extra_fields.pop(key, None)
        extra_fields.update(
            {
                "teacher_ids": teacher_ids,
                "teacher_logprobs": teacher_logprobs,
                "memory_request_id": request_id,
                "retrieved_memory_request_id": memory_request_id,
                "memory_retrieval_score": memory_score,
                "trajectory_summary": summary,
                "turn_scores": [],
                "tool_rewards": [],
            }
        )
        return AgentLoopOutput(
            prompt_ids=prompt_a_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=response_logprobs,
            source_topk_ids=source_ids,
            source_topk_logprobs=source_logprobs,
            fused_topk_ids=teacher_ids,
            fused_topk_logprobs=teacher_logprobs,
            multi_modal_data={},
            num_turns=2,
            metrics=AgentLoopMetrics(
                generate_sequences=metrics["generate_sequences"],
                tool_calls=0.0,
                compute_score=0.0,
                num_preempted=output_a.num_preempted if output_a.num_preempted is not None else -1,
            ),
            extra_fields=extra_fields,
        )

    def _validate_mix_output(self, output: TokenOutput) -> None:
        required = {
            "source_topk_ids",
            "source_topk_logprobs",
            "fused_topk_ids",
            "fused_topk_logprobs",
            "source_log_probs",
            "fused_log_probs",
        }
        missing = sorted(required.difference(output.extra_fields))
        if missing:
            raise RuntimeError(
                "mix-sglang did not return the OPSD probability protocol fields "
                f"{missing}. The PDS backend must expose source and fused token/top-k logprobs."
            )
        response_length = len(output.token_ids)
        for key in required:
            if len(output.extra_fields[key]) != response_length:
                raise RuntimeError(
                    f"mix-sglang field {key!r} has length {len(output.extra_fields[key])}, "
                    f"expected {response_length}."
                )
        if output.log_probs != output.extra_fields["fused_log_probs"]:
            raise RuntimeError("rollout log_probs must be the fused behavior-policy logprobs.")

    async def _summarize(
        self,
        *,
        request_id: str,
        prompt_a: str,
        trajectory: str,
        priority: int,
    ) -> str:
        _, summary_prompt_ids = await self._render_prompt_with_budget(
            template=self.summary_template,
            fields={"prompt_a": prompt_a, "trajectory": trajectory},
            trim_order=("prompt_a", "trajectory"),
            max_prompt_tokens=self._context_limit() - 2,
        )
        max_new_tokens = min(
            self.summary_max_tokens,
            self._context_limit() - len(summary_prompt_ids) - 1,
        )
        summary_output = await self.server_manager.generate(
            request_id=f"{request_id}-summary",
            prompt_ids=summary_prompt_ids,
            sampling_params={
                "temperature": 0,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": max_new_tokens,
                "logprobs": False,
            },
            priority=priority,
        )
        return self.tokenizer.decode(summary_output.token_ids, skip_special_tokens=True).strip()

    def _context_limit(self) -> int:
        context_limit = self.rollout_config.prompt_length + self.rollout_config.response_length
        if self.rollout_config.max_model_len is not None:
            context_limit = min(context_limit, self.rollout_config.max_model_len)
        return context_limit

    async def _render_prompt_with_budget(
        self,
        *,
        template: str,
        fields: dict[str, str],
        trim_order: tuple[str, ...],
        max_prompt_tokens: int,
    ) -> tuple[str, list[int]]:
        """Render a chat prompt while preserving a generation-token reserve."""
        if max_prompt_tokens <= 0:
            raise ValueError(f"Prompt token budget must be positive, got {max_prompt_tokens}.")
        fields = dict(fields)

        async def render() -> tuple[str, list[int]]:
            text = template.format(**fields)
            ids = await self.apply_chat_template([{"role": "user", "content": text}])
            return text, ids

        text, prompt_ids = await render()
        for field_name in trim_order:
            while len(prompt_ids) > max_prompt_tokens and fields[field_name]:
                field_ids = self.tokenizer.encode(fields[field_name], add_special_tokens=False)
                if not field_ids:
                    fields[field_name] = ""
                    break
                overflow = len(prompt_ids) - max_prompt_tokens
                # A small margin accounts for tokenizer boundary changes after
                # inserting the shortened text back into the template.
                remove_count = min(len(field_ids), overflow + 8)
                fields[field_name] = self.tokenizer.decode(
                    field_ids[remove_count:],
                    skip_special_tokens=True,
                )
                text, prompt_ids = await render()

        if len(prompt_ids) > max_prompt_tokens:
            raise ValueError(
                "The fixed OPSD prompt template and untrimmable fields exceed the available context: "
                f"{len(prompt_ids)} > {max_prompt_tokens} tokens."
            )
        return text, prompt_ids
