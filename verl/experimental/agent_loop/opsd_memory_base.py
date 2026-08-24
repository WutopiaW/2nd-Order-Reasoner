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
"""Reusable memory-augmented mix-sglang primitives for OPSD agent loops."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import ray
import torch
from omegaconf import DictConfig, OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopBase
from verl.experimental.agent_loop.trajectory_memory import TrajectoryMemoryActor
from verl.workers.rollout.replica import TokenOutput

DEFAULT_PROMPT_B_TEMPLATE = """A previous attempt on a semantically related problem is provided below.

Previous problem:
{memory_prompt}

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

PDS_OUTPUT_FIELDS = (
    "source_topk_ids",
    "source_topk_logprobs",
    "fused_topk_ids",
    "fused_topk_logprobs",
    "source_log_probs",
    "fused_log_probs",
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*\Z", flags=re.IGNORECASE | re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)


def extract_formal_response(text: str) -> str:
    """Return only text outside Qwen-style ``<think>`` blocks."""
    text = _THINK_BLOCK_RE.sub("", str(text))
    text = _UNCLOSED_THINK_RE.sub("", text)
    return _THINK_TAG_RE.sub("", text).strip()


@dataclass(frozen=True)
class OPSDMemoryContext:
    """One trajectory's stable memory identity and retrieved experience."""

    request_id: str
    prompt_a_text: str
    retrieved_request_id: str | None = None
    retrieval_score: float | None = None
    memory_prompt: str = ""
    memory_summary: str = ""
    memory_trajectory: str = ""

    @property
    def has_memory(self) -> bool:
        return self.retrieved_request_id is not None


@dataclass(frozen=True)
class OPSDPromptPair:
    """The two contexts and one shared generation budget sent to mix-sglang."""

    prompt_a_ids: list[int]
    prompt_b_ids: list[int]
    max_new_tokens: int
    # Exact post-budget user content used to render prompt B. ``None`` means B
    # reused prompt A unchanged instead of rendering a distinct context.
    prompt_b_text: str | None = None


@dataclass(frozen=True)
class OPSDTurnOutput:
    """One paired model turn after validating the mix-sglang PDS contract."""

    prompt_pair: OPSDPromptPair
    token_ids: list[int]
    source_logprobs: list[float]
    fused_logprobs: list[float]
    source_topk_ids: list[list[int]]
    source_topk_logprobs: list[list[float]]
    fused_topk_ids: list[list[int]]
    fused_topk_logprobs: list[list[float]]
    stop_reason: str | None
    num_preempted: int | None
    extra_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OPSDTrajectoryTargets:
    """Unpadded full-sequence targets plus response-aligned behavior logprobs."""

    response_logprobs: list[float]
    source_topk_ids: torch.Tensor
    source_topk_logprobs: torch.Tensor
    fused_topk_ids: torch.Tensor
    fused_topk_logprobs: torch.Tensor


@dataclass(frozen=True)
class _RecordedModelTurn:
    output: OPSDTurnOutput
    response_positions: tuple[int, ...]


class OPSDTrajectoryTargetAccumulator:
    """Accumulate PDS targets across model/tool/environment turns.

    A multi-turn loop records only model-generated turns. Tool and environment
    responses still occupy positions in ``response_ids``, but their
    ``response_mask`` is zero and their target rows remain dummy. Model-generated
    tool-call tokens remain part of a recorded model turn with mask one. The normal
    AgentLoop postprocessor performs final left/right padding after this accumulator
    returns unpadded tensors.
    """

    def __init__(self, *, prompt_ids: Sequence[int], target_topk: int, pad_token_id: int):
        self.prompt_ids = list(prompt_ids)
        self.target_topk = int(target_topk)
        self.pad_token_id = int(pad_token_id)
        self._turns: list[_RecordedModelTurn] = []
        self.extra_fields: dict[str, Any] = {}
        self.num_preempted = -1  # -1 means the backend did not report this metric.

    @property
    def model_turns(self) -> int:
        return len(self._turns)

    def record_model_turn(
        self,
        output: OPSDTurnOutput,
        *,
        response_start: int | None = None,
        response_positions: Sequence[int] | None = None,
    ) -> None:
        """Record where one model turn lands in the final flattened response."""
        if (response_start is None) == (response_positions is None):
            raise ValueError("Provide exactly one of response_start or response_positions.")
        if response_positions is None:
            response_positions = range(int(response_start), int(response_start) + len(output.token_ids))
        positions = tuple(int(position) for position in response_positions)
        if len(positions) != len(output.token_ids):
            raise ValueError(
                "The number of response positions must match the generated token count: "
                f"{len(positions)} != {len(output.token_ids)}."
            )
        if len(set(positions)) != len(positions) or any(position < 0 for position in positions):
            raise ValueError(f"Response positions must be unique and non-negative, got {positions}.")
        self._turns.append(_RecordedModelTurn(output=output, response_positions=positions))
        self._merge_turn_metadata(output)

    def _merge_turn_metadata(self, output: OPSDTurnOutput) -> None:
        if output.num_preempted is not None:
            if self.num_preempted < 0:
                self.num_preempted = output.num_preempted
            else:
                self.num_preempted += output.num_preempted
        if not self.extra_fields:
            self.extra_fields.update(output.extra_fields)
            return
        max_global_steps = output.extra_fields.get("max_global_steps")
        if max_global_steps is not None:
            self.extra_fields["max_global_steps"] = max_global_steps

    def finalize(
        self,
        *,
        response_ids: Sequence[int],
        response_mask: Sequence[int],
        require_all_model_tokens: bool = True,
    ) -> OPSDTrajectoryTargets:
        """Align every recorded model distribution with the flattened trajectory."""
        response_ids = list(response_ids)
        response_mask = list(response_mask)
        if len(response_ids) != len(response_mask):
            raise ValueError(
                f"response_ids/response_mask length mismatch: {len(response_ids)} != {len(response_mask)}."
            )
        if any(mask not in (0, 1) for mask in response_mask):
            raise ValueError("response_mask values must be 0 for environment tokens or 1 for model tokens.")
        sequence_length = len(self.prompt_ids) + len(response_ids)
        shape = (sequence_length, self.target_topk)
        source_ids = torch.full(shape, self.pad_token_id, dtype=torch.int32)
        source_logprobs = torch.zeros(shape, dtype=torch.float32)
        fused_ids = torch.full(shape, self.pad_token_id, dtype=torch.int32)
        fused_logprobs = torch.zeros(shape, dtype=torch.float32)
        response_logprobs = [0.0] * len(response_ids)
        covered_positions: set[int] = set()

        for recorded in self._turns:
            output = recorded.output
            rows = (
                output.source_topk_ids,
                output.source_topk_logprobs,
                output.fused_topk_ids,
                output.fused_topk_logprobs,
            )
            if any(len(row) != len(output.token_ids) for row in rows):
                raise ValueError("A recorded OPSD turn has target rows misaligned with its token IDs.")
            if any(len(values) != self.target_topk for row in rows for values in row):
                raise ValueError(
                    f"A recorded OPSD turn has a target row whose width is not {self.target_topk}."
                )
            retained_turn_indices: list[int] = []
            target_positions: list[int] = []
            for turn_index, response_position in enumerate(recorded.response_positions):
                if response_position >= len(response_ids):
                    # The normal response-length cap may truncate a final model turn.
                    continue
                if response_position in covered_positions:
                    raise ValueError(f"Multiple OPSD turns target response position {response_position}.")
                if response_mask[response_position] != 1:
                    raise ValueError(
                        f"OPSD model token at response position {response_position} has response_mask="
                        f"{response_mask[response_position]}, expected 1."
                    )
                expected_token = output.token_ids[turn_index]
                if response_ids[response_position] != expected_token:
                    raise ValueError(
                        "Generated-token alignment changed before OPSD target assembly: "
                        f"response_ids[{response_position}]={response_ids[response_position]} != "
                        f"turn token {expected_token}. Continuous-token agent loops must provide "
                        "an exact post-merge position mapping."
                    )
                retained_turn_indices.append(turn_index)
                target_positions.append(len(self.prompt_ids) - 1 + response_position)
                response_logprobs[response_position] = output.fused_logprobs[turn_index]
                covered_positions.add(response_position)
            if target_positions:
                source_ids[target_positions] = torch.tensor(
                    [output.source_topk_ids[index] for index in retained_turn_indices],
                    dtype=torch.int32,
                )
                source_logprobs[target_positions] = torch.tensor(
                    [output.source_topk_logprobs[index] for index in retained_turn_indices],
                    dtype=torch.float32,
                )
                fused_ids[target_positions] = torch.tensor(
                    [output.fused_topk_ids[index] for index in retained_turn_indices],
                    dtype=torch.int32,
                )
                fused_logprobs[target_positions] = torch.tensor(
                    [output.fused_topk_logprobs[index] for index in retained_turn_indices],
                    dtype=torch.float32,
                )

        model_positions = {position for position, mask in enumerate(response_mask) if mask == 1}
        if require_all_model_tokens and covered_positions != model_positions:
            missing = sorted(model_positions - covered_positions)
            unexpected = sorted(covered_positions - model_positions)
            raise ValueError(
                "OPSD targets do not cover exactly the model-generated response positions: "
                f"{missing=}, {unexpected=}."
            )
        return OPSDTrajectoryTargets(
            response_logprobs=response_logprobs,
            source_topk_ids=source_ids,
            source_topk_logprobs=source_logprobs,
            fused_topk_ids=fused_ids,
            fused_topk_logprobs=fused_logprobs,
        )


class OPSDMemoryAgentLoopBase(AgentLoopBase):
    """Abstract OPSD memory base; concrete agent loops own their ``run`` state machine."""

    def __init__(
        self,
        *args,
        target_topk: int = 32,
        memory_prompt_max_length: int = 512,
        prompt_b_template: str = DEFAULT_PROMPT_B_TEMPLATE,
        summary_template: str = DEFAULT_SUMMARY_TEMPLATE,
        summary_max_tokens: int = 256,
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
        if memory_prompt_max_length is None:
            raise ValueError("OPSDMemoryAgentLoopBase requires memory_prompt_max_length.")
        self.target_topk = int(target_topk)
        self.memory_prompt_max_length = int(memory_prompt_max_length)
        self.summary_max_tokens = int(summary_max_tokens)
        memory_capacity = int(memory_capacity)
        if self.memory_prompt_max_length < int(self.rollout_config.prompt_length):
            raise ValueError(
                "memory_prompt_max_length must be at least rollout.prompt_length so prompt B can "
                "always represent an unmodified prompt A: "
                f"{self.memory_prompt_max_length} < {int(self.rollout_config.prompt_length)}."
            )
        if self.summary_max_tokens <= 0:
            raise ValueError(f"summary_max_tokens must be positive, got {self.summary_max_tokens}.")
        self.prompt_b_template = prompt_b_template
        self.summary_template = summary_template
        self.prompt_a_fuse_weight = float(prompt_a_fuse_weight)
        self.prompt_b_fuse_weight = float(prompt_b_fuse_weight)
        if isinstance(memory_embedder, DictConfig):
            memory_embedder = OmegaConf.to_container(memory_embedder, resolve=True)

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

    async def initialize_prompt_a(self, messages: list[dict[str, Any]], **template_kwargs: Any) -> list[int]:
        """Render initial prompt A once, then fail instead of silently truncating it."""
        prompt_a_ids = await self.apply_chat_template(
            messages,
            cap_prompt_length=False,
            **template_kwargs,
        )
        prompt_limit = int(self.rollout_config.prompt_length)
        if len(prompt_a_ids) > prompt_limit:
            raise ValueError(
                "Prompt A exceeds rollout.prompt_length after applying the chat template: "
                f"{len(prompt_a_ids)} > {prompt_limit}. Dataset filtering and the AgentLoop "
                "must use the same tokenizer, template, and tool schemas."
            )
        return prompt_a_ids

    async def initialize_memory_context(
        self,
        prompt_a_ids: Sequence[int],
        *,
        request_id: str | None = None,
        query_text: str | None = None,
    ) -> OPSDMemoryContext:
        """Create the trajectory identity and retrieve one related memory record."""
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
            memory_summary=extract_formal_response(record["summary"]),
            memory_trajectory=extract_formal_response(record["trajectory"]),
        )

    async def initialize_prompt_pair(
        self,
        *,
        prompt_a_ids: Sequence[int],
        memory_context: OPSDMemoryContext,
        max_new_tokens: int,
        prompt_b_extra_fields: Mapping[str, str] | None = None,
        force_prompt_b: bool = False,
    ) -> OPSDPromptPair:
        """Build A/B using the caller-specified shared generation length."""
        prompt_a_ids = list(prompt_a_ids)
        prompt_b_text = None
        if not memory_context.has_memory and not force_prompt_b:
            prompt_b_ids = list(prompt_a_ids)
        else:
            current_prompt_a = self.tokenizer.decode(prompt_a_ids, skip_special_tokens=True)
            fields = {
                "prompt_a": current_prompt_a,
                "memory_prompt": memory_context.memory_prompt,
                "memory_summary": memory_context.memory_summary,
                "memory_trajectory": memory_context.memory_trajectory,
            }
            prompt_b_extra_fields = dict(prompt_b_extra_fields or {})
            duplicate_fields = fields.keys() & prompt_b_extra_fields.keys()
            if duplicate_fields:
                raise ValueError(f"Prompt B extra fields must not replace built-in fields: {sorted(duplicate_fields)}.")
            fields.update(prompt_b_extra_fields)
            prompt_b_text, prompt_b_ids = await self._render_prompt_with_budget(
                template=self.prompt_b_template,
                fields=fields,
                trim_order=("memory_trajectory", "memory_summary", "memory_prompt", *prompt_b_extra_fields),
                trim_sides={"memory_prompt": "right"},
                max_prompt_tokens=self.memory_prompt_max_length,
            )
        return OPSDPromptPair(
            prompt_a_ids=prompt_a_ids,
            prompt_b_ids=prompt_b_ids,
            max_new_tokens=int(max_new_tokens),
            prompt_b_text=prompt_b_text,
        )

    async def generate_paired(
        self,
        *,
        prompt_a_ids: Sequence[int],
        memory_context: OPSDMemoryContext,
        sampling_params: dict[str, Any],
        priority: int = 0,
        prompt_b_extra_fields: Mapping[str, str] | None = None,
        force_prompt_b: bool = False,
    ) -> OPSDTurnOutput:
        """Construct and submit one native two-member PDS request group."""
        max_new_tokens = sampling_params.get(
            "max_new_tokens",
            sampling_params.get("max_tokens", self.rollout_config.response_length),
        )
        pair = await self.initialize_prompt_pair(
            prompt_a_ids=prompt_a_ids,
            memory_context=memory_context,
            max_new_tokens=max_new_tokens,
            prompt_b_extra_fields=prompt_b_extra_fields,
            force_prompt_b=force_prompt_b,
        )
        sample_group = f"opsd-{memory_context.request_id}-{uuid4().hex}"
        base_params = dict(sampling_params)
        base_params.pop("max_tokens", None)
        base_params["max_new_tokens"] = pair.max_new_tokens
        base_params["logprobs"] = False

        def paired_params(weight: float, *, return_probabilities: bool) -> dict[str, Any]:
            params = dict(base_params)
            custom_params = dict(params.get("custom_params") or {})
            custom_params.update(
                {
                    "__pds_sample_group": sample_group,
                    "__pds_fuse_method": "avg_probs",
                    "__pds_fuse_weight": weight,
                }
            )
            if return_probabilities:
                custom_params.update(
                    {
                        "__pds_return_prob_trajectory": True,
                        "__pds_return_top_k": self.target_topk,
                    }
                )
            else:
                custom_params.pop("__pds_return_prob_trajectory", None)
                custom_params.pop("__pds_return_top_k", None)
            params["custom_params"] = custom_params
            return params

        outputs = await self.server_manager.generate_group(
            request_id=memory_context.request_id,
            requests=[
                {
                    "prompt_ids": pair.prompt_a_ids,
                    "sampling_params": paired_params(self.prompt_a_fuse_weight, return_probabilities=True),
                    "priority": int(priority),
                },
                {
                    "prompt_ids": pair.prompt_b_ids,
                    "sampling_params": paired_params(self.prompt_b_fuse_weight, return_probabilities=False),
                    "priority": int(priority),
                },
            ],
        )
        return self.process_mix_outputs(outputs, prompt_pair=pair)

    def process_mix_outputs(
        self,
        outputs: Sequence[TokenOutput],
        *,
        prompt_pair: OPSDPromptPair,
    ) -> OPSDTurnOutput:
        """Validate one PDS pair and retain only prompt A's target payload."""
        if len(outputs) != 2:
            raise RuntimeError(f"mix-sglang returned {len(outputs)} outputs for a two-member PDS group.")
        output_a, output_b = outputs
        if output_a.stop_reason == "aborted" or output_b.stop_reason == "aborted":
            raise RuntimeError(
                "A synchronized mix-sglang request group was aborted. Both PDS members must restart together."
            )
        if output_a.token_ids != output_b.token_ids:
            raise RuntimeError(
                "mix-sglang PDS returned divergent trajectories for one sample group: "
                f"{output_a.token_ids=} != {output_b.token_ids=}."
            )
        self._validate_mix_output(output_a)
        length = min(len(output_a.token_ids), prompt_pair.max_new_tokens)
        extra_fields = dict(output_a.extra_fields)
        turn_output = OPSDTurnOutput(
            prompt_pair=prompt_pair,
            token_ids=list(output_a.token_ids[:length]),
            source_logprobs=list(output_a.extra_fields["source_log_probs"][:length]),
            fused_logprobs=list(output_a.extra_fields["fused_log_probs"][:length]),
            source_topk_ids=list(output_a.extra_fields["source_topk_ids"][:length]),
            source_topk_logprobs=list(output_a.extra_fields["source_topk_logprobs"][:length]),
            fused_topk_ids=list(output_a.extra_fields["fused_topk_ids"][:length]),
            fused_topk_logprobs=list(output_a.extra_fields["fused_topk_logprobs"][:length]),
            stop_reason=output_a.stop_reason,
            num_preempted=output_a.num_preempted,
            extra_fields={key: value for key, value in extra_fields.items() if key not in PDS_OUTPUT_FIELDS},
        )
        self._validate_turn_topk_width(turn_output)
        return turn_output

    def new_target_accumulator(self, prompt_ids: Sequence[int]) -> OPSDTrajectoryTargetAccumulator:
        return OPSDTrajectoryTargetAccumulator(
            prompt_ids=prompt_ids,
            target_topk=self.target_topk,
            pad_token_id=self.tokenizer.pad_token_id or 0,
        )

    async def finalize_memory(
        self,
        *,
        memory_context: OPSDMemoryContext,
        trajectory: str,
        priority: int = 0,
        validate: bool = False,
    ) -> str:
        """Summarize the completed full trajectory and write it to global memory."""
        generated_summary = await self._summarize(
            request_id=memory_context.request_id,
            prompt_a=memory_context.prompt_a_text,
            trajectory=trajectory,
            priority=int(priority),
        )
        memory_trajectory = extract_formal_response(trajectory)
        memory_summary = extract_formal_response(generated_summary)
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
                metadata={
                    "retrieved_request_id": memory_context.retrieved_request_id,
                    "retrieval_score": memory_context.retrieval_score,
                    "retrieved_memory": retrieved_memory,
                },
            )
        return memory_summary

    def _validate_mix_output(self, output: TokenOutput) -> None:
        missing = sorted(set(PDS_OUTPUT_FIELDS).difference(output.extra_fields))
        if missing:
            raise RuntimeError(
                "mix-sglang did not return the OPSD probability protocol fields "
                f"{missing}. The PDS backend must expose source and fused token/top-k logprobs."
            )
        response_length = len(output.token_ids)
        for key in PDS_OUTPUT_FIELDS:
            if len(output.extra_fields[key]) != response_length:
                raise RuntimeError(
                    f"mix-sglang field {key!r} has length {len(output.extra_fields[key])}, "
                    f"expected {response_length}."
                )
        if output.log_probs != output.extra_fields["fused_log_probs"]:
            raise RuntimeError("rollout log_probs must be the fused behavior-policy logprobs.")

    def _validate_turn_topk_width(self, output: OPSDTurnOutput) -> None:
        for field_name in (
            "source_topk_ids",
            "source_topk_logprobs",
            "fused_topk_ids",
            "fused_topk_logprobs",
        ):
            rows = getattr(output, field_name)
            for position, row in enumerate(rows):
                if len(row) != self.target_topk:
                    raise RuntimeError(
                        f"mix-sglang {field_name}[{position}] has width {len(row)}, "
                        f"expected target_topk={self.target_topk}."
                    )

    async def _summarize(
        self,
        *,
        request_id: str,
        prompt_a: str,
        trajectory: str,
        priority: int,
        summary_extra_fields: Mapping[str, str] | None = None,
    ) -> str:
        fields = {"prompt_a": prompt_a, "trajectory": trajectory}
        summary_extra_fields = dict(summary_extra_fields or {})
        duplicate_fields = fields.keys() & summary_extra_fields.keys()
        if duplicate_fields:
            raise ValueError(f"Summary extra fields must not replace built-in fields: {sorted(duplicate_fields)}.")
        fields.update(summary_extra_fields)
        _, summary_prompt_ids = await self._render_prompt_with_budget(
            template=self.summary_template,
            fields=fields,
            trim_order=("prompt_a", "trajectory", *summary_extra_fields),
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
        if self.rollout_config.max_model_len is not None:
            return int(self.rollout_config.max_model_len)
        return int(self.rollout_config.prompt_length + self.rollout_config.response_length)

    async def _render_prompt_with_budget(
        self,
        *,
        template: str,
        fields: dict[str, str],
        trim_order: tuple[str, ...],
        max_prompt_tokens: int,
        trim_sides: dict[str, str] | None = None,
    ) -> tuple[str, list[int]]:
        """Render and trim selected variable fields to a hard token cap."""
        if max_prompt_tokens <= 0:
            raise ValueError(f"Prompt token budget must be positive, got {max_prompt_tokens}.")
        fields = dict(fields)
        trim_sides = dict(trim_sides or {})
        invalid_trim_sides = {
            field_name: side for field_name, side in trim_sides.items() if side not in {"left", "right"}
        }
        if invalid_trim_sides:
            raise ValueError(f"Prompt trim sides must be 'left' or 'right', got {invalid_trim_sides}.")

        async def render() -> tuple[str, list[int]]:
            text = template.format(**fields)
            ids = await self.apply_chat_template(
                [{"role": "user", "content": text}],
                cap_prompt_length=False,
            )
            return text, ids

        text, prompt_ids = await render()
        for field_name in trim_order:
            while len(prompt_ids) > max_prompt_tokens and fields[field_name]:
                field_ids = self.tokenizer.encode(fields[field_name], add_special_tokens=False)
                if not field_ids:
                    fields[field_name] = ""
                    break
                overflow = len(prompt_ids) - max_prompt_tokens
                remove_count = min(len(field_ids), overflow + 8)
                if trim_sides.get(field_name, "left") == "right":
                    retained_ids = field_ids[: len(field_ids) - remove_count]
                else:
                    retained_ids = field_ids[remove_count:]
                fields[field_name] = self.tokenizer.decode(
                    retained_ids,
                    skip_special_tokens=True,
                )
                text, prompt_ids = await render()

        if len(prompt_ids) > max_prompt_tokens:
            raise ValueError(
                "The fixed OPSD prompt template and untrimmable fields exceed the prompt budget: "
                f"{len(prompt_ids)} > {max_prompt_tokens} tokens."
            )
        return text, prompt_ids
