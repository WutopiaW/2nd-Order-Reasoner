#!/usr/bin/env python3
"""Dependency-free contract smoke for the memory-augmented OPSD flow.

This does not run a model or SGLang. It exercises the orchestration contract
with deterministic fakes while reusing verl's real PDS response parser.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_probability_parser():
    module_path = Path(__file__).parents[2] / "verl/workers/rollout/logprob_protocol.py"
    spec = importlib.util.spec_from_file_location("opsd_smoke_logprob_protocol", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load PDS protocol module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_pds_probability_fields


extract_pds_probability_fields = _load_probability_parser()


@dataclass
class MemoryRecord:
    request_id: str
    prompt: str
    trajectory: str
    summary: str


class FakeGlobalMemory:
    """Small request-id-keyed store with deterministic lexical retrieval."""

    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}

    def search(self, prompt: str) -> MemoryRecord | None:
        query_tokens = set(prompt.casefold().split())
        if not self.records:
            return None
        return max(
            self.records.values(),
            key=lambda record: len(query_tokens & set(record.prompt.casefold().split())),
        )

    def upsert(self, record: MemoryRecord) -> None:
        self.records[record.request_id] = record


class FakeMixSGLang:
    """Deterministic stand-in for one native mix-sglang batch call."""

    def __init__(self) -> None:
        self.batch_calls = 0

    def generate_group(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(requests) != 2:
            raise AssertionError("PDS smoke expects prompt A and prompt B in one two-member batch.")
        custom_params = [request["sampling_params"]["custom_params"] for request in requests]
        group_ids = {params["__pds_sample_group"] for params in custom_params}
        if len(group_ids) != 1:
            raise AssertionError("Both PDS members must use the same sample group.")
        for params in custom_params:
            assert params["__pds_fuse_method"] == "avg_probs"
        assert custom_params[0]["__pds_return_prob_trajectory"] is True
        assert custom_params[0]["__pds_return_top_k"] == 2
        assert "__pds_return_prob_trajectory" not in custom_params[1]
        assert "__pds_return_top_k" not in custom_params[1]

        self.batch_calls += 1
        output_ids = [101, 102]
        source_distributions = [
            [
                {101: 0.7, 201: 0.2, 301: 0.1},
                {102: 0.6, 202: 0.3, 302: 0.1},
            ],
            [
                {101: 0.4, 201: 0.1, 401: 0.5},
                {102: 0.3, 202: 0.2, 402: 0.5},
            ],
        ]
        fused_distributions = [
            {101: 0.55, 201: 0.15, 301: 0.05, 401: 0.25},
            {102: 0.45, 202: 0.25, 302: 0.05, 402: 0.25},
        ]

        def top_k(distribution: dict[int, float]) -> list[dict[str, int | float]]:
            ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:2]
            return [{"token_id": token_id, "prob": probability} for token_id, probability in ranked]

        source_a = source_distributions[0]
        return [
            {
                "output_ids": list(output_ids),
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "pds_source_token_probs": [
                        distribution[token_id]
                        for distribution, token_id in zip(source_a, output_ids, strict=True)
                    ],
                    "pds_fused_token_probs": [
                        distribution[token_id]
                        for distribution, token_id in zip(fused_distributions, output_ids, strict=True)
                    ],
                    "pds_source_top_k": [top_k(distribution) for distribution in source_a],
                    "pds_fused_top_k": [top_k(distribution) for distribution in fused_distributions],
                },
            },
            {
                "output_ids": list(output_ids),
                "meta_info": {"finish_reason": {"type": "stop"}},
            },
        ]


def build_prompt_b(prompt_a: str, memory: MemoryRecord | None) -> str:
    if memory is None:
        return prompt_a
    return (
        f"Previous problem: {memory.prompt}\n"
        f"Previous summary: {memory.summary}\n"
        f"Previous trajectory: {memory.trajectory}\n"
        f"Current problem: {prompt_a}"
    )


def align_topk(prompt_length: int, rows: list[list[Any]], fill_value: Any) -> list[list[Any]]:
    if prompt_length <= 0 or not rows:
        raise ValueError("The smoke requires a non-empty prompt and response.")
    width = len(rows[0])
    aligned = [[fill_value] * width for _ in range(prompt_length + len(rows))]
    aligned[prompt_length - 1 : prompt_length - 1 + len(rows)] = rows
    return aligned


def run_rollout(
    *,
    request_id: str,
    prompt_a: str,
    memory: FakeGlobalMemory,
    engine: FakeMixSGLang,
) -> dict[str, Any]:
    retrieved = memory.search(prompt_a)
    prompt_b = build_prompt_b(prompt_a, retrieved)
    sample_group = f"opsd-{request_id}"

    def sampling_params(weight: float, *, return_probabilities: bool) -> dict[str, Any]:
        custom_params = {
            "__pds_sample_group": sample_group,
            "__pds_fuse_method": "avg_probs",
            "__pds_fuse_weight": weight,
        }
        if return_probabilities:
            custom_params.update(
                {
                    "__pds_return_prob_trajectory": True,
                    "__pds_return_top_k": 2,
                }
            )
        return {"custom_params": custom_params}

    raw_outputs = engine.generate_group(
        [
            {
                "prompt": prompt_a,
                "sampling_params": sampling_params(1.0, return_probabilities=True),
            },
            {
                "prompt": prompt_b,
                "sampling_params": sampling_params(1.0, return_probabilities=False),
            },
        ]
    )
    if raw_outputs[0]["output_ids"] != raw_outputs[1]["output_ids"]:
        raise AssertionError("PDS members returned divergent trajectories.")

    parsed = extract_pds_probability_fields(
        raw_outputs[0]["meta_info"],
        output_token_ids=raw_outputs[0]["output_ids"],
        expected_topk=2,
    )

    output_ids = raw_outputs[0]["output_ids"]
    trajectory = " ".join(map(str, output_ids))
    summary = f"Reusable trajectory with {len(output_ids)} generated tokens."
    memory.upsert(MemoryRecord(request_id, prompt_a, trajectory, summary))

    prompt_length = len(prompt_a.split())
    source_ids = align_topk(prompt_length, parsed["source_topk_ids"], 0)
    source_logprobs = align_topk(prompt_length, parsed["source_topk_logprobs"], 0.0)
    fused_ids = align_topk(prompt_length, parsed["fused_topk_ids"], 0)
    fused_logprobs = align_topk(prompt_length, parsed["fused_topk_logprobs"], 0.0)
    return {
        "request_id": request_id,
        "retrieved_request_id": retrieved.request_id if retrieved else None,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "response_ids": output_ids,
        "source_topk_ids": source_ids,
        "source_topk_logprobs": source_logprobs,
        "fused_topk_ids": fused_ids,
        "fused_topk_logprobs": fused_logprobs,
        "teacher_ids": fused_ids,
        "teacher_logprobs": fused_logprobs,
        "summary": summary,
    }


def consume_training_batch(batch: list[dict[str, Any]]) -> float:
    """Check causal target placement and consume fused targets as a toy KL."""
    total_loss = 0.0
    positions = 0
    for sample in batch:
        prompt_length = len(sample["prompt_a"].split())
        response_length = len(sample["response_ids"])
        target_rows = sample["teacher_logprobs"][prompt_length - 1 : prompt_length - 1 + response_length]
        if len(target_rows) != response_length:
            raise AssertionError("Fused targets are not aligned with response logits.")
        for row in target_rows:
            for target_logprob in row:
                target_probability = math.exp(target_logprob)
                student_logprob = math.log(max(target_probability * 0.9, 1e-12))
                total_loss += target_probability * (target_logprob - student_logprob)
            positions += 1
    if positions == 0 or total_loss <= 0:
        raise AssertionError("The toy training consumer did not receive valid fused targets.")
    return total_loss / positions


def main() -> None:
    memory = FakeGlobalMemory()
    engine = FakeMixSGLang()
    first = run_rollout(
        request_id="request-1",
        prompt_a="solve the quadratic equation by factoring",
        memory=memory,
        engine=engine,
    )
    second = run_rollout(
        request_id="request-2",
        prompt_a="factor this quadratic equation",
        memory=memory,
        engine=engine,
    )
    batch = [first, second]

    assert first["retrieved_request_id"] is None
    assert first["prompt_b"] == first["prompt_a"]
    assert second["retrieved_request_id"] == "request-1"
    assert f"Previous problem: {first['prompt_a']}" in second["prompt_b"]
    assert "Previous summary:" in second["prompt_b"]
    assert set(memory.records) == {"request-1", "request-2"}
    assert engine.batch_calls == 2
    assert all(sample["teacher_ids"] == sample["fused_topk_ids"] for sample in batch)
    toy_loss = consume_training_batch(batch)

    print("[1/6] global memory retrieval: OK")
    print("[2/6] prompt B construction: OK")
    print("[3/6] one native A/B batch per trajectory: OK")
    print("[4/6] prompt-A-only source/fused PDS response contract: OK")
    print("[5/6] request-id memory write-back: OK")
    print(f"[6/6] batch target consumption: OK (toy_forward_kl={toy_loss:.6f})")
    print("OPSD_FLOW_SMOKE_OK")


if __name__ == "__main__":
    main()
