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
"""Backend-neutral helpers for token-level probability response protocols."""

import math
from collections.abc import Mapping, Sequence
from typing import Any



PDS_SELECTED_PROBABILITY_FIELDS = (
    "pds_source_token_probs",
    "pds_fused_token_probs",
)
PDS_TOP_K_FIELDS = (
    "pds_source_top_k",
    "pds_fused_top_k",
)
PDS_PROBABILITY_FIELDS = PDS_SELECTED_PROBABILITY_FIELDS + PDS_TOP_K_FIELDS


def _parse_logprob_entry(entry: Any, *, field_name: str) -> tuple[float, int]:
    """Parse an SGLang-style ``(logprob, token_id, ...)`` entry."""
    if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)) or len(entry) < 2:
        raise ValueError(f"{field_name} entries must be (logprob, token_id, ...), got {entry!r}.")
    return float(entry[0]), int(entry[1])


def extract_token_logprobs(
    entries: Any,
    *,
    expected_length: int,
    field_name: str,
) -> tuple[list[float], list[int]]:
    """Extract one sampled-token logprob and token id per response position."""
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence, got {type(entries).__name__}.")
    if len(entries) != expected_length:
        raise ValueError(f"{field_name} has {len(entries)} positions, expected {expected_length}.")

    logprobs: list[float] = []
    token_ids: list[int] = []
    for entry in entries:
        logprob, token_id = _parse_logprob_entry(entry, field_name=field_name)
        logprobs.append(logprob)
        token_ids.append(token_id)
    return logprobs, token_ids


def extract_topk_logprobs(
    entries: Any,
    *,
    expected_length: int,
    expected_topk: int | None,
    field_name: str,
) -> tuple[list[list[float]], list[list[int]]]:
    """Extract a fixed-width top-k distribution from SGLang logprob tuples."""
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence, got {type(entries).__name__}.")
    if len(entries) != expected_length:
        raise ValueError(f"{field_name} has {len(entries)} positions, expected {expected_length}.")

    all_logprobs: list[list[float]] = []
    all_token_ids: list[list[int]] = []
    inferred_topk = expected_topk
    for position, position_entries in enumerate(entries):
        if not isinstance(position_entries, Sequence) or isinstance(position_entries, (str, bytes)):
            raise ValueError(f"{field_name}[{position}] must be a sequence.")
        if inferred_topk is None:
            inferred_topk = len(position_entries)
        if len(position_entries) != inferred_topk:
            raise ValueError(
                f"{field_name}[{position}] has top-k width {len(position_entries)}, expected {inferred_topk}."
            )
        position_logprobs: list[float] = []
        position_token_ids: list[int] = []
        for entry in position_entries:
            logprob, token_id = _parse_logprob_entry(entry, field_name=field_name)
            position_logprobs.append(logprob)
            position_token_ids.append(token_id)
        all_logprobs.append(position_logprobs)
        all_token_ids.append(position_token_ids)
    return all_logprobs, all_token_ids


def _probability_to_logprob(value: Any, *, field_name: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain numeric probabilities, got {value!r}.") from exc
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError(f"{field_name} probabilities must be finite and in [0, 1], got {probability!r}.")
    return -math.inf if probability == 0.0 else math.log(probability)


def _extract_probability_trajectory(
    entries: Any,
    *,
    expected_length: int,
    field_name: str,
) -> list[float]:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence, got {type(entries).__name__}.")
    if len(entries) != expected_length:
        raise ValueError(f"{field_name} has {len(entries)} positions, expected {expected_length}.")
    return [_probability_to_logprob(value, field_name=field_name) for value in entries]


def _extract_probability_topk(
    entries: Any,
    *,
    expected_length: int,
    expected_topk: int,
    field_name: str,
) -> tuple[list[list[float]], list[list[int]]]:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence, got {type(entries).__name__}.")
    if len(entries) != expected_length:
        raise ValueError(f"{field_name} has {len(entries)} positions, expected {expected_length}.")

    all_logprobs: list[list[float]] = []
    all_token_ids: list[list[int]] = []
    for position, position_entries in enumerate(entries):
        if not isinstance(position_entries, Sequence) or isinstance(position_entries, (str, bytes)):
            raise ValueError(f"{field_name}[{position}] must be a sequence.")
        if len(position_entries) != expected_topk:
            raise ValueError(
                f"{field_name}[{position}] has top-k width {len(position_entries)}, expected {expected_topk}."
            )

        position_logprobs: list[float] = []
        position_token_ids: list[int] = []
        for rank, entry in enumerate(position_entries):
            if not isinstance(entry, Mapping) or "token_id" not in entry or "prob" not in entry:
                raise ValueError(
                    f"{field_name}[{position}][{rank}] must contain token_id and prob, got {entry!r}."
                )
            position_token_ids.append(int(entry["token_id"]))
            position_logprobs.append(
                _probability_to_logprob(entry["prob"], field_name=f"{field_name}[{position}][{rank}].prob")
            )
        all_logprobs.append(position_logprobs)
        all_token_ids.append(position_token_ids)
    return all_logprobs, all_token_ids


def extract_pds_probability_fields(
    meta_info: dict[str, Any],
    *,
    output_token_ids: list[int],
    expected_topk: int | None,
) -> dict[str, Any]:
    """Validate and convert the current mix-sglang PDS probability contract.

    mix-sglang returns raw probabilities so zero-probability entries remain valid
    JSON values.  verl converts them to log probabilities here; exact zeros become
    ``-inf`` and are handled by the configured distillation logprob clamp.
    """
    selected_present = [field for field in PDS_SELECTED_PROBABILITY_FIELDS if meta_info.get(field) is not None]
    topk_present = [field for field in PDS_TOP_K_FIELDS if meta_info.get(field) is not None]
    if not selected_present and not topk_present:
        return {}

    expected_length = len(output_token_ids)
    fields: dict[str, Any] = {}
    if selected_present:
        missing = [field for field in PDS_SELECTED_PROBABILITY_FIELDS if meta_info.get(field) is None]
        if missing:
            raise ValueError(f"Incomplete mix-sglang selected-token probability response; missing {missing}.")
        fields["source_log_probs"] = _extract_probability_trajectory(
            meta_info["pds_source_token_probs"],
            expected_length=expected_length,
            field_name="pds_source_token_probs",
        )
        fields["fused_log_probs"] = _extract_probability_trajectory(
            meta_info["pds_fused_token_probs"],
            expected_length=expected_length,
            field_name="pds_fused_token_probs",
        )

    if topk_present:
        missing = [field for field in PDS_TOP_K_FIELDS if meta_info.get(field) is None]
        if missing:
            raise ValueError(f"Incomplete mix-sglang PDS top-k response; missing {missing}.")
        if expected_topk is None or expected_topk <= 0:
            raise ValueError("mix-sglang PDS top-k fields require a positive expected_topk.")
        source_topk_logprobs, source_topk_ids = _extract_probability_topk(
            meta_info["pds_source_top_k"],
            expected_length=expected_length,
            expected_topk=expected_topk,
            field_name="pds_source_top_k",
        )
        fused_topk_logprobs, fused_topk_ids = _extract_probability_topk(
            meta_info["pds_fused_top_k"],
            expected_length=expected_length,
            expected_topk=expected_topk,
            field_name="pds_fused_top_k",
        )
        fields.update(
            {
                "source_topk_logprobs": source_topk_logprobs,
                "source_topk_ids": source_topk_ids,
                "fused_topk_logprobs": fused_topk_logprobs,
                "fused_topk_ids": fused_topk_ids,
            }
        )
    return fields
