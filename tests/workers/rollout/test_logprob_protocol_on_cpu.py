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

import math

import pytest

from verl.workers.rollout.logprob_protocol import (
    extract_pds_probability_fields,
    extract_pds_topk_logits_fields,
    extract_token_logprobs,
    extract_topk_logprobs,
)


def test_extract_pds_topk_logits_fields_preserves_raw_values_and_teacher_order():
    fields = extract_pds_topk_logits_fields(
        {
            "pds_fused_top_k_logits": [
                [{"token_id": 7, "logit": 4.5}, {"token_id": 2, "logit": -1.25}],
                [{"token_id": 3, "logit": 8.0}, {"token_id": 9, "logit": 0.0}],
            ]
        },
        output_token_ids=[7, 3],
        expected_topk=2,
    )

    assert fields["fused_topk_ids"] == [[7, 2], [3, 9]]
    assert fields["fused_topk_logits"] == [[4.5, -1.25], [8.0, 0.0]]


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([[{"token_id": 1, "logit": float("nan")}]], "finite"),
        (
            [[{"token_id": 1, "logit": 2.0}, {"token_id": 1, "logit": 1.0}]],
            "duplicate",
        ),
    ],
)
def test_extract_pds_topk_logits_fields_fails_closed(entries, message):
    with pytest.raises(ValueError, match=message):
        extract_pds_topk_logits_fields(
            {"pds_fused_top_k_logits": entries},
            output_token_ids=[1],
            expected_topk=len(entries[0]),
        )


def test_extract_token_logprobs_accepts_sglang_entries():
    logprobs, token_ids = extract_token_logprobs(
        [(-0.1, 12, "a"), [-0.2, 34, "b"]],
        expected_length=2,
        field_name="tokens",
    )

    assert logprobs == [-0.1, -0.2]
    assert token_ids == [12, 34]


def test_extract_topk_logprobs_preserves_position_and_rank():
    logprobs, token_ids = extract_topk_logprobs(
        [
            [(-0.1, 1, None), (-0.2, 2, None)],
            [(-0.3, 3, None), (-0.4, 4, None)],
        ],
        expected_length=2,
        expected_topk=2,
        field_name="topk",
    )

    assert logprobs == [[-0.1, -0.2], [-0.3, -0.4]]
    assert token_ids == [[1, 2], [3, 4]]


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([(-0.1, 1)], "positions"),
        ([[(-0.1, 1)], [(-0.2, 2), (-0.3, 3)]], "top-k width"),
    ],
)
def test_extract_topk_logprobs_validates_shape(entries, message):
    with pytest.raises(ValueError, match=message):
        extract_topk_logprobs(
            entries,
            expected_length=2,
            expected_topk=1,
            field_name="topk",
        )


def test_extract_pds_probability_fields_converts_complete_mix_contract():
    meta_info = {
        "pds_source_token_probs": [0.5, 0.25],
        "pds_fused_token_probs": [0.75, 0.5],
        "pds_source_top_k": [
            [{"token_id": 10, "prob": 0.5}, {"token_id": 11, "prob": 0.25}],
            [{"token_id": 20, "prob": 0.4}, {"token_id": 21, "prob": 0.3}],
        ],
        "pds_fused_top_k": [
            [{"token_id": 10, "prob": 0.75}, {"token_id": 12, "prob": 0.1}],
            [{"token_id": 20, "prob": 0.5}, {"token_id": 22, "prob": 0.2}],
        ],
    }

    fields = extract_pds_probability_fields(
        meta_info,
        output_token_ids=[10, 20],
        expected_topk=2,
    )

    assert fields["source_log_probs"] == pytest.approx([math.log(0.5), math.log(0.25)])
    assert fields["fused_log_probs"] == pytest.approx([math.log(0.75), math.log(0.5)])
    assert fields["source_topk_ids"] == [[10, 11], [20, 21]]
    assert fields["fused_topk_ids"] == [[10, 12], [20, 22]]
    assert fields["fused_topk_logprobs"][0] == pytest.approx([math.log(0.75), math.log(0.1)])


def test_extract_pds_probability_fields_preserves_zero_as_negative_infinity():
    meta_info = {
        "pds_source_token_probs": [0.0],
        "pds_fused_token_probs": [0.5],
        "pds_source_top_k": [[{"token_id": 10, "prob": 1.0}, {"token_id": 11, "prob": 0.0}]],
        "pds_fused_top_k": [[{"token_id": 10, "prob": 0.5}, {"token_id": 11, "prob": 0.5}]],
    }

    fields = extract_pds_probability_fields(meta_info, output_token_ids=[10], expected_topk=2)

    assert fields["source_log_probs"] == [-math.inf]
    assert fields["source_topk_logprobs"][0][1] == -math.inf


def test_extract_pds_probability_fields_accepts_topk_without_selected_token_probs():
    fields = extract_pds_probability_fields(
        {
            "pds_source_top_k": [[{"token_id": 10, "prob": 0.6}]],
            "pds_fused_top_k": [[{"token_id": 10, "prob": 0.7}]],
        },
        output_token_ids=[10],
        expected_topk=1,
    )

    assert "source_log_probs" not in fields
    assert fields["source_topk_ids"] == [[10]]
    assert fields["fused_topk_logprobs"][0] == pytest.approx([math.log(0.7)])


def test_extract_pds_probability_fields_rejects_partial_contract():
    with pytest.raises(ValueError, match="missing"):
        extract_pds_probability_fields(
            {"pds_fused_token_probs": [0.5]},
            output_token_ids=[10],
            expected_topk=2,
        )


def test_extract_pds_probability_fields_rejects_wrong_topk_width():
    meta_info = {
        "pds_source_token_probs": [0.5],
        "pds_fused_token_probs": [0.5],
        "pds_source_top_k": [[{"token_id": 10, "prob": 0.5}]],
        "pds_fused_top_k": [[{"token_id": 10, "prob": 0.5}]],
    }

    with pytest.raises(ValueError, match="top-k width"):
        extract_pds_probability_fields(meta_info, output_token_ids=[10], expected_topk=2)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_extract_pds_probability_fields_rejects_invalid_probability(probability):
    meta_info = {
        "pds_source_token_probs": [probability],
        "pds_fused_token_probs": [0.5],
        "pds_source_top_k": [[{"token_id": 10, "prob": 0.5}]],
        "pds_fused_top_k": [[{"token_id": 10, "prob": 0.5}]],
    }

    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        extract_pds_probability_fields(
            meta_info,
            output_token_ids=[10],
            expected_topk=1,
        )
