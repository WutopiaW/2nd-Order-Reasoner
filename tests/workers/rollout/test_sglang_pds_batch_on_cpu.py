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
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")
pytest.importorskip("ray")

import verl.workers.rollout.sglang_rollout.async_sglang_server as async_sglang_server
from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer


def _pds_meta(source_prob: float) -> dict:
    return {
        "finish_reason": {"type": "stop"},
        "pds_source_token_probs": [source_prob],
        "pds_fused_token_probs": [0.5],
        "pds_source_top_k": [[{"token_id": 7, "prob": source_prob}, {"token_id": 8, "prob": 0.1}]],
        "pds_fused_top_k": [[{"token_id": 7, "prob": 0.5}, {"token_id": 8, "prob": 0.25}]],
    }


@pytest.mark.asyncio
async def test_pds_group_uses_one_native_sglang_batch(monkeypatch):
    captured_requests = []

    class FakeGenerateReqInput:
        def __init__(self, **kwargs):
            captured_requests.append(kwargs)

    class FakeTokenizerManager:
        def generate_request(self, request, _):
            async def responses():
                yield [
                    {"output_ids": [7], "meta_info": _pds_meta(0.4)},
                    {"output_ids": [7], "meta_info": {"finish_reason": {"type": "stop"}}},
                ]

            return responses()

    monkeypatch.setattr(async_sglang_server, "GenerateReqInput", FakeGenerateReqInput)
    monkeypatch.setattr(
        async_sglang_server.RLInsightLogger,
        "trace_state",
        staticmethod(lambda *args, **kwargs: nullcontext()),
    )

    server = object.__new__(SGLangHttpServer)
    server.config = SimpleNamespace(
        max_model_len=64,
        prompt_length=16,
        response_length=8,
        enable_rollout_routing_replay=False,
        mtp=None,
    )
    server.model_config = SimpleNamespace(lora_rank=0)
    server.tokenizer_manager = FakeTokenizerManager()
    server.replica_rank = 0
    server.global_steps = 3

    def request(prompt_ids, weight, *, return_probabilities):
        custom_params = {
            "__pds_sample_group": "group-1",
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
        return {
            "request_id": f"request-{weight}",
            "prompt_ids": prompt_ids,
            "sampling_params": {
                "max_new_tokens": 1,
                "custom_params": custom_params,
            },
        }

    outputs = await server._generate_group_local(
        [
            request([1, 2], 1.0, return_probabilities=True),
            request([3, 4], 2.0, return_probabilities=False),
        ]
    )

    assert len(captured_requests) == 1
    assert captured_requests[0]["input_ids"] == [[1, 2], [3, 4]]
    assert len(captured_requests[0]["sampling_params"]) == 2
    captured_custom_params = [params["custom_params"] for params in captured_requests[0]["sampling_params"]]
    assert captured_custom_params[0]["__pds_return_prob_trajectory"] is True
    assert captured_custom_params[0]["__pds_return_top_k"] == 2
    assert "__pds_return_prob_trajectory" not in captured_custom_params[1]
    assert "__pds_return_top_k" not in captured_custom_params[1]
    assert outputs[0].token_ids == outputs[1].token_ids == [7]
    assert outputs[0].log_probs == pytest.approx([math.log(0.5)])
    assert outputs[0].extra_fields["fused_topk_ids"] == [[7, 8]]
    assert outputs[1].log_probs is None
    assert "source_topk_ids" not in outputs[1].extra_fields
    assert "fused_topk_ids" not in outputs[1].extra_fields


def test_convert_generate_output_accepts_topk_without_selected_probs():
    server = object.__new__(SGLangHttpServer)
    server.config = SimpleNamespace(
        max_model_len=64,
        prompt_length=16,
        response_length=8,
        enable_rollout_routing_replay=False,
        mtp=None,
    )
    server.model_config = SimpleNamespace(lora_rank=0)
    server.global_steps = 3
    _, context = server._prepare_generate_request(
        prompt_ids=[1, 2],
        sampling_params={
            "max_new_tokens": 1,
            "custom_params": {"__pds_return_top_k": 2},
        },
        request_id="request-a",
    )
    output = {
        "output_ids": [7],
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "pds_source_top_k": [[{"token_id": 7, "prob": 0.7}, {"token_id": 8, "prob": 0.2}]],
            "pds_fused_top_k": [[{"token_id": 7, "prob": 0.8}, {"token_id": 8, "prob": 0.1}]],
        },
    }

    converted = server._convert_generate_output(output, context)

    assert converted.token_ids == [7]
    assert converted.log_probs is None
    assert converted.extra_fields["source_topk_ids"] == [[7, 8]]
    assert converted.extra_fields["fused_topk_ids"] == [[7, 8]]
