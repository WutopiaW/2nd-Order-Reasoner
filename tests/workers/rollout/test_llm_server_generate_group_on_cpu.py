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

import asyncio
from types import MethodType, SimpleNamespace

import pytest

from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput


@pytest.mark.asyncio
async def test_generate_group_acquires_once_and_uses_one_server():
    client = LLMServerClient(config=SimpleNamespace(), load_balancer_handle=None)
    selected_server = object()
    acquired = []
    released = []
    calls = []

    async def fake_acquire(self, request_id):
        acquired.append(request_id)
        return "server-0", selected_server

    def fake_release(self, server_id):
        released.append(server_id)

    async def fake_generate(self, server, *, prompt_ids, sampling_params, **kwargs):
        calls.append((server, list(prompt_ids), dict(sampling_params)))
        await asyncio.sleep(0)
        return TokenOutput(token_ids=[prompt_ids[-1]], extra_fields={"global_steps": 1})

    client._acquire_server = MethodType(fake_acquire, client)
    client._release_server = MethodType(fake_release, client)
    client._generate_on_server = MethodType(fake_generate, client)

    outputs = await client.generate_group(
        request_id="logical-group",
        requests=[
            {"prompt_ids": [1, 2], "sampling_params": {"temperature": 1.0}},
            {"prompt_ids": [3, 4], "sampling_params": {"temperature": 1.0}},
        ],
    )

    assert acquired == ["logical-group"]
    assert released == ["server-0"]
    assert [call[0] for call in calls] == [selected_server, selected_server]
    assert [output.token_ids for output in outputs] == [[2], [4]]


@pytest.mark.asyncio
async def test_generate_group_rejects_empty_group_before_acquiring():
    client = LLMServerClient(config=SimpleNamespace(), load_balancer_handle=None)

    with pytest.raises(ValueError, match="at least one"):
        await client.generate_group(request_id="empty", requests=[])


@pytest.mark.asyncio
async def test_sglang_group_uses_one_backend_actor_invocation():
    captured_requests = []

    class RemoteMethod:
        async def _call(self, *, requests):
            captured_requests.extend(requests)
            return [
                TokenOutput(token_ids=[request["prompt_ids"][-1]], extra_fields={"global_steps": 7})
                for request in requests
            ]

        def remote(self, **kwargs):
            return self._call(**kwargs)

    fake_server = SimpleNamespace(generate_group=RemoteMethod())
    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(name="sglang")),
    )
    client = LLMServerClient(config=config, load_balancer_handle=None)
    released = []

    async def fake_acquire(self, request_id):
        return "server-0", fake_server

    def fake_release(self, server_id):
        released.append(server_id)

    client._acquire_server = MethodType(fake_acquire, client)
    client._release_server = MethodType(fake_release, client)

    outputs = await client.generate_group(
        request_id="pds-group",
        requests=[
            {"prompt_ids": [1], "sampling_params": {}, "priority": 3},
            {"prompt_ids": [2], "sampling_params": {}, "priority": 4},
        ],
    )

    assert len(captured_requests) == 2
    assert all("request_id" in request for request in captured_requests)
    assert all("priority" not in request for request in captured_requests)
    assert [output.extra_fields["min_global_steps"] for output in outputs] == [7, 7]
    assert released == ["server-0"]
