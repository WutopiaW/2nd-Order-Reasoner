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

import torch

from verl.trainer.ppo.padding_utils import construct_minimal_padding_template


def test_minimal_padding_template_rebuilds_all_topk_distribution_tensors():
    seq_len, topk = 7, 3
    source = {
        "prompts": torch.tensor([1, 2, 3]),
        "responses": torch.tensor([4, 5, 6, 7]),
        "input_ids": torch.arange(seq_len),
        "attention_mask": torch.ones(seq_len),
        "position_ids": torch.arange(seq_len),
        "response_mask": torch.ones(4),
    }
    for index, prefix in enumerate(("teacher", "source_topk", "fused_topk")):
        source[f"{prefix}_ids"] = torch.full((seq_len, topk), index + 10, dtype=torch.int32)
        source[f"{prefix}_logprobs"] = torch.full((seq_len, topk), -float(index + 1))

    padded, tag = construct_minimal_padding_template(
        source_td=source,
        source_tag={"prompt_len": 3, "response_len": 4, "seq_len": 7},
        eos_token_id=99,
    )

    assert tag["is_padding"]
    assert tag["seq_len"] == 2
    for prefix in ("teacher", "source_topk", "fused_topk"):
        assert padded[f"{prefix}_ids"].shape == (2, topk)
        assert padded[f"{prefix}_logprobs"].shape == (2, topk)
        assert padded[f"{prefix}_ids"].eq(99).all()
        assert padded[f"{prefix}_logprobs"].eq(0).all()
