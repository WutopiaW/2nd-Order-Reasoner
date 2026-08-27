# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import torch

from verl.experimental.agent_loop.opsd_memory_base import OPSDPromptPair
from verl.experimental.math_max_logit_agent.agent_loop import (
    MathMaxLogitDistillationAgentLoop,
    MaxLogitTurnOutput,
)
from verl.workers.rollout.replica import TokenOutput


def _loop() -> MathMaxLogitDistillationAgentLoop:
    loop = object.__new__(MathMaxLogitDistillationAgentLoop)
    loop.target_topk = 2
    loop.tokenizer = SimpleNamespace(pad_token_id=0)
    return loop


def test_process_max_logit_outputs_and_align_on_prompt_a_positions():
    loop = _loop()
    pair = OPSDPromptPair(prompt_a_ids=[10, 11, 12], prompt_b_ids=[20, 21], max_new_tokens=2)
    metadata = {
        "fused_topk_ids": [[7, 8], [9, 10]],
        "fused_topk_logits": [[5.0, 1.0], [4.0, -2.0]],
    }
    outputs = [
        TokenOutput(token_ids=[7, 9], stop_reason="stop", extra_fields=metadata),
        TokenOutput(token_ids=[7, 9], stop_reason="stop", extra_fields={}),
    ]

    turn = loop.process_max_logit_outputs(outputs, prompt_pair=pair)
    teacher_ids, teacher_logits = loop._align_teacher_targets(
        prompt_ids=pair.prompt_a_ids,
        response_ids=turn.token_ids,
        turn_output=turn,
    )

    assert teacher_ids.shape == teacher_logits.shape == (5, 2)
    torch.testing.assert_close(teacher_ids[2:4], torch.tensor([[7, 8], [9, 10]], dtype=torch.int32))
    torch.testing.assert_close(teacher_logits[2:4], torch.tensor([[5.0, 1.0], [4.0, -2.0]]))
    assert torch.count_nonzero(teacher_logits[:2]) == 0
    assert torch.count_nonzero(teacher_logits[4:]) == 0


def test_max_logit_turn_type_is_accepted_by_inherited_truncation_check():
    turn = MaxLogitTurnOutput(
        prompt_pair=OPSDPromptPair(prompt_a_ids=[1], prompt_b_ids=[2], max_new_tokens=2),
        token_ids=[3, 4],
        fused_topk_ids=[[3, 4], [4, 3]],
        fused_topk_logits=[[2.0, 1.0], [3.0, 0.0]],
        stop_reason="length",
        num_preempted=None,
        extra_fields={},
    )
    assert MathMaxLogitDistillationAgentLoop._response_was_truncated(turn, response_limit=2)
