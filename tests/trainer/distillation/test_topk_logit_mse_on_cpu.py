# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.fsdp.losses import compute_topk_logit_mse
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config.distillation import DistillationConfig, DistillationLossConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def test_topk_logit_mse_gathers_teacher_ids_not_student_topk():
    student_logits = torch.tensor(
        [[[10.0, 1.0, 2.0, 3.0], [7.0, 6.0, 5.0, 4.0]]],
        requires_grad=True,
    )
    teacher_ids = torch.nested.nested_tensor(
        [torch.tensor([[3, 1], [2, 0]], dtype=torch.int64)],
        layout=torch.jagged,
    )
    teacher_logits = torch.nested.nested_tensor(
        [torch.tensor([[1.0, 2.0], [3.0, 8.0]])],
        layout=torch.jagged,
    )

    outputs = compute_topk_logit_mse(
        student_logits=student_logits,
        teacher_topk_logits=teacher_logits,
        teacher_topk_ids=teacher_ids,
        config=SimpleNamespace(),
        data_format="thd",
    )

    # Position 0 gathers ids [3, 1] -> student [3, 1], never student top-1 id 0.
    torch.testing.assert_close(outputs["distillation_losses"], torch.tensor([[2.5, 2.5]]))
    outputs["distillation_losses"].sum().backward()
    assert student_logits.grad[0, 0, 0] == 0
    assert student_logits.grad[0, 0, 2] == 0
    assert student_logits.grad[0, 0, 3] != 0
    assert student_logits.grad[0, 0, 1] != 0


def test_topk_logit_mse_config_requires_rollout_targets_and_direct_loss():
    with pytest.raises(ValueError, match="use_policy_gradient=False"):
        DistillationLossConfig(loss_mode="topk_logit_mse", use_policy_gradient=True)

    loss = DistillationLossConfig(loss_mode="topk_logit_mse", use_policy_gradient=False)
    with pytest.raises(ValueError, match="target_source=rollout"):
        DistillationConfig(enabled=True, target_source="teacher", distillation_loss=loss)


def test_fsdp_engine_passes_raw_logits_to_topk_logit_mse_processor():
    raw_logits = torch.tensor([[[4.0, 2.0, -1.0], [6.0, 3.0, 0.0]]])
    output = SimpleNamespace(logits=raw_logits.clone())
    offsets = torch.tensor([0, 2], dtype=torch.int64)
    input_ids = torch.nested.nested_tensor_from_jagged(torch.tensor([1, 2]), offsets=offsets)
    micro_batch = TensorDict({"input_ids": input_ids}, batch_size=[])
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=True,
        pad_mode=DatasetPadMode.NO_PADDING,
        use_fused_kernels=False,
        calculate_entropy=False,
        calculate_sum_pi_squared=False,
        distillation_use_topk=True,
        distillation_loss_mode="topk_logit_mse",
        distillation_only=True,
        max_response_length=2,
    )
    captured = {}

    def processor(*, student_logits, data):
        del data
        captured["student_logits"] = student_logits.detach().clone()
        shape = student_logits.shape[:2]
        return {
            "distillation_losses": torch.zeros(shape),
            "logit_abs_error": torch.zeros(shape),
        }

    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False
    engine.engine_config = SimpleNamespace(entropy_checkpointing=False)
    FSDPEngineWithLMHead.prepare_model_outputs(
        engine,
        output=output,
        output_args={
            "input_ids_rmpad_rolled": torch.tensor([2, 1]),
            "temperature_rmpad": torch.tensor([2.0, 2.0]),
        },
        micro_batch=micro_batch,
        logits_processor_func=processor,
    )

    torch.testing.assert_close(captured["student_logits"], raw_logits)
