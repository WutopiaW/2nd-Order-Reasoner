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

import pytest
from omegaconf import OmegaConf

from verl.trainer.distillation import uses_rollout_targets, uses_teacher_models
from verl.trainer.ppo.utils import need_teacher_policy
from verl.workers.config import DistillationConfig


def test_rollout_target_source_does_not_require_teacher_resources():
    config = DistillationConfig(
        enabled=True,
        target_source="rollout",
        n_gpus_per_node=0,
        nnodes=0,
        teacher_models={},
    )

    assert config.enabled
    assert not uses_teacher_models(config)
    assert uses_rollout_targets(config)


def test_teacher_is_the_backward_compatible_default_source():
    config = OmegaConf.create({"distillation": {"enabled": True}})

    assert need_teacher_policy(config)


def test_need_teacher_policy_is_false_for_rollout_targets():
    config = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "target_source": "rollout",
            }
        }
    )

    assert not need_teacher_policy(config)


def test_invalid_target_source_is_rejected_even_when_disabled():
    with pytest.raises(ValueError, match="target_source"):
        DistillationConfig(enabled=False, target_source="invalid")
