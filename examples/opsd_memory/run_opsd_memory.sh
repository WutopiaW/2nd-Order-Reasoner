#!/usr/bin/env bash
#
# Memory-augmented rollout self-distillation with the mix-sglang PDS fork.

set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
TRAIN_FILES=${TRAIN_FILES:-"$HOME/data/gsm8k/train.parquet"}
VAL_FILES=${VAL_FILES:-"$HOME/data/gsm8k/test.parquet"}
: "${OPSD_EMBEDDING_MODEL:?Set OPSD_EMBEDDING_MODEL to a local semantic embedding model path}"

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
TOPK=${TOPK:-32}
export OPSD_TARGET_TOPK="$TOPK"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
MEMORY_PROMPT_MAX_LENGTH=${MEMORY_PROMPT_MAX_LENGTH:-2048}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-8192}
if ((MEMORY_PROMPT_MAX_LENGTH < MAX_PROMPT_LENGTH)); then
    echo "MEMORY_PROMPT_MAX_LENGTH must be at least MAX_PROMPT_LENGTH" >&2
    exit 2
fi
export OPSD_MEMORY_PROMPT_MAX_LENGTH="$MEMORY_PROMPT_MAX_LENGTH"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AGENT_CONFIG="$SCRIPT_DIR/agent.yaml"
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MEMORY_PROMPT_MAX_LENGTH + MAX_RESPONSE_LENGTH + 1))}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.default_agent_loop=opsd_memory_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_CONFIG" \
    distillation.enabled=True \
    distillation.target_source=rollout \
    distillation.distillation_loss.loss_mode=forward_kl_topk \
    distillation.distillation_loss.topk="$TOPK" \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=False \
    distillation.distillation_loss.log_prob_min_clamp=-20.0 \
    trainer.logger=console \
    trainer.project_name=verl_opsd_memory \
    trainer.experiment_name=mix_sglang_opsd_memory \
    trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.val_before_train=False \
    "$@"
