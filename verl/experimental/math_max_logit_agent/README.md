# Math max-logit distillation AgentLoop

This agent inherits the retrieval, prompt-B construction, Math-Verify,
outcome-aware summary, and memory write-back behavior from
`MathOPSDMemoryAgentLoop`. It changes only the paired rollout target contract:

1. request `__pds_fuse_method=max_logits` from mix-sglang;
2. sample the shared A/B trajectory from the max-fused distribution;
3. request `__pds_return_top_k_logits=K` only on prompt A's response;
4. return only prompt A as a training sample;
5. fit prompt A's raw logits at the returned teacher token ids with
   `topk_logit_mse`.

The student support is always selected by the fused teacher. Prompt A's own
top-k ranking is not used. This is a local top-k raw-logit objective: it does
not reconstruct or normalize a full-vocabulary teacher distribution and does
not use a tail bucket.

Configure the rollout and loss with:

```bash
actor_rollout_ref.rollout.agent.default_agent_loop=math_max_logit_agent \
actor_rollout_ref.rollout.agent.agent_loop_config_path=verl/experimental/math_max_logit_agent/agent.yaml \
distillation.enabled=True \
distillation.target_source=rollout \
distillation.distillation_loss.loss_mode=topk_logit_mse \
distillation.distillation_loss.topk=32 \
distillation.distillation_loss.use_task_rewards=False \
distillation.distillation_loss.use_policy_gradient=False
```

The objective requires the eager logits-processor path. Set
`actor_rollout_ref.model.use_fused_kernels=False` for this loss. The remaining dataset and memory requirements are the
same as `verl/experimental/math_memory_agent/README.md`.
