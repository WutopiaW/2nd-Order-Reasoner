# Math memory AgentLoop

This recipe specializes the reusable OPSD memory primitives for one-shot math
rollouts. It preserves the normal OPSD flow:

1. retrieve the closest global trajectory memory;
2. build prompt B from that memory;
3. run one synchronized A/B mix-sglang generation;
4. verify the completed trajectory with Math-Verify;
5. select a success or failure summary prompt, render it in Qwen's
   `enable_thinking=False` mode, and write the summary back;
6. return prompt A's aligned source/fused targets for rollout distillation.

The dataset must provide an unboxed or boxed scalar answer at
`reward_model.ground_truth`. The model response must contain a final boxed
answer that Math-Verify can extract. A missing/malformed answer, verifier error,
or verifier timeout is treated as incorrect.

The no-thinking setting applies only to summary generation. Prompt A and prompt
B keep their configured chat-template behavior. This recipe stores and retrieves
the trajectory and summary as generated; unlike the generic OPSD loop, it does
not post-process memory text with `extract_formal_response()`.

Configure the rollout with:

```bash
actor_rollout_ref.rollout.agent.default_agent_loop=math_memory_agent \
actor_rollout_ref.rollout.agent.agent_loop_config_path=verl/experimental/math_memory_agent/agent.yaml
```

The remaining mix-sglang PDS, memory embedder, and rollout-distillation settings
are the same as `examples/opsd_memory/run_opsd_memory.sh`. In particular,
`distillation.target_source=rollout` and `forward_kl_topk` consume the fused
targets returned by this loop.

Each output exposes `math_verifier_score`, `math_answer_correct`, and
`math_verifier_seconds` in `extra_fields`. Validation rollouts still run the
verifier and summary generation, but they do not update global memory.
