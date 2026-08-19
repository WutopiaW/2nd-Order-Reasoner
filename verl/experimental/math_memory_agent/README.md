# Math memory AgentLoop

This recipe specializes the reusable OPSD memory primitives for one-shot math
rollouts. It preserves the normal OPSD flow:

1. retrieve the closest global trajectory memory;
2. build prompt B from that memory;
3. run one synchronized A/B mix-sglang generation;
4. verify the completed trajectory with Math-Verify;
5. classify the response as truncated, correct, or incorrect, then select the
   corresponding summary prompt and render it in Qwen's
   `enable_thinking=False` mode, and write the summary back;
6. return prompt A's aligned source/fused targets for rollout distillation.

The dataset must provide an unboxed or boxed scalar answer at
`reward_model.ground_truth`. The model response must contain a final boxed
answer that Math-Verify can extract. A response stopped by the generation length
limit receives a neutral summary that captures useful partial progress without
calling the answer correct or incorrect. For non-truncated responses, a
missing/malformed answer, verifier error, or verifier timeout is treated as
incorrect. The verifier still runs for truncated responses so its raw result
remains available for auditing.

The no-thinking setting applies only to summary generation. Prompt A and prompt
B keep their configured chat-template behavior. The recipe stores the complete
trajectory, including Qwen thinking, for auditing. When a record is retrieved to
construct prompt B, `extract_formal_response()` removes the prior trajectory's
`<think>...</think>` block so only its concise formal response is reused.

Each JSONL memory record also contains `trajectory_a` and `trajectory_b` as full
chat-message lists, plus the normalized verifier `ground_truth`. The assistant
message in both trajectories retains Qwen thinking tags and their contents while
omitting chat-template control tokens such as `<|im_end|>`. Prompt retrieval uses
the original user-message text. The `Current problem` section of prompt B uses
the decoded prompt A, intentionally preserving its rendered `user` and
`assistant` role markers. Prompt B also includes the retrieved record's original
problem so its summary and formal trajectory remain grounded. When memory is
used, `trajectory_b` records the exact prompt-B text after token-budget trimming.
The legacy string `trajectory` field remains available for retrieval and
backward compatibility.

Configure the rollout with:

```bash
actor_rollout_ref.rollout.agent.default_agent_loop=math_memory_agent \
actor_rollout_ref.rollout.agent.agent_loop_config_path=verl/experimental/math_memory_agent/agent.yaml
```

The remaining mix-sglang PDS, memory embedder, and rollout-distillation settings
are the same as `examples/opsd_memory/run_opsd_memory.sh`. In particular,
`distillation.target_source=rollout` and `forward_kl_topk` consume the fused
targets returned by this loop.

Each output exposes `math_verifier_score`, `math_answer_correct`,
`math_response_truncated`, `math_summary_outcome`, `rollout_stop_reason`, and
`math_verifier_seconds` in `extra_fields`. `math_answer_correct` is the raw
verifier classification, while `math_summary_outcome` records which of the
three summary prompts was used. Validation rollouts still run the verifier and
summary generation, but they do not update global memory.
