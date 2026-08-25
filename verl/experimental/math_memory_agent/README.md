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
6. return aligned fused targets for both prompt A and prompt B during training.

The dataset must provide an unboxed or boxed scalar answer at
`reward_model.ground_truth` and a non-empty worked reference solution at
`extra_info.solution`. The model response must contain a final boxed answer that
Math-Verify can extract. All three outcome-aware summaries compare the generated
trajectory with the reference solution. A response stopped by the generation
length limit receives a neutral summary that does not call the answer correct or
incorrect. For non-truncated responses, a missing/malformed answer, verifier
error, or verifier timeout is treated as
incorrect. The verifier still runs for truncated responses so its raw result
remains available for auditing.

The no-thinking setting applies only to summary generation. Prompt A and prompt
B keep their configured chat-template behavior. During training, prompt B uses
the OPSD teacher-style template with the retrieved problem, its saved reference
solution, and its summary as privileged context. The current problem's reference
solution is never inserted into its own prompt B; it is used only after rollout
generation for summary comparison and memory write-back. The recipe stores the
complete generated trajectory, including Qwen thinking, for auditing.

Each JSONL memory record also contains `trajectory_a` and `trajectory_b` as full
chat-message lists, plus the normalized verifier `ground_truth` and dataset
`solution`. The assistant message in both trajectories retains Qwen thinking
tags and their contents while omitting chat-template control tokens such as
`<|im_end|>`. Prompt retrieval uses the original user-message text. The `Current
problem` section of prompt B uses the decoded prompt A, intentionally preserving
its rendered `user` and `assistant` role markers. Prompt B also includes the
retrieved record's original problem and solution so its summary remains
grounded. When memory is used, `trajectory_b` records the exact prompt-B text
after token-budget trimming. The legacy string `trajectory` field remains
available for retrieval and backward compatibility.

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
summary generation, but they do not update global memory. During training, an
incorrect trajectory returns an all-zero `response_mask`, so it contributes no
direct distillation gradient; verification, summary generation, and memory
write-back still run for both outcomes. Each training rollout sends prompt A and
the privileged prompt B through one paired PDS request and returns both as
distillation samples supervised by the shared fused distribution. Prompt B's
targets are independently aligned to its token length. Before memory warmup,
prompt B is identical to prompt A and is not duplicated. Validation keeps the
original response mask and returns only prompt A.
