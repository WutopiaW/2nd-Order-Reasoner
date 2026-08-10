# Memory-augmented OPSD with mix-sglang

This example implements a rollout-driven variant of on-policy self-distillation:

1. Search a Ray-owned global trajectory memory for the record most similar to
   prompt A.
2. Render prompt B from prompt A plus the retrieved trajectory and summary.
3. Send A and B to one mix-sglang replica in a single native batch as a
   two-member PDS sample group.
4. Sample a shared trajectory from the weighted mixture of the two next-token
   distributions.
5. Summarize and store the new trajectory under a unique request ID.
6. Attach the fused top-k distribution to the rollout batch and minimize
   forward KL from that target to the policy conditioned on prompt A.

This differs from conventional OPD: the sampled behavior policy and target are
the A/B mixture, rather than a separately hosted teacher.

## Components

- `agent.yaml` configures `OPSDMemoryAgentLoop`.
- `verl.experimental.agent_loop.trajectory_memory` owns request-ID-keyed
  records and cosine retrieval. The named Ray actor makes one memory visible
  to every agent-loop worker in the job.
- `verl.workers.rollout.llm_server.LLMServerClient.generate_group` routes both
  group members to the same rollout replica.
- `distillation.target_source=rollout` enables the existing top-k distillation
  loss without allocating a teacher resource pool.

Each returned `DataProto.batch` contains the causally aligned tensor pairs
`source_topk_ids`/`source_topk_logprobs` and
`fused_topk_ids`/`fused_topk_logprobs`. The fused pair is also exposed through
the existing `teacher_ids`/`teacher_logprobs` keys consumed by
`forward_kl_topk`.

The production config requires `OPSD_EMBEDDING_MODEL` and uses
`HuggingFaceTextEmbedder`. Embedding inference remains on CPU by default,
although it may become the memory actor's throughput bottleneck. The hashing
embedder remains available as an explicitly documented lexical fallback for
smoke tests only.

The memory lives for the Ray job and has bounded insertion-order eviction.
`TrajectoryMemoryActor.state_dict()` and `load_state_dict()` are provided for
external persistence, but trainer checkpoint integration is not automatic.
When prompt B would exceed the model context, the loop trims the oldest tokens
from the retrieved trajectory and then its summary while reserving
`min_response_tokens` for the shared rollout.

## Required mix-sglang response contract

The local mix-sglang PDS branch already accepts
`sampling_params.custom_params` containing:

- `__pds_sample_group`
- `__pds_fuse_method="avg_probs"`
- `__pds_fuse_weight`
- `__pds_return_prob_trajectory=true`
- `__pds_return_top_k=distillation.distillation_loss.topk`

Prompt A and prompt B are submitted as one native SGLang batch request. The
tokenizer manager forwards that batch to the scheduler as one
`BatchTokenizedGenerateReqInput`, so both PDS members are registered before
either member can enter deferred sampling.

For every generated position, mix-sglang must return these arrays in
`meta_info`:

- `pds_source_token_probs`
- `pds_fused_token_probs`
- `pds_source_top_k`
- `pds_fused_top_k`

The first two arrays contain the emitted token's raw probability. Each Top-K
position is a fixed-width list of `{"token_id": ..., "prob": ...}` entries.
All four arrays must have the same number of positions as `output_ids`. verl
converts the raw probabilities to log probabilities before constructing the
rollout batch; exact zeros become `-inf` and are handled by the configured loss
clamp.

`source` means the current request's normalized, post-sampling-filter
distribution. `fused` means the normalized weighted distribution used to
sample the shared token. Top-k logprobs must remain probabilities under the
full distribution; do not renormalize only the returned top-k entries.

verl fails closed if any dedicated PDS field is absent or misaligned. Ordinary
`output_token_logprobs` and `output_top_logprobs` are deliberately not accepted
as source-distribution fallbacks because they may have been computed before the
deferred fused token was selected.

This contract is implemented by mix-sglang commits `6fe87c2c8` (selected-token
probability trajectories) and `dd6da0198` (source/fused Top-K distributions).

## Data and launch

The RL dataset must contain the normal `prompt` column. If it contains an
`agent_name` column, its value must be `opsd_memory_agent`; otherwise the
configured default loop is used.

Install the mix-sglang fork so `import sglang` resolves to it, then adjust the
paths and resource settings:

```bash
MODEL_PATH=/path/to/model \
OPSD_EMBEDDING_MODEL=/path/to/semantic-embedding-model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/val.parquet \
bash examples/opsd_memory/run_opsd_memory.sh
```

The launch configuration uses supervised `forward_kl_topk` only:

- `distillation.target_source=rollout`
- `distillation.distillation_loss.loss_mode=forward_kl_topk`
- task-reward and policy-gradient terms disabled

No separate teacher model is started. GPU/NPU execution is still required for
the actual actor update and mix-sglang rollout.
