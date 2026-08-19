# Memory-augmented OPSD with mix-sglang

This example implements a rollout-driven variant of on-policy self-distillation:

1. Search a Ray-owned global trajectory memory for the record most similar to
   prompt A.
2. Render prompt B from prompt A plus the retrieved problem, trajectory, and
   summary.
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

- `agent.yaml` configures `SingleTurnOPSDMemoryAgentLoop`, the thin one-shot
  specialization used by this math example.
- `OPSDMemoryAgentLoopBase` deliberately has no `run` implementation. It
  exposes memory lookup, prompt-pair construction, paired generation, output
  validation, memory finalization, and a multi-turn target accumulator for
  reuse by tool/code AgentLoops with their own state machines.
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

The memory actor is detached and has bounded insertion-order eviction. Set
`OPSD_MEMORY_OUTPUT` to append each completed training trajectory to a JSONL
journal, and optionally set `OPSD_MEMORY_SEED` to rebuild memory from a previous
journal. Each saved record includes the complete retrieved memory under
`metadata.retrieved_memory` (request ID, retrieval score, prompt, trajectory,
and summary), so the experience used to construct prompt B is directly
auditable. Trainer checkpoint integration is not automatic.

The production embedding model is instantiated lazily inside the named global
memory actor. AgentLoops are created per trajectory, but they pass the nested
Hydra embedder config through without constructing the model, avoiding a Qwen3
embedding-model reload for every sample.

Qwen-style generation may still produce `<think>...</think>` blocks. The full
rollout tokens remain unchanged for OPSD training and the full trajectory is
available to the summarization request, but only text outside thinking blocks
is saved as the memory trajectory and summary. Retrieved seed records are
cleaned again before prompt B is constructed.

Prompt A uses the normal verl limits: the initial templated prompt must fit
`data.max_prompt_length`, the flattened response must fit
`data.max_response_length`. There is no separate prompt-A total-length setting
or prompt-A-dependent generation-budget calculation. Each paired request uses
the `max_new_tokens` supplied in its sampling parameters, defaulting to the
configured response length when absent. Prompt B is always trimmed to at most
`MEMORY_PROMPT_MAX_LENGTH`. The launch script defaults
`max_model_len` to `MEMORY_PROMPT_MAX_LENGTH + MAX_RESPONSE_LENGTH + 1`.
When prompt B exceeds its cap, it trims the retrieved trajectory first, then
the retrieved summary, and finally the retrieved problem while preserving the
current problem. Trajectory and summary trimming keeps their suffixes; retrieved
problem trimming keeps its prefix so the original setup and conditions survive.

## Reusing the base in a multi-turn AgentLoop

A tool/code AgentLoop keeps its own `run` state machine and calls the base at
four integration points:

1. Render the initial prompt with `initialize_prompt_a`, then call
   `initialize_memory_context` once for the complete trajectory identity.
2. On every model turn, call `generate_paired` with the current full prompt A
   and that turn's desired `max_new_tokens` in `sampling_params`.
3. Before appending that model turn, record it in an
   `OPSDTrajectoryTargetAccumulator` using its final response offset. Tool and
   environment response tokens are appended normally with `response_mask=0`
   and are not recorded as model turns; model-generated tool-call tokens remain
   part of the recorded model turn with `response_mask=1`.
4. After the state machine terminates, cap the flattened response, call the
   accumulator's `finalize`, summarize the full model/tool/environment
   trajectory with `finalize_memory`, and return the tensors in
   `AgentLoopOutput`.

The accumulator places each sampled distribution at causal-logit position
`initial_prompt_length - 1 + response_position`. Model positions must have
`response_mask=1`; tool/environment positions remain zero-filled target rows
with `response_mask=0`, so the distillation loss ignores them. It returns
unpadded tensors. The shared AgentLoop postprocessor still left-pads prompts
and right-pads responses, masks, logprobs, and top-k targets to the configured
batch widths. Continuous-token loops must provide an exact post-merge position
mapping; the accumulator fails closed if retokenization changes token IDs.
For a multi-turn loop, configure `memory_prompt_max_length` above the largest
current prompt-A context plus prompt-B's fixed template overhead; otherwise the
base correctly fails once the untrimmable current trajectory no longer fits.

## Required mix-sglang response contract

The local mix-sglang PDS branch already accepts
`sampling_params.custom_params` containing these fields on both group members:

- `__pds_sample_group`
- `__pds_fuse_method="avg_probs"`
- `__pds_fuse_weight`

Only prompt A requests the probability payload used for training:

- `__pds_return_prob_trajectory=true`
- `__pds_return_top_k=distillation.distillation_loss.topk`

Prompt A and prompt B are submitted as one native SGLang batch request. The
tokenizer manager forwards that batch to the scheduler as one
`BatchTokenizedGenerateReqInput`, so both PDS members are registered before
either member can enter deferred sampling.

For every generated position, prompt A's mix-sglang result must return these
arrays in `meta_info`:

- `pds_source_token_probs`
- `pds_fused_token_probs`
- `pds_source_top_k`
- `pds_fused_top_k`

The first two arrays contain the emitted token's raw probability. Each Top-K
position is a fixed-width list of `{"token_id": ..., "prob": ...}` entries.
All four arrays must have the same number of positions as `output_ids`. Prompt B
still participates in every forward, fusion, and shared sampling step, but it
does not duplicate the probability payload. verl converts prompt A's raw
probabilities to log probabilities before constructing the rollout batch;
exact zeros become `-inf` and are handled by the configured loss clamp.

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
MEMORY_PROMPT_MAX_LENGTH=2048 \
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

## Dependency-free flow smoke

On a development machine without PyTorch, Ray, SGLang, or an accelerator, run:

```bash
python3 examples/opsd_memory/smoke_flow.py
```

This uses deterministic fake memory and fake generation, but imports the real
`verl/workers/rollout/logprob_protocol.py` response parser. It verifies the
complete control/data flow: memory retrieval, prompt-B construction, one A/B
batch call, source/fused probability parsing, request-ID write-back, causal
target alignment, batch construction, and target consumption. It does not
validate model numerics, Ray concurrency, the real mix-sglang scheduler, or an
actor optimizer step.
