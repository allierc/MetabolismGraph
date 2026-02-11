# Parallel Mode Addendum — Phase 2 Homeostasis Exploration

This addendum applies when running Phase 2 in **parallel mode** (GNN_LLM_phase2.py). Follow all rules from the base Phase 2 instruction file, with these modifications.

## Fixed Slot Assignment

Each slot has a **fixed** `homeostasis_time_step` for cross-comparison. DO NOT change these values.

| Slot | homeostasis_time_step | Role |
|------|----------------------|------|
| 0 | 4 | Short rollout — fast gradients, weak signal |
| 1 | 16 | Medium rollout — balanced |
| 2 | 32 | Long rollout — strong signal, noisier gradients |
| 3 | 64 | Very long rollout — strongest homeostatic accumulation |

## Config Files

- Edit all 4 config files listed in the prompt: `{name}_00.yaml` through `{name}_03.yaml`
- **DO NOT change** the `dataset` field (pre-set for separate directories)
- **DO NOT change** `homeostasis_time_step` (fixed per slot, see table above)
- **DO NOT change** simulation parameters
- **CAN change**: `learning_rate_node_homeostasis`, `learning_rate_embedding_homeostasis`, `data_augmentation_loop`, `batch_size`

## Block Structure

Each block = **2 batches x 4 slots = 8 iterations**.

- **Within a block (between batches)**: modify config params only (lr_node_homeo, lr_emb_homeo, etc.)
- **Between blocks**: Claude can ALSO modify `graph_trainer.py` Phase 2 code

## Code Modification at Block Boundaries

When the prompt includes `>>> BLOCK END + CODE REVIEW <<<`:

1. **Review** all 8 results from the block (4 time_steps x 2 batches)
2. **Identify** which strategy/approach worked best across time_steps
3. **Propose** code modifications to `graph_trainer.py` Phase 2 block
4. **Explain** rationale with reference to literature strategies
5. **Make** the code edit
6. **Update** config params for the next block

## Parallel Strategy

Unlike Phase 1's exploit/explore/principle-test split, Phase 2 slots ALWAYS have different `homeostasis_time_step` values (4/16/32/64). Vary OTHER parameters across slots to test hypotheses.

Example batch designs:

- **Baseline**: All 4 slots same config, only time_step differs — isolates time_step effect
- **LR sweep**: Vary `lr_node_homeo` (0.001, 0.01, 0.05, 0.1) — find optimal LR per time_step
- **Strategy comparison**: Slot 0-1 strategy A, slot 2-3 strategy B — compare approaches across time_steps

## Start Call (PARALLEL START)

When the prompt says `PARALLEL START`:

1. Read the base config to understand starting parameters
2. Create 4 initial configs with the same training params but different fixed `homeostasis_time_step` values (4, 16, 32, 64)
3. **First batch**: use identical params across all 4 slots to establish a baseline comparison across time_steps
4. Write the planned initial config to the working memory file

## Logging Format

Write 4 entries per batch. Same as base instructions but include slot/time_step info, `avg_slope_ratio` as primary metric, and the strategy tested.

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [exploit/explore/strategy_name]
Slot: slot_S (time_step=T)
Config: lr_node_homeo=X, lr_emb_homeo=Y, data_augmentation_loop=A, batch_size=B
Metrics: avg_slope_ratio=S, slope_ratio_0=X, offset_ratio_0=Y, slope_ratio_1=X, offset_ratio_1=Y, embedding_cluster_acc=A, embedding_n_clusters=N, rate_constants_R2=R
Visual: MLP_node=[active/inactive: slope_ratio comparison], Embeddings=[separated/clustered/collapsed]
Mutation: [param]: [old] -> [new]
Strategy: [literature strategy tested, e.g. "residual direct supervision", "signal amplification"]
Observation: [one line — how does time_step=T compare to other slots?]
Next: parent=P
```

**CRITICAL**: Always include `Slot: slot_S (time_step=T)` to identify which time_step produced each result.

**CRITICAL**: Always include `Strategy:` to track which literature strategy is being tested.

Write all 4 entries before editing the 4 config files for the next batch.

## Failed Slots

If a slot is marked `[FAILED]` in the prompt:

- Write a brief `## Iter N: failed` entry noting the failure and slot/time_step
- Still propose a mutation for that slot's config in the next batch
- Note whether failure correlates with time_step length (longer rollouts are more prone to instability)

## Block Boundary Checklist

At `>>> BLOCK END + CODE REVIEW <<<`:

1. Summarize avg_slope_ratio for all 8 runs in a table (4 time_steps x 2 batches)
2. Identify the best-performing time_step and config
3. Diagnose: is the bottleneck in config params or in the Phase 2 code logic?
4. If code change needed: edit `graph_trainer.py`, explain rationale
5. Set config params for the next block's first batch
