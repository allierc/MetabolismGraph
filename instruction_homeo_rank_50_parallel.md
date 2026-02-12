# Parallel Mode Addendum — Joint Metabolism Recovery (Homeostasis Active)

This addendum applies when running in **parallel mode** (GNN_LLM_parallel.py). Follow all rules from the base instruction file, with these modifications.

## Batch Processing

- You receive **4 results per batch** and must propose **4 mutations**
- Each slot has its own config file, metrics log, and activity image
- Write **4 separate** `## Iter N:` log entries (one per slot/iteration)
- Each iteration gets its own Node id in the UCB tree

## Config Files

- Edit all 4 config files listed in the prompt: `{name}_00.yaml` through `{name}_03.yaml`
- Each config's `dataset` field is pre-set to route data to separate directories -- **DO NOT change the `dataset` field**
- **DO NOT change `simulation:` parameters** -- this is a fixed-regime exploration
- Modify `training:` parameters: `lr_k`, `lr_embedding`, `lr_node`, `lr_sub`, `batch_size`, `n_epochs`, `data_augmentation_loop`, `coeff_MLP_node_L1`, `coeff_MLP_sub_norm`, `coeff_MLP_sub_diff`, `coeff_k_floor`, `k_floor_threshold`, `time_step`, `recurrent_training`
- Modify `graph_model:` architecture: `hidden_dim_sub`, `n_layers_sub`, `hidden_dim_node`, `n_layers_node`

## Parallel UCB Strategy

When selecting parents for 4 simultaneous mutations, **diversify** your choices:

| Slot | Role | Description |
| ---- | ---- | ----------- |
| 0 | **exploit** | Highest UCB node, conservative mutation |
| 1 | **exploit** | 2nd highest UCB node, or same parent different param |
| 2 | **explore** | Under-visited node, or new parameter dimension |
| 3 | **principle-test** | Test or challenge one Established Principle from memory.md |

You may deviate from this split based on context (e.g., all exploit if early in block, all explore if everything fails).

**When rate_constants_R2 plateaus but homeostasis is not learned**: focus mutations on lr_node, lr_embedding, and coeff_MLP_node_L1 while keeping k-related parameters stable.

### Slot 3: Principle Testing

1. Read the "Established Principles" section in memory.md
2. **Randomly select one principle** (rotate -- do not repeat consecutively)
3. Design a config that specifically tests this principle
4. In the log entry, write: `Mode/Strategy: principle-test`
5. In the Mutation line, include: `Testing principle: "[quoted principle text]"`
6. After results, update the principle's evidence level in memory.md

If there are no Established Principles yet, use slot 3 as a **boundary-probe** instead.

## Start Call (first batch, no results yet)

When the prompt says `PARALLEL START`:
- Read the base config to understand the starting training parameters
- Create 4 diverse initial training parameter variations
- Suggested initial spread across slots:
  - Slot 0: baseline (iter_116 settings unchanged)
  - Slot 1: higher lr_node (5E-4) to kick homeostasis learning
  - Slot 2: lower coeff_MLP_node_L1 (0.1) to let MLP_node grow
  - Slot 3: multi-step rollout (recurrent_training=true, time_step=4)
- See "Training Parameters Reference" in base instruction file for parameter descriptions and ranges
- All 4 slots share the same simulation parameters (DO NOT change them)
- Write the planned initial variations to the working memory file

## Logging Format

Same as base instructions, but you write 4 entries per batch:

```
## Iter N: [converged/partial-k/partial-homeo/failed]
Node: id=N, parent=P
Mode/Strategy: [strategy]
Config: seed=S, lr_k=X, lr_embedding=E, lr_node=Y, lr_sub=Z, batch_size=B, n_epochs=E, data_augmentation_loop=A, coeff_MLP_node_L1=L, coeff_MLP_sub_norm=N, coeff_k_floor=K, time_step=T
Metrics: rate_constants_R2=C, trimmed_R2=T, n_outliers=N, slope=S, test_R2=A, test_pearson=B, final_loss=E, alpha=A, avg_slope_ratio=R, MLP_node_slope_0=X, MLP_node_gt_slope_0=Y, MLP_node_slope_1=X, MLP_node_gt_slope_1=Y, embedding_cluster_acc=A, embedding_n_clusters=N, embedding_silhouette=S
Visual: MLP_sub=[description], MLP_node=[description], Embedding=[clustered/mixed], k_scatter=[description]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Next: parent=P
```

**CRITICAL**: The `Visual:` line must describe what you see in the last MLP_sub, MLP_node, rate_constants, and embedding plots. Always read the last (highest iteration) plots before writing this line.

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder. Always include the exact parameter change.

**CRITICAL**: The `Next: parent=P` line selects the parent for the **next batch's** mutations. `P` must refer to a node from a **previous** batch or the current batch -- but NEVER set `Next: parent=P` where P is `id+1` (circular reference).

Write all 4 entries before editing the 4 config files for the next batch.

## Block Boundaries

- At block boundaries, simulation stays the same (fixed-regime exploration)
- Training parameters can differ across the 4 slots
- Choose a new **parameter subspace** to explore in the next block

## Failed Slots

If a slot is marked `[FAILED]` in the prompt:
- Write a brief `## Iter N: failed` entry noting the failure
- Still propose a mutation for that slot's config in the next batch
- Do not draw conclusions from a single failure (may be stochastic)
