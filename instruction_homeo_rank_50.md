# Metabolism Joint Recovery — S Given, Homeostasis Active (High-Rank Regime)

## Regime Note

This exploration extends the **simulation_oscillatory_rank_50** exploration by jointly training all four learnable components: rate constants **k**, metabolite **embeddings** a_i, **MLP_node** (homeostasis), and **MLP_sub** (substrate function). The previous exploration kept MLP_node small and embeddings passive; here, MLP_node must learn type-dependent homeostatic regulation and embeddings must self-organize to distinguish the two metabolite types.

The starting config comes from **iter_116** of the previous exploration, which achieved good rate constant recovery. The challenge is now to simultaneously maintain k recovery while learning correct homeostasis.

## Goal

Find GNN training hyperparameters that **jointly recover**:

1. **Rate constants k_j** (256 learnable log-scale parameters) — primary metric: `rate_constants_R2`
2. **MLP_node homeostasis slopes** — per-type slopes matching ground truth: `MLP_node_slope_t` vs `MLP_node_gt_slope_t`
3. **Embedding separation** — metabolite types cluster in embedding space: `embedding_cluster_acc`
4. **MLP_sub substrate function** — correct c^s power law: visual assessment

The **stoichiometric matrix S is frozen** (given from ground truth).

**Primary optimization target**: `rate_constants_R2` (must not regress from ~0.9 baseline).
**Secondary targets**: `avg_slope_ratio` near 1.0, `embedding_cluster_acc` = 1.0.

## Metabolism Regime Characteristics

The system models a metabolic network as a bipartite graph (metabolites <-> reactions):

```
dc_i/dt = homeostasis_i(c_i, a_i) + sum_j S_ij * v_j(c)
```

- **c** = metabolite concentrations (100 metabolites, 2 types)
- **a_i** = per-metabolite embedding (2D, learnable) — input to MLP_node
- **S** = stoichiometric matrix (100 x 256), **FROZEN** from ground truth
- **v_j(c)** = reaction rate: `v_j = k_j * prod_i(substrate_func(c_i, |S_ij|))` for substrates
- **k_j** = per-reaction rate constant (learnable), true values log-uniform in [-2.0, -1.0]
- **homeostasis_i** = per-metabolite homeostatic term, learned by MLP_node(c_i, a_i)
  - True: `-lambda_type(i) * (c_i - c_baseline_type(i))`
  - Type 0: lambda=0.001, baseline=4.0
  - Type 1: lambda=0.002, baseline=6.0

### The Joint Learning Challenge

Unlike the previous exploration where MLP_node was kept small with L1 regularization:

- **MLP_node must now learn correct type-dependent slopes** — not just stay near zero
- **Embeddings must self-organize** into two clusters (one per metabolite type) so MLP_node can produce different outputs per type
- **Chicken-and-egg problem**: MLP_node needs separated embeddings to learn type-dependent slopes; embeddings need type-dependent MLP_node output to receive gradient signal for separation
- **Scale competition**: Homeostatic signal (~0.001-0.002) is ~1000x weaker than reaction dynamics. MLP_node gradients are tiny compared to k and MLP_sub gradients

### Key Differences from Previous Exploration (simulation_oscillatory_rank_50)

| Aspect | Previous (k-focused) | This (joint recovery) |
|--------|----------------------|----------------------|
| MLP_node role | Kept small via L1 | Must learn correct slopes |
| Embedding role | Passive | Must cluster by type |
| Learning rates | lr_k, lr_node, lr_sub | lr_k, lr_embedding, lr_node, lr_sub |
| Success criteria | rate_constants_R2 > 0.9 | R2 > 0.9 AND slopes correct AND embeddings clustered |
| coeff_MLP_node_L1 | High (suppress) | May need reduction to let MLP_node grow |

## Training Parameters Reference

All parameters below are in the `training:` section of the YAML config. **Simulation parameters are FROZEN -- do not change them.**

### Learning Rates

| Parameter | Config key | Description | Typical range |
|-----------|-----------|-------------|---------------|
| `lr_k` | `learning_rate_k` | Learning rate for rate constants (log_k) | 1E-4 to 1E-2 |
| `lr_embedding` | `learning_rate_embedding` | Learning rate for metabolite embeddings a_i | 1E-5 to 1E-3 |
| `lr_node` | `learning_rate_node` | Learning rate for MLP_node (homeostasis function) | 1E-5 to 1E-3 |
| `lr_sub` | `learning_rate_sub` | Learning rate for MLP_sub (substrate function) | 1E-4 to 1E-2 |
| `lr` | `learning_rate_start` | Fallback learning rate for other parameters | 1E-4 to 1E-2 |

**Key insight**: Four learning rates now interact. The hierarchy matters:
- **lr_k and lr_sub** drive reaction dynamics learning (fast, strong gradients)
- **lr_node and lr_embedding** drive homeostasis learning (slow, weak gradients)
- If lr_node/lr_embedding are too high, homeostasis interferes with k recovery
- If too low, MLP_node stays at zero and embeddings never separate
- **lr_embedding should generally be <= lr_node** — MLP_node must learn a useful function before embedding gradients become meaningful

### Training Schedule

| Parameter | Config key | Description | Typical range |
|-----------|-----------|-------------|---------------|
| `n_epochs` | `n_epochs` | Number of training epochs | 1 to 10 |
| `batch_size` | `batch_size` | Number of time frames per gradient step | 4 to 32 |
| `data_augmentation_loop` | `data_augmentation_loop` | Multiplier for iterations per epoch | 100 to 5000 |
| `seed` | `seed` | Random seed for training reproducibility | any integer |

**Key insight**: Joint learning may need longer training (`data_augmentation_loop` up to 5000) because the homeostatic signal is weak and needs time to accumulate. The `claude.data_augmentation_loop` controls the per-iteration training budget.

### MLP Regularization

| Parameter | Config key | Description | Typical range |
|-----------|-----------|-------------|---------------|
| `coeff_MLP_sub_diff` | `coeff_MLP_sub_diff` | Monotonicity constraint for MLP_sub | 0 to 500 |
| `coeff_MLP_node_L1` | `coeff_MLP_node_L1` | L1 penalty on MLP_node output magnitude | 0 to 10 |
| `coeff_MLP_sub_norm` | `coeff_MLP_sub_norm` | Penalizes substrate_func(c=1, \|s\|=1) deviating from 1 | 0 to 10 |
| `coeff_k_floor` | `coeff_k_floor` | Penalizes log_k below threshold | 0 to 10 |
| `k_floor_threshold` | `k_floor_threshold` | Threshold for k_floor penalty | -3.0 to -1.0 |

**Key insight for joint learning**: `coeff_MLP_node_L1` creates a tension -- too high suppresses homeostasis learning, too low lets MLP_node absorb reaction dynamics. Start with the iter_116 value (1.0) and adjust based on whether MLP_node slopes are too small or too large.

**Key insight**: `coeff_MLP_sub_norm` pins MLP_sub scale to prevent k from absorbing a global factor. Essential for joint learning where four components can trade scale factors.

### Recurrent Rollout Training

| Parameter | Config key | Description | Typical range |
|-----------|-----------|-------------|---------------|
| `time_step` | `time_step` | Number of Euler integration steps per training sample | 1 to 16 |
| `recurrent_training` | `recurrent_training` | Enable multi-step rollout training | true/false |

**Key insight for homeostasis**: Multi-step rollout amplifies the homeostatic signal. At `time_step=1`, the homeostatic contribution to dc/dt is ~0.001 (tiny). Over N steps, it accumulates to ~0.001*N, making it more visible to the optimizer. Consider `time_step=4` or higher to help MLP_node learning.

### MLP Architecture

These parameters are in the `graph_model:` section.

| Parameter | Config key | Description | Typical range |
|-----------|-----------|-------------|---------------|
| `hidden_dim_sub` | `hidden_dim_sub` | Hidden width of MLP_sub | 16 to 128 |
| `n_layers_sub` | `n_layers_sub` | Depth of MLP_sub | 2 to 5 |
| `hidden_dim_node` | `hidden_dim_node` | Hidden width of MLP_node | 16 to 64 |
| `n_layers_node` | `n_layers_node` | Depth of MLP_node | 2 to 4 |

**Key insight**: MLP_node needs at least `n_layers_node >= 2` to produce embedding-dependent slopes (with Tanh nonlinearity creating c*a interaction). A single linear layer cannot learn type-dependent slopes.

### Frozen Parameters (DO NOT CHANGE)

| Parameter | Config key | Value | Reason |
|-----------|-----------|-------|--------|
| `freeze_stoichiometry` | `freeze_stoichiometry` | true | S is given from GT |
| `lr_S` | `learning_rate_S_start` | 0.0 | S is frozen |
| `coeff_S_L1` | `coeff_S_L1` | 0.0 | No S regularization |
| `coeff_S_integer` | `coeff_S_integer` | 0.0 | No S regularization |
| `coeff_mass` | `coeff_mass_conservation` | 0.0 | No S regularization |

## Training Metrics

The following metrics are written to `analysis.log` at the end of training:

| Metric | Description | Good value |
|--------|-------------|------------|
| `rate_constants_R2` | R2 between learned and true rate constants (after MLP_sub scalar correction) | > 0.9 |
| `trimmed_R2` | R2 excluding outlier reactions (\|delta_log_k\| > 0.3) | > 0.9 |
| `n_outliers` | Outlier reactions (\|delta_log_k\| > 0.3) | < 10% of 256 |
| `slope` | Slope of learned vs true log_k linear fit | ~1.0 |
| `test_R2` | R2 on held-out test frames | > 0.9 |
| `test_pearson` | Pearson correlation on test frames | > 0.95 |
| `alpha` | MLP_sub scale factor (ideal: 1.0) | ~1.0 |
| `MLP_node_slope_t` | Learned MLP_node slope for type t | ~ GT slope |
| `MLP_node_gt_slope_t` | Ground-truth slope -lambda_t | Reference |
| `avg_slope_ratio` | Mean of (learned_slope / gt_slope) across types | ~1.0 |
| `embedding_cluster_acc` | Cluster accuracy of embeddings vs GT types | 1.0 |
| `embedding_n_clusters` | Number of clusters found by DBSCAN | 2 |
| `embedding_silhouette` | Silhouette score of embedding clustering | > 0.5 |

### Interpretation

**Joint success requires ALL THREE:**
- **rate_constants_R2 > 0.9**: k values correctly recovered
- **avg_slope_ratio in [0.5, 2.0]**: MLP_node learned correct homeostatic slopes (slope_ratio near 1.0 is ideal)
- **embedding_cluster_acc = 1.0**: Embeddings separate metabolite types

**Classification:**
- **Converged**: rate_constants_R2 > 0.9 AND avg_slope_ratio in [0.5, 2.0] AND embedding_cluster_acc = 1.0
- **Partial-k**: rate_constants_R2 > 0.9 but slopes or embeddings wrong
- **Partial-homeo**: slopes improving but rate_constants_R2 < 0.9
- **Failed**: rate_constants_R2 < 0.5 or avg_slope_ratio < 0.1

**Degeneracy Detection:**

Compute the **degeneracy gap** = `test_pearson - rate_constants_R2`:

| test_pearson | rate_constants_R2 | Gap | Diagnosis |
|:------------:|:-----------------:|:---:|-----------|
| > 0.95 | > 0.9 | < 0.1 | **Healthy** -- good dynamics from correct k |
| > 0.95 | 0.3-0.9 | 0.1-0.7 | **Degenerate** -- MLP compensation |
| > 0.95 | < 0.3 | > 0.7 | **Severely degenerate** |
| < 0.5 | < 0.5 | ~0 | **Failed** |

**Homeostasis Assessment:**

| avg_slope_ratio | Diagnosis |
|:---:|-----------|
| 0.8-1.2 | Correct homeostasis |
| 0.3-0.8 or 1.2-3.0 | Partial -- direction correct, magnitude off |
| > 3.0 or < 0.3 | Wrong -- MLP_node overshoot or undershoot |
| ~0 | Inactive -- MLP_node stuck at zero |

## Visual Analysis

Examine plots in `{log_dir}/tmp_training/`:

- **`function/substrate_func/MLP_sub_*.png`**: Should show c^1 and c^2 curves matching ground truth (dashed lines). Good: learned curves overlap dashed GT. Bad: curves diverge or wrong shape.
- **`function/rate_func/MLP_node_*.png`**: Per-type homeostasis function. X-axis is concentration, Y-axis is MLP_node output. Should show near-linear curves with small negative slopes. Type 0 slope should be ~ -0.001, type 1 slope should be ~ -0.002. Dashed lines show ground truth. Good: learned curves overlap dashed GT. Bad: flat lines at zero, large offsets, or non-linear shapes.
- **`rate_constants/comparison_*.png`**: Scatter plot of learned vs true log10(k_j). Good: points along diagonal. Bad: cloud or offset.
- **`embedding/embedding_*.png`**: 2D scatter of metabolite embeddings colored by type. Good: two distinct clusters. Bad: single blob or random scatter.

**IMPORTANT**: Always read the last plot file in each folder to visually assess all four components. Include your visual assessment in the log.

## Iteration Workflow

### Step 1: Read Working Memory

Read `{config}_memory.md` to recall:
- Established principles about lr_k, lr_embedding, lr_node, lr_sub interactions
- Previous iteration findings and the current frontier
- Current block progress

### Step 2: Analyze Current Results

**Metrics from `analysis.log`:**
- `rate_constants_R2`: Primary metric (must not regress)
- `avg_slope_ratio`: Homeostasis quality (target: 1.0)
- `embedding_cluster_acc`: Embedding separation (target: 1.0)
- `test_pearson`: Dynamics prediction
- `alpha`: MLP_sub scale factor (ideal: 1.0)

**UCB scores from `ucb_scores.txt`:**
- At block boundaries, UCB file will be empty -- use `parent=root`

### Step 3: Write Outputs

Append to Full Log (`{config}_analysis.md`) and Working Memory (`{config}_memory.md`):

**Log Form:**

```
## Iter N: [converged/partial-k/partial-homeo/failed]
Node: id=N, parent=P
Mode/Strategy: [exploit/explore/boundary]
Config: seed=S, lr_k=X, lr_embedding=E, lr_node=Y, lr_sub=Z, batch_size=B, n_epochs=E, data_augmentation_loop=A, coeff_MLP_node_L1=L, coeff_MLP_sub_norm=N, coeff_k_floor=K, time_step=T
Metrics: rate_constants_R2=C, trimmed_R2=T, n_outliers=N, slope=S, test_R2=A, test_pearson=B, final_loss=E, alpha=A, avg_slope_ratio=R, MLP_node_slope_0=X, MLP_node_gt_slope_0=Y, MLP_node_slope_1=X, MLP_node_gt_slope_1=Y, embedding_cluster_acc=A, embedding_n_clusters=N, embedding_silhouette=S
Visual: MLP_sub=[good/partial/bad: description], MLP_node=[description], Embedding=[clustered/mixed: description], k_scatter=[description]
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Next: parent=P
```

**CRITICAL**: The `Visual:` line must describe what you see in the last MLP_sub, MLP_node, rate_constants, and embedding plots.

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder.

**CRITICAL**: The `Next: parent=P` line selects the parent for the next iteration.

### Step 4: Parent Selection (UCB)

1. Read `ucb_scores.txt`
2. If empty -> `parent=root`
3. Otherwise -> select node with **highest UCB** as parent

### Step 5: Propose Next Mutation

**Strategy selection:**

| Condition | Strategy | Action |
|-----------|----------|--------|
| Default | **exploit** | Highest UCB node, conservative mutation |
| rate_constants_R2 > 0.9 but slopes ~0 | **homeo-kick** | Increase lr_node, decrease coeff_MLP_node_L1 |
| Slopes moving but embeddings stuck | **embedding-focus** | Increase lr_embedding |
| All partial-k for 4+ iters | **k-protect** | Decrease lr_node/lr_embedding to protect k |
| Degeneracy gap > 0.3 | **degeneracy-break** | Enable recurrent_training, increase time_step |
| 3+ consecutive converged | **boundary-probe** | Extreme parameter to find boundary |

**Parameter exploration order:**

1. **lr_k first**: Establish good k recovery (baseline from iter_116)
2. **lr_node second**: Gradually increase to let homeostasis emerge
3. **lr_embedding third**: Only matters once MLP_node produces non-zero type-dependent output
4. **coeff_MLP_node_L1**: Decrease if slopes stuck at zero; increase if MLP_node too large
5. **time_step**: Try multi-step rollout to amplify homeostatic signal

**Typical interactions:**

- **lr_node up + coeff_MLP_node_L1 down**: Lets MLP_node grow, may destabilize k
- **lr_embedding up**: Speeds embedding separation IF MLP_node already type-dependent
- **time_step up + lr_k down**: Longer rollout amplifies all gradients
- **coeff_MLP_node_L1 too high**: Suppresses homeostasis completely (slopes = 0)
- **coeff_MLP_node_L1 = 0**: MLP_node may absorb reaction dynamics

### Step 6: Edit Config

Edit the config YAML file with the proposed mutation.

**DO NOT change simulation parameters** -- this is a fixed-regime exploration.

## File Structure

You maintain **TWO** files:

### 1. Full Log (append-only)

**File**: `{config}_analysis.md`
- Append every iteration's full log entry
- **Never read this file** -- it's for human record only

### 2. Working Memory

**File**: `{config}_memory.md`
- **READ at start of each iteration**
- **UPDATE at end of each iteration**
- Contains: established principles + current block iterations
- Fixed size (~300 lines max)

## Loss Figure

The training progress is visualized in `{log_dir}/tmp_training/loss.tif`:

| Color | Component | Description |
|-------|-----------|-------------|
| **blue** (thick) | `loss` | Prediction loss (MSE on dc/dt) |
| **cyan** | `regul_total` | Total regularization |

## Parameter Sweep Guidelines

### Suggested initial spread

- Slot 0: baseline (iter_116 settings: lr_k=5E-3, lr_embedding=1E-4, lr_node=1E-4, lr_sub=5E-3)
- Slot 1: higher lr_node (lr_node=5E-4) to kick homeostasis
- Slot 2: lower coeff_MLP_node_L1 (0.1 instead of 1.0) to let MLP_node grow
- Slot 3: multi-step rollout (recurrent_training=true, time_step=4, lr_k=2E-3)

### Key questions to resolve

- **Can k recovery be maintained while learning homeostasis?** If lr_node/lr_embedding destabilize k, the joint problem may need curriculum learning (Phase 1 k-only, then Phase 1+homeo)
- **What lr_node breaks through zero?** MLP_node is initialized to zero output -- needs enough LR to escape
- **Does time_step help homeostasis?** Multi-step rollout accumulates homeostatic signal
- **What coeff_MLP_node_L1 balances constraint vs freedom?** Too high = zero slopes, too low = MLP_node absorbs reaction dynamics
