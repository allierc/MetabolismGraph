# Phase 2: Homeostasis Recovery — MLP_node and Embedding Training

## Regime Note

This config runs **Phase 2 only**. All reaction parameters (rate constants k, MLP_sub, stoichiometric matrix S) are **frozen** from a well-trained Phase 1 checkpoint (rate_constants_R2 ~0.87). Phase 2 trains only:

- **MLP_node** (`node_func`): learns per-type homeostatic regulation `-lambda_t * (c_i - c_baseline)`
- **Embeddings a_i**: per-metabolite embeddings that should cluster by metabolite type

The Phase 1 model has already captured the reaction dynamics. Phase 2 isolates the homeostatic signal and embedding structure, which Phase 1 could not resolve because the homeostatic contribution is ~1000x weaker than reaction terms.

## Goal

Find training hyperparameters and code strategies that recover **homeostatic regulation slopes** and **metabolite type embeddings** from Phase 1 residuals.

**Primary optimization target**: `phase2_score` — a composite metric defined as:

```
slope_accuracy = 1 - mean(|learned_slope_t - gt_slope_t| / |gt_slope_t|)   clamped to [0, 1]
phase2_score = 0.5 * slope_accuracy + 0.4 * embedding_cluster_acc + 0.1 * (rate_constants_R2 > 0.5)
```

**Three sub-goals:**

1. **MLP_node_slope_0 -> -0.001** (ground truth): Type 0 metabolites should exhibit homeostatic decay with slope -0.001
2. **MLP_node_slope_1 -> -0.002** (ground truth): Type 1 metabolites should exhibit homeostatic decay with slope -0.002
3. **embedding_cluster_acc -> 1.0**: Learned embeddings a_i should cleanly separate the two metabolite types into distinct clusters

## The Challenge

Phase 2 is fundamentally harder than Phase 1 because the homeostatic signal is buried under dominant reaction dynamics.

**Signal magnitude mismatch.** Homeostatic slopes are 0.001-0.002, producing per-step concentration changes of ~0.001-0.006. Reaction rates are 0.01-0.1, producing per-step changes of ~0.01-0.1. The homeostatic signal is 10-1000x weaker than the reaction signal in the dc/dt loss.

**Zero initialization trap.** MLP_node is initialized to output zero. At initialization, the gradient of the loss with respect to MLP_node parameters is approximately zero because the output contributes negligibly to the total dc/dt. Standard learning rates cannot escape this near-zero-gradient region.

**Single-step invisibility.** With single-step training (time_step=1), the homeostatic contribution to dc/dt is ~0.001 against a total dc/dt of ~0.01-0.1. The loss gradient attributes almost all error to the reaction components, not to MLP_node. Homeostasis is invisible in the gradient signal.

**Recurrent rollout trade-off.** Multi-step rollout (time_step=16, 32, 64) allows the homeostatic signal to accumulate — concentration drift from incorrect homeostasis compounds over steps. But longer rollouts also amplify gradient noise and can destabilize training.

**Embedding-homeostasis coupling.** Embeddings a_i only separate by type if MLP_node uses them differentially — i.e., MLP_node must produce different outputs for different embedding values. If MLP_node stays flat at zero, there is no gradient signal to push embeddings apart.

## Training Parameters Reference

All parameters below are in the `training:` section of the YAML config. **Simulation parameters and all Phase 1 parameters are FROZEN — do not change them.**

### Phase 2 Learning Rates

| Parameter | Config key | Description | Typical range |
|-----------|-----------|-------------|---------------|
| `lr_node_homeo` | `learning_rate_node_homeostasis` | Learning rate for MLP_node parameters during Phase 2 | 1E-4 to 1E-1 |
| `lr_emb_homeo` | `learning_rate_embedding_homeostasis` | Learning rate for per-metabolite embeddings a_i | 1E-4 to 1E-1 |

**Key insight**: Both learning rates may need to be significantly higher than Phase 1 equivalents. MLP_node starts at zero with near-zero gradients — standard LRs (1E-3) may be insufficient to escape the flat region. LRs of 1E-2 to 1E-1 are worth exploring.

**Key insight**: The two learning rates interact. If `lr_node_homeo` is too low, MLP_node stays flat, embeddings receive no gradient, and `lr_emb_homeo` is irrelevant. MLP_node must move first before embeddings can separate.

### Phase 2 Training Schedule

| Parameter | Config key | Description | Typical range |
|-----------|-----------|-------------|---------------|
| `homeostasis_time_step` | `homeostasis_time_step` | Number of Euler rollout steps for Phase 2 training. **FIXED per slot** — do not change | 4, 16, 32, 64 |
| `data_augmentation_loop` | `data_augmentation_loop` | Training iterations multiplier. Controls total gradient steps | 100 to 5000 |
| `batch_size` | `batch_size` | Batch size per gradient step | 4 to 32 |

**Key insight**: `homeostasis_time_step` is assigned per slot and must not be changed. Different slots explore different rollout lengths (4, 16, 32, 64). Longer rollouts accumulate more homeostatic signal but cost proportionally more compute and produce noisier gradients.

**Key insight**: `data_augmentation_loop` may need to be much larger than Phase 1 (up to 5000) because the weak homeostatic signal requires more gradient steps to move MLP_node parameters meaningfully.

**DO NOT change `homeostasis_time_step`** — it is fixed per slot (4/16/32/64).

### Frozen Parameters (DO NOT CHANGE)

| Parameter | Config key | Value | Reason |
|-----------|-----------|-------|--------|
| `freeze_stoichiometry` | `freeze_stoichiometry` | true | S is frozen from Phase 1 |
| `skip_phase1` | `skip_phase1` | true | Phase 1 training is skipped |
| `homeostasis_training` | `homeostasis_training` | true | Enables Phase 2 training loop |
| All Phase 1 LRs | `learning_rate_k`, `learning_rate_sub`, etc. | N/A | Phase 1 parameters are frozen |
| All simulation params | `n_metabolites`, `n_reactions`, etc. | N/A | Fixed simulation configuration |

## Training Metrics

The following metrics are written to `analysis.log` at the end of training:

| Metric | Description | Good value |
|--------|-------------|------------|
| `phase2_loss` | Phase 2 training loss (MSE on concentration trajectory) | Lower is better |
| `phase2_node_magnitude` | Mean absolute MLP_node output across all metabolites | > 0 (non-zero means learning) |
| `MLP_node_slope_0` | Learned slope of MLP_node for type 0 metabolites (linear fit over concentration sweep) | ~ -0.001 |
| `MLP_node_slope_1` | Learned slope of MLP_node for type 1 metabolites (linear fit over concentration sweep) | ~ -0.002 |
| `MLP_node_gt_slope_0` | Ground-truth slope for type 0 metabolites | -0.001 (reference) |
| `MLP_node_gt_slope_1` | Ground-truth slope for type 1 metabolites | -0.002 (reference) |
| `embedding_cluster_acc` | DBSCAN cluster accuracy of learned embeddings vs GT metabolite types (Hungarian optimal mapping) | 1.0 |
| `embedding_n_clusters` | Number of clusters found by DBSCAN in embedding space | 2 |
| `embedding_silhouette` | Silhouette score of embedding clustering (cluster tightness and separation) | > 0.5 |
| `rate_constants_R2` | Phase 1 sanity check — R-squared of learned vs true rate constants | > 0.5 (must stay high) |
| `phase2_score` | Composite metric: 0.5 * slope_accuracy + 0.4 * cluster_acc + 0.1 * (R2 > 0.5) | > 0.7 |

### Interpretation

- **phase2_node_magnitude > 0 but slopes ~ 0**: MLP_node is producing non-zero output but has not learned the correct linear shape. It may have learned a constant offset or random noise rather than the concentration-dependent regulation. Try increasing `data_augmentation_loop` or changing the training strategy.

- **Slopes moving toward GT values**: Correct direction. The homeostatic signal is being captured. Increase training duration (`data_augmentation_loop`) or slightly increase `lr_node_homeo` to accelerate convergence.

- **embedding_cluster_acc improving**: Embeddings are separating by type, which means MLP_node is using the embedding input differentially. This is a positive sign — MLP_node and embeddings are co-adapting.

- **rate_constants_R2 dropped significantly (below 0.5)**: PROBLEM. Phase 1 parameters have been corrupted. Check that all Phase 1 parameters are properly frozen (`skip_phase1=true`, `freeze_stoichiometry=true`). This should not happen if freezing is correct.

- **phase2_score stuck near 0.1**: Only the R2 > 0.5 baseline term is contributing. Both slopes and embeddings are failing. Consider a code-level strategy change (see Literature-Informed Strategies).

## Literature-Informed Strategies

These are the strategies to explore by modifying Phase 2 training code between blocks. They address the core difficulty: the homeostatic signal is ~1000x weaker than reaction dynamics.

### Strategy 1: Residual-Based Direct Supervision (HIGHEST PRIORITY)

Instead of relying on end-to-end rollout loss (where homeostasis is buried), compute the homeostatic residual directly and supervise MLP_node against it:

```python
# Compute true dc/dt from data
true_dcdt = (x_list[k+1][:, 3] - x_list[k][:, 3]) / delta_t

# Compute reaction-only prediction (S * k * MLP_sub, without MLP_node)
reaction_pred = model_forward_reaction_only(x)

# The residual is what homeostasis must explain
residual = true_dcdt - reaction_pred

# Direct supervision for MLP_node
loss = MSE(model.node_func(node_in), residual)
```

This provides **direct supervision** for MLP_node, bypassing the gradient attenuation problem entirely. The residual isolates the homeostatic component, giving MLP_node a clear target.

**References**: Physics-Informed Neural Networks residual decomposition (Raissi et al. 2019), SINDy operator splitting (Brunton et al. 2016), additive model decomposition.

### Strategy 2: Signal Amplification

Temporarily multiply MLP_node output by a large amplification factor (10-100x) during loss computation:

```python
node_output = model.node_func(node_in)
amplified_output = node_output * amplification_factor  # e.g., 50x
# Use amplified_output in dc/dt computation for loss
```

This makes the homeostatic contribution comparable in magnitude to reaction terms, producing meaningful gradients. After MLP_node learns the correct functional shape, reduce the amplification factor gradually toward 1x.

**References**: Curriculum learning (Bengio et al. 2009), loss scaling in mixed-precision training.

### Strategy 3: Auxiliary Contrastive Loss for Embeddings

Add an auxiliary loss that directly pushes embeddings apart based on metabolite type information from MLP_node output patterns:

```python
# Metabolites with similar MLP_node response should have similar embeddings
# Metabolites with different MLP_node response should have different embeddings
loss_contrastive = contrastive_loss(embeddings, groups_from_node_output)
loss_total = loss_main + lambda_contrastive * loss_contrastive
```

This decouples embedding learning from the weak homeostatic gradient, providing a direct signal to separate embeddings by functional type.

**References**: Contrastive learning (Chen et al. 2020), triplet loss (Schroff et al. 2015).

### Strategy 4: Curriculum on Time Steps

Within a single training run, start with the longest available rollout (strongest accumulated homeostatic signal), then gradually reduce:

- Epochs 1-2: Use full rollout length (maximum signal accumulation)
- Epochs 3-4: Reduce to half the rollout length (refine with cleaner gradients)

Alternatively, weight the loss from different rollout positions differently — emphasize later time steps where homeostatic drift is largest.

**References**: Curriculum learning (Bengio et al. 2009), scheduled sampling (Bengio et al. 2015).

### Strategy 5: Higher Learning Rates + Warmup

MLP_node is at zero. Standard learning rates may be insufficient to escape the near-zero-gradient region. Try aggressive warmup schedules:

- **LR warmup**: Start at 0.01 or 0.1, decay to 0.001 over the first 10% of training
- **Cosine annealing**: High LR -> low LR -> restart cycle
- **Linear warmup + decay**: Ramp up LR over first 500 steps, then decay

The key is to provide a large initial kick to move MLP_node parameters away from zero initialization.

**References**: Warmup schedules (Goyal et al. 2017), cosine annealing (Loshchilov & Hutter 2017).

### Strategy 6: Multi-Scale Rollout

Sample different rollout lengths within each batch rather than using a fixed length:

```python
# Mix rollout lengths within each batch
rollout_lengths = [4, 16, 32, 64]
for batch in batches:
    length = random.choice(rollout_lengths)
    loss += compute_loss(batch, rollout_length=length)
```

Short rollouts provide clean gradients for local dynamics. Long rollouts provide signal for homeostasis. The mixture captures both.

**References**: Multi-scale training (Lin et al. 2017), stochastic depth (Huang et al. 2016).

### Strategy 7: Gradient Accumulation

Accumulate gradients over 4-8 mini-batches before each optimizer step:

```python
for i, batch in enumerate(batches):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Weak signals benefit from variance reduction. Larger effective batch size averages out noise, making the small homeostatic gradient more reliable.

**References**: Gradient accumulation for large-batch training (Goyal et al. 2017).

### Strategy 8: Separate Loss Components

Decompose the loss into reaction error and homeostasis error, and only backpropagate the homeostasis error through MLP_node:

```python
dcdt_reaction = S @ (k * mlp_sub(c))     # reaction component
dcdt_homeo = mlp_node(node_in)            # homeostatic component
dcdt_total = dcdt_reaction + dcdt_homeo

loss_reaction = MSE(dcdt_reaction, true_dcdt)  # for monitoring only
loss_homeo = MSE(dcdt_total, true_dcdt)        # backprop through MLP_node only

# Detach reaction gradients so MLP_node gets the full error signal
loss_homeo.backward()
```

This prevents the optimizer from attributing the entire loss reduction to reaction parameters (which are frozen anyway in Phase 2, but the gradient flow through the computation graph still matters for MLP_node).

**References**: Gradient isolation, stop-gradient techniques (Chen & He 2021).

## Iteration Workflow

### Step 1: Read Working Memory

Read `{config}_memory.md` to recall:
- Established principles about Phase 2 learning rate interactions
- Which strategies have been attempted and their outcomes
- Current block progress and best-performing configurations

### Step 2: Analyze Current Results

**Metrics from `analysis.log`:**
- `phase2_score`: Primary composite metric
- `MLP_node_slope_0` vs `MLP_node_gt_slope_0`: How close is type 0 homeostasis?
- `MLP_node_slope_1` vs `MLP_node_gt_slope_1`: How close is type 1 homeostasis?
- `embedding_cluster_acc`: Are embeddings separating?
- `phase2_node_magnitude`: Is MLP_node producing any output at all?
- `rate_constants_R2`: Sanity check — Phase 1 parameters intact?

**Classification:**

| phase2_score | Classification |
|:------------:|:--------------:|
| > 0.7 | converged |
| 0.3 - 0.7 | partial |
| < 0.3 | failed |

**Visual Analysis:**

Examine the **last** (highest iteration number) plot in each folder under `{log_dir}/tmp_training/`:
- `function/rate_func/MLP_node_p2_*.png`: Per-type homeostasis function. Should show near-linear curves with small negative slopes (-0.001 for type 0, -0.002 for type 1). Bad: flat lines at zero, large constant offsets, or non-linear shapes.
- `rate_constants/comparison_p2_*.png`: Rate constants scatter plot (should be unchanged from Phase 1). Any degradation indicates broken parameter freezing.
- Embedding visualization (if available): Should show two separated clusters.

**IMPORTANT**: Always read the last plot file in `function/rate_func/` and `rate_constants/` to visually assess Phase 2 progress. Sort by filename to find the latest iteration. Include your visual assessment in the log.

**UCB scores from `ucb_scores.txt`:**
- Provides computed UCB scores for all exploration nodes
- At block boundaries, the UCB file will be empty — use `parent=root`

### Step 3: Write Log Entry

Append to Full Log (`{config}_analysis.md`):

```
## Iter N: [converged/partial/failed]
Node: id=N, parent=P
Mode/Strategy: [exploit/explore/strategy_name]
Slot: slot_S (time_step=T)
Config: lr_node_homeo=X, lr_emb_homeo=Y, data_augmentation_loop=A, batch_size=B
Metrics: phase2_score=S, slope_accuracy=SA, MLP_node_slope_0=X, MLP_node_gt_slope_0=Y, MLP_node_slope_1=X, MLP_node_gt_slope_1=Y, embedding_cluster_acc=A, embedding_n_clusters=N, rate_constants_R2=R
Visual: MLP_node=[active/inactive: slope comparison], Embeddings=[separated/clustered/collapsed]
Mutation: [param]: [old] -> [new]
Strategy: [which literature strategy was attempted]
Observation: [one line]
Next: parent=P
```

**CRITICAL**: The `Visual:` line must describe what you see in the last MLP_node and embedding plots. Example: `Visual: MLP_node=inactive: slopes 0.000 vs GT -0.001/-0.002, flat at zero, Embeddings=collapsed: single cluster, acc=0.52`

**CRITICAL**: The `Mutation:` line is parsed by the UCB tree builder. Always include the exact parameter change.

**CRITICAL**: The `Strategy:` line must reference which literature-informed strategy (1-8) was used, or "hyperparameter-only" if only config params were changed.

**CRITICAL**: The `Next: parent=P` line selects the parent for the next iteration.

### Step 4: Parent Selection (UCB)

1. Read `ucb_scores.txt`
2. If empty -> `parent=root`
3. Otherwise -> select node with **highest UCB** as parent

### Step 5: Propose Next Mutation

**Strategy selection:**

| Condition | Strategy | Action |
|-----------|----------|--------|
| Default | **exploit** | Highest UCB node, conservative parameter mutation |
| phase2_score stuck < 0.3 for 3+ iters | **strategy-switch** | Try a different literature strategy (code change) |
| Slopes moving but embeddings stuck | **embedding-focus** | Increase lr_emb_homeo, add contrastive loss |
| Embeddings separated but slopes wrong | **slope-focus** | Increase lr_node_homeo, try residual supervision |
| phase2_score > 0.5 for 3+ iters | **exploit** | Fine-tune LRs and training duration |
| All failed for 4+ iters | **explore** | Try a fundamentally different approach (code change) |

**Parameter exploration order:**

1. **lr_node_homeo first**: This has the largest impact on whether MLP_node escapes the zero initialization
2. **lr_emb_homeo second**: Only matters once MLP_node is producing non-zero, type-dependent output
3. **data_augmentation_loop**: Increase if slopes are moving in the right direction but haven't converged
4. **Code strategy**: If hyperparameter tuning alone fails after 3+ iterations, switch to a code-level strategy

**Typical interactions:**

- **lr_node_homeo >> lr_emb_homeo**: MLP_node must learn first; embeddings follow
- **data_augmentation_loop up**: Weak signal needs more gradient steps
- **batch_size down**: Smaller batches = more updates per epoch, may help with weak signals
- **batch_size up**: Larger batches = less noise, may help stabilize weak gradient signal

### Step 6: Edit Config

Edit the config YAML file with the proposed mutation.

**DO NOT change simulation parameters** — this is a fixed-regime exploration.
**DO NOT change `homeostasis_time_step`** — it is fixed per slot.

## Code Modification Rules (Between Blocks Only)

When the prompt includes `>>> BLOCK END + CODE REVIEW <<<`:

- You may modify the Phase 2 training code in `src/MetabolismGraph/models/graph_trainer.py`
- Only modify code between `# ===== Phase 2: Homeostasis training (recurrent) =====` and the next section marker `# --- final analysis`
- **DO NOT** modify Phase 1 code, model architecture, or data generation
- **DO NOT** change the function signature or checkpoint saving format
- **Ensure** `analysis.log` still receives all metrics (`phase2_loss`, `phase2_node_magnitude`, `MLP_node_slope_0`, `MLP_node_slope_1`, `embedding_cluster_acc`, `phase2_score`, etc.)
- **Explain** your code changes with literature rationale (reference the strategy number)
- Changes will be synced to the cluster automatically

**Allowed modifications:**
- Loss function computation within Phase 2 training loop
- Optimizer configuration (LR schedules, gradient accumulation)
- Auxiliary losses (contrastive, residual supervision)
- Signal amplification or scaling of MLP_node output
- Training loop structure (curriculum, multi-scale rollout)

**Forbidden modifications:**
- Model class definition (MLP architecture, forward pass signature)
- Phase 1 training code
- Data generation or loading code
- Checkpoint format or saving logic
- Metric computation in the analysis section (must remain comparable across runs)

## Visual Analysis

Examine plots in `{log_dir}/tmp_training/`:

- **`function/rate_func/MLP_node_p2_*.png`**: Per-type homeostasis function. X-axis is concentration, Y-axis is MLP_node output. Should show near-linear curves with small negative slopes. Type 0 slope should be ~ -0.001, type 1 slope should be ~ -0.002. Dashed lines show ground truth. Good: learned curves overlap dashed GT. Bad: flat lines at zero, large offsets, or non-linear shapes.

- **`rate_constants/comparison_p2_*.png`**: Scatter plot of learned vs true log10(k_j). Should be identical to Phase 1 results (parameters are frozen). Any change indicates broken parameter freezing.

- **Embedding visualization** (if available): Should show two distinct clusters corresponding to the two metabolite types. Good: tight, well-separated clusters. Bad: single blob or random scatter.

## Loss Figure

The training progress is visualized in `{log_dir}/tmp_training/loss.tif`:

| Color | Component | Description |
|-------|-----------|-------------|
| **blue** (thick) | `phase2_loss` | Phase 2 prediction loss (MSE on concentration trajectory) |

**Monitor**: Blue curve should decrease steadily. If it plateaus immediately (no decrease), MLP_node is not learning — consider a code-level strategy change. If it decreases slowly, increase `data_augmentation_loop` or `lr_node_homeo`.

## File Structure

You maintain **TWO** files:

### 1. Full Log (append-only)

**File**: `{config}_analysis.md`
- Append every iteration's full log entry
- **Never read this file** — it is for human record only

### 2. Working Memory

**File**: `{config}_memory.md`
- **READ at start of each iteration**
- **UPDATE at end of each iteration**
- Contains: established principles + current block iterations + strategy outcomes
- Fixed size (~300 lines max)

## Parameter Sweep Guidelines

### Suggested initial spread (across slots)

- Slot 0 (time_step=4): baseline (lr_node_homeo=1E-2, lr_emb_homeo=1E-3, data_augmentation_loop=1000)
- Slot 1 (time_step=16): baseline (lr_node_homeo=1E-2, lr_emb_homeo=1E-3, data_augmentation_loop=1000)
- Slot 2 (time_step=32): baseline (lr_node_homeo=1E-2, lr_emb_homeo=1E-3, data_augmentation_loop=1000)
- Slot 3 (time_step=64): baseline (lr_node_homeo=1E-2, lr_emb_homeo=1E-3, data_augmentation_loop=1000)

### Key questions to resolve

- **Which rollout length works best?** Longer rollouts accumulate more homeostatic signal but have noisier gradients. Compare phase2_score across slots.
- **LR magnitude**: Does MLP_node need 1E-1 to escape zero, or is 1E-2 sufficient?
- **Training duration**: How many iterations before slopes converge? Is 1000 enough or do we need 5000?
- **Code strategy**: If hyperparameter-only exploration fails across all slots, which literature strategy (1-8) breaks the deadlock?
