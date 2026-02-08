# Experiment Log: simulation_oscillatory_rank_50 (parallel)

## Block 1: Initial Exploration

### Batch 1 (Iter 1-4): PARALLEL START - Initial Spread

Regime: S frozen (from GT), learning rate constants k
- 100 metabolites, 256 reactions, 2880 frames
- log_k range: [-2.0, -1.0] (activity rank ~50)
- Base config: lr_k=0.005, lr_node=0.001, lr_sub=0.0005

Planned initial variations:

**Slot 0 (id=1)**: Baseline
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=1000
- Mutation: None (baseline from previous oscillatory exploration)
- Parent rule: root (initial batch)

**Slot 1 (id=2)**: Explore lr_k high
- Config: seed=42, lr_k=0.01, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=1000
- Mutation: lr_k: 0.005 -> 0.01
- Parent rule: root (initial batch)

**Slot 2 (id=3)**: Explore lr_node high
- Config: seed=42, lr_k=0.005, lr_node=0.002, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=1000
- Mutation: lr_node: 0.001 -> 0.002
- Parent rule: root (initial batch)

**Slot 3 (id=4)**: Explore lr_sub low
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0002, batch_size=8, n_epochs=1, data_augmentation_loop=1000
- Mutation: lr_sub: 0.0005 -> 0.0002
- Parent rule: root (initial batch)

Rationale: Initial spread across the 3 key learning rate dimensions (lr_k, lr_node, lr_sub) to establish baseline performance and identify which dimension has most impact on rate_constants_R2 in the high-rank regime.

---

## Iter 1: failed
Node: id=1, parent=root
Mode/Strategy: exploit (baseline)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=1000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=0.0, coeff_k_floor=0.0
Metrics: rate_constants_R2=0.0443, trimmed_R2=0.9234, n_outliers=43, slope=0.9684, test_R2=-766043.33, test_pearson=-0.0125, final_loss=17381.72, alpha=0.5196
Visual: MLP_sub=bad: c^1 close to GT but c^2 deviates significantly (linear instead of quadratic), MLP_node=bad: flat lines at 0, no homeostasis learned
Mutation: None (baseline)
Parent rule: root (initial batch)
Observation: Training failed to converge — test_R2 massively negative indicates exploding predictions; MLP_node not learning at all
Next: parent=root

## Iter 2: failed
Node: id=2, parent=root
Mode/Strategy: explore (lr_k high)
Config: seed=42, lr_k=0.01, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=1000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=0.0, coeff_k_floor=0.0
Metrics: rate_constants_R2=0.0268, trimmed_R2=0.9133, n_outliers=47, slope=0.9544, test_R2=-73378.61, test_pearson=0.1167, final_loss=18537.14, alpha=0.4709
Visual: MLP_sub=bad: c^1 close to GT but c^2 deviates (linear), MLP_node=bad: flat lines at 0
Mutation: lr_k: 0.005 -> 0.01
Parent rule: root (initial batch)
Observation: Higher lr_k worsened rate_constants_R2; training still not converging; MLP_node dead
Next: parent=root

## Iter 3: failed
Node: id=3, parent=root
Mode/Strategy: explore (lr_node high)
Config: seed=42, lr_k=0.005, lr_node=0.002, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=1000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=0.0, coeff_k_floor=0.0
Metrics: rate_constants_R2=0.0442, trimmed_R2=0.9215, n_outliers=45, slope=0.9636, test_R2=-1922287.55, test_pearson=-0.0435, final_loss=17976.47, alpha=0.4664
Visual: MLP_sub=bad: c^1 close to GT but c^2 deviates (linear), MLP_node=bad: flat lines at 0
Mutation: lr_node: 0.001 -> 0.002
Parent rule: root (initial batch)
Observation: Higher lr_node did not help MLP_node learn; test_R2 even worse; MLP_node still dead despite higher learning rate
Next: parent=root

## Iter 4: failed
Node: id=4, parent=root
Mode/Strategy: explore (lr_sub low)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0002, batch_size=8, n_epochs=1, data_augmentation_loop=1000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=0.0, coeff_k_floor=0.0
Metrics: rate_constants_R2=0.0314, trimmed_R2=0.9164, n_outliers=53, slope=0.9680, test_R2=-487746.94, test_pearson=0.1415, final_loss=18170.46, alpha=0.4535
Visual: MLP_sub=bad: c^1 close to GT but c^2 deviates significantly (linear instead of quadratic), MLP_node=bad: flat lines at 0
Mutation: lr_sub: 0.0005 -> 0.0002
Parent rule: root (initial batch)
Observation: Lower lr_sub slightly worse; same fundamental issue — training not converging, MLP_node dead
Next: parent=root

---

### Batch 1 Summary

**Critical Finding**: All 4 configurations failed with rate_constants_R2 < 0.05. The fundamental issue is that training is not converging:
- test_R2 is massively negative (predictions explode)
- test_pearson near 0 (no correlation with GT dynamics)
- MLP_node is completely flat (not learning homeostasis at all)
- MLP_sub c^2 curve is nearly linear instead of quadratic

**Hypotheses for next batch:**
1. **coeff_MLP_node_L1 = 1.0 may be too strong** — suppressing MLP_node learning entirely
2. **Need coeff_MLP_sub_norm > 0** — alpha ~0.5 indicates MLP_sub scale is wrong; pinning c=1 scale may help
3. **Learning rates may be too high for this regime** — try lower lr_k (0.001 or 0.002)
4. **May need longer training** — data_augmentation_loop=1000 may be insufficient for convergence

**Next batch strategy:**
- Slot 0: Reduce coeff_MLP_node_L1 to 0.1 (let MLP_node learn)
- Slot 1: Enable coeff_MLP_sub_norm=1.0 (pin MLP_sub scale)
- Slot 2: Lower lr_k to 0.002 (more conservative k learning)
- Slot 3: Increase data_augmentation_loop to 2000 (longer training)

### Batch 2 (Iter 5-8): Addressing Training Failure

**Slot 0 (id=5)**: Reduce MLP_node L1 penalty
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=1000, coeff_MLP_node_L1=0.1, coeff_MLP_sub_norm=0.0
- Mutation: coeff_MLP_node_L1: 1.0 -> 0.1
- Parent rule: root (addressing MLP_node dead issue)

**Slot 1 (id=6)**: Enable MLP_sub scale normalization
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=1000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0
- Mutation: coeff_MLP_sub_norm: 0.0 -> 1.0
- Parent rule: root (addressing alpha ~0.5 issue)

**Slot 2 (id=7)**: Lower lr_k for stability
- Config: seed=42, lr_k=0.002, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=1000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=0.0
- Mutation: lr_k: 0.005 -> 0.002
- Parent rule: root (testing if lr_k too high for rank~50)

**Slot 3 (id=8)**: Longer training duration
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=0.0
- Mutation: data_augmentation_loop: 1000 -> 2000
- Parent rule: root (testing if more iterations help)

Rationale: All 4 experiments from batch 1 failed with test_R2 massively negative and MLP_node completely flat. This batch targets 4 possible causes: (1) MLP_node L1 penalty too strong, (2) MLP_sub scale not pinned, (3) lr_k too high, (4) not enough training iterations.

