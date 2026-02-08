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

---

## Iter 5: failed
Node: id=5, parent=1
Mode/Strategy: exploit (addressing MLP_node dead issue)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=1000, coeff_MLP_node_L1=0.1, coeff_MLP_sub_norm=0.0, coeff_k_floor=0.0
Metrics: rate_constants_R2=0.0130, trimmed_R2=0.9135, n_outliers=61, slope=0.9572, test_R2=-1647195.44, test_pearson=0.0337, final_loss=21632.09, alpha=0.4211
Visual: MLP_sub=bad: c^1 slightly below GT dashed, c^2 linear instead of quadratic (massive divergence), MLP_node=bad: flat lines at 0 (no homeostasis learned despite reduced L1)
Mutation: coeff_MLP_node_L1: 1.0 -> 0.1
Parent rule: Addressing MLP_node dead issue from batch 1
Observation: Reducing MLP_node L1 penalty alone didn't help — R² actually worse (0.013 vs 0.044), more outliers (61 vs 43); MLP_node still dead
Next: parent=6

## Iter 6: failed
Node: id=6, parent=1
Mode/Strategy: explore (pin MLP_sub scale)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=1000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=0.0
Metrics: rate_constants_R2=0.0668, trimmed_R2=0.9189, n_outliers=36, slope=0.9394, test_R2=-100671.97, test_pearson=0.1613, final_loss=17011.14, alpha=0.5018
Visual: MLP_sub=partial: c^1 matches GT well, c^2 linear but closer to GT shape than other configs, MLP_node=bad: flat lines at 0
Mutation: coeff_MLP_sub_norm: 0.0 -> 1.0
Parent rule: Addressing alpha ~0.5 scale ambiguity from batch 1
Observation: **BEST so far** — coeff_MLP_sub_norm=1.0 improved R² to 0.0668 (50% better than baseline), fewer outliers (36 vs 43), better test_pearson (0.16); MLP_sub scale pinning helps
Next: parent=6

## Iter 7: failed
Node: id=7, parent=1
Mode/Strategy: explore (conservative lr_k)
Config: seed=42, lr_k=0.002, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=1000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=0.0, coeff_k_floor=0.0
Metrics: rate_constants_R2=0.0413, trimmed_R2=0.9160, n_outliers=45, slope=0.9596, test_R2=-459153.21, test_pearson=0.1199, final_loss=17036.52, alpha=0.5132
Visual: MLP_sub=bad: c^1 matches GT well, c^2 linear instead of quadratic (significant divergence), MLP_node=bad: flat lines at 0
Mutation: lr_k: 0.005 -> 0.002
Parent rule: Testing if lr_k too high for rank~50 regime
Observation: Lower lr_k slightly worse than baseline (0.041 vs 0.044) — lr_k=0.005 may be appropriate; MLP_node still not learning
Next: parent=6

## Iter 8: failed
Node: id=8, parent=1
Mode/Strategy: explore (longer training)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=0.0, coeff_k_floor=0.0
Metrics: rate_constants_R2=0.0508, trimmed_R2=0.9198, n_outliers=32, slope=0.9510, test_R2=-443580.04, test_pearson=0.1050, final_loss=25074.48, alpha=0.6109
Visual: MLP_sub=partial: c^1 slightly below GT, c^2 linear but leveling toward quadratic at high c, MLP_node=bad: flat lines at 0
Mutation: data_augmentation_loop: 1000 -> 2000
Parent rule: Testing if more training iterations help convergence
Observation: Longer training improved R² (0.051 vs 0.044) and alpha (0.61 vs 0.52), fewer outliers (32 vs 43); training duration helps but not enough alone
Next: parent=6

---

### Batch 2 Summary

**Results Table:**
| Slot | Iter | Mutation | R² | outliers | alpha | test_pearson |
|------|------|----------|-----|----------|-------|--------------|
| 0 | 5 | coeff_MLP_node_L1: 1.0→0.1 | 0.0130 | 61 | 0.42 | 0.03 |
| 1 | 6 | coeff_MLP_sub_norm: 0.0→1.0 | **0.0668** | **36** | 0.50 | **0.16** |
| 2 | 7 | lr_k: 0.005→0.002 | 0.0413 | 45 | 0.51 | 0.12 |
| 3 | 8 | data_augmentation_loop: 1000→2000 | 0.0508 | 32 | **0.61** | 0.11 |

**Key Findings:**
1. **coeff_MLP_sub_norm=1.0 is the most effective single change** — Iter 6 achieved best R²=0.0668
2. **Longer training helps** — Iter 8 improved alpha toward 1.0 and reduced outliers
3. **Reducing coeff_MLP_node_L1 alone doesn't work** — Iter 5 was worst; MLP_node stays dead
4. **Lower lr_k slightly worse** — lr_k=0.005 seems appropriate for this regime

**Persistent Issue:**
- **MLP_node is dead (flat at 0) in ALL configs** — this is the critical bottleneck
- The homeostasis function is essential for dynamics prediction
- Even with coeff_MLP_node_L1=0.1, MLP_node doesn't learn
- This suggests a deeper issue: either lr_node too low, or model needs to be forced to use homeostasis

**Next Batch Strategy:**
Focus on combining the best findings and trying more aggressive MLP_node activation:
- Slot 0: Combine coeff_MLP_sub_norm=1.0 + data_augmentation_loop=2000 (best from both)
- Slot 1: Combine coeff_MLP_sub_norm=1.0 + coeff_MLP_node_L1=0.0 (remove L1 entirely)
- Slot 2: Increase lr_node to 0.005 (force MLP_node to learn faster)
- Slot 3: Try coeff_MLP_sub_norm=1.0 + lr_node=0.005 + coeff_MLP_node_L1=0.0 (aggressive combo)

### Batch 3 (Iter 9-12): Combining Best Findings + Activating MLP_node

**Slot 0 (id=9)**: Combine best findings from Iter 6 and Iter 8
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0
- Mutation: coeff_MLP_sub_norm: 0.0 -> 1.0, data_augmentation_loop: 1000 -> 2000
- Parent rule: Combining best single changes from Iter 6 (norm) and Iter 8 (longer training)

**Slot 1 (id=10)**: Remove MLP_node L1 entirely + keep sub_norm
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=1000, coeff_MLP_node_L1=0.0, coeff_MLP_sub_norm=1.0
- Mutation: coeff_MLP_node_L1: 1.0 -> 0.0 (from Iter 6 baseline)
- Parent rule: Test if removing L1 entirely helps MLP_node learn while keeping sub_norm

**Slot 2 (id=11)**: Increase lr_node aggressively
- Config: seed=42, lr_k=0.005, lr_node=0.005, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=1000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0
- Mutation: lr_node: 0.001 -> 0.005 (5x increase)
- Parent rule: Force MLP_node to learn by increasing its learning rate to match lr_k

**Slot 3 (id=12)**: Aggressive combination — all MLP_node activation strategies
- Config: seed=42, lr_k=0.005, lr_node=0.005, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=0.0, coeff_MLP_sub_norm=1.0
- Mutation: lr_node: 0.001 -> 0.005, coeff_MLP_node_L1: 1.0 -> 0.0, data_augmentation_loop: 1000 -> 2000, coeff_MLP_sub_norm: 0.0 -> 1.0
- Parent rule: Testing principle: "MLP_node needs both higher lr_node AND no L1 penalty to learn"

Rationale: The persistent issue is dead MLP_node (flat at 0). This batch combines the best hyperparameters (coeff_MLP_sub_norm=1.0, data_augmentation_loop=2000) while aggressively trying to activate MLP_node through higher lr_node and/or removing L1 penalty.

---

## Iter 9: failed
Node: id=9, parent=6
Mode/Strategy: exploit (combine best)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=0.0
Metrics: rate_constants_R2=0.0535, trimmed_R2=0.9352, n_outliers=27, slope=0.9592, test_R2=-122383.04, test_pearson=0.1386, final_loss=23952.81, alpha=0.6192
Visual: MLP_sub=partial: c^1 slightly compressed but correct shape, c^2 quadratic and correct shape, MLP_node=good: linear homeostasis with correct type-differentiated slopes (λ=0.001, 0.002)
Mutation: coeff_MLP_sub_norm: 0.0 -> 1.0, data_augmentation_loop: 1000 -> 2000
Parent rule: Combining best from Iter 6 (sub_norm) and Iter 8 (longer training)
Observation: MLP_node is NOW ACTIVE with correct linear homeostasis! R²=0.0535 slightly worse than Iter 6 (0.067) but better alpha (0.62) and fewer outliers (27)
Next: parent=9

## Iter 10: failed
Node: id=10, parent=6
Mode/Strategy: explore (remove L1 entirely)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=1000, coeff_MLP_node_L1=0.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=0.0
Metrics: rate_constants_R2=0.0608, trimmed_R2=0.9145, n_outliers=38, slope=0.9371, test_R2=-1561792.89, test_pearson=-0.0033, final_loss=16456.76, alpha=0.5081
Visual: MLP_sub=partial: c^1 correct, c^2 quadratic and correct shape, MLP_node=good: linear homeostasis with type-differentiated slopes
Mutation: coeff_MLP_node_L1: 1.0 -> 0.0 (from Iter 6 baseline)
Parent rule: Test if removing L1 entirely helps while keeping sub_norm=1.0
Observation: MLP_node ACTIVE. R²=0.0608 close to best. But test_pearson collapsed (-0.003) suggesting dynamics diverge despite good k recovery
Next: parent=9

## Iter 11: failed
Node: id=11, parent=6
Mode/Strategy: explore (high lr_node)
Config: seed=42, lr_k=0.005, lr_node=0.005, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=1000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=0.0
Metrics: rate_constants_R2=0.0166, trimmed_R2=0.9011, n_outliers=57, slope=0.9234, test_R2=-756025.23, test_pearson=-0.0508, final_loss=19371.92, alpha=0.4422
Visual: MLP_sub=partial: c^1 correct, c^2 quadratic, MLP_node=good: linear homeostasis with correct slopes
Mutation: lr_node: 0.001 -> 0.005
Parent rule: Force MLP_node to learn by increasing lr_node to match lr_k
Observation: High lr_node HURT R² badly (0.017 vs 0.067). MLP_node active but lr_node=0.005 destabilizes k learning; test_pearson negative
Next: parent=9

## Iter 12: failed
Node: id=12, parent=6
Mode/Strategy: principle-test
Config: seed=42, lr_k=0.005, lr_node=0.005, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=0.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=0.0
Metrics: rate_constants_R2=0.0613, trimmed_R2=0.9336, n_outliers=30, slope=0.9438, test_R2=-40940.78, test_pearson=0.1180, final_loss=24102.87, alpha=0.6197
Visual: MLP_sub=partial: c^1 correct, c^2 quadratic, MLP_node=good: linear homeostasis with correct type slopes
Mutation: lr_node: 0.001 -> 0.005, coeff_MLP_node_L1: 1.0 -> 0.0, data_augmentation_loop: 1000 -> 2000, coeff_MLP_sub_norm: 0.0 -> 1.0. Testing principle: "MLP_node needs higher lr_node AND no L1"
Parent rule: Aggressive combo to activate MLP_node
Observation: R²=0.0613 comparable to best, alpha=0.62 (best), MLP_node active. But combination didn't beat simpler approaches; longer training offset lr_node instability
Next: parent=9

---

### Batch 3 Summary

**Results Table:**
| Slot | Iter | Mutation | R² | outliers | alpha | test_pearson |
|------|------|----------|-----|----------|-------|--------------|
| 0 | 9 | sub_norm=1.0 + aug=2000 | 0.0535 | **27** | **0.62** | 0.139 |
| 1 | 10 | node_L1=0.0 | 0.0608 | 38 | 0.51 | -0.003 |
| 2 | 11 | lr_node=0.005 | 0.0166 | 57 | 0.44 | -0.051 |
| 3 | 12 | lr_node=0.005 + node_L1=0.0 + aug=2000 | 0.0613 | 30 | **0.62** | 0.118 |

**Key Findings:**
1. **MLP_node is now ACTIVE in all 4 configs** — showing correct linear homeostasis with per-type slopes (λ=0.001, 0.002)
2. **MLP_sub c^2 curve is now quadratic** (no longer linear) — significant improvement over batches 1-2
3. **Best R² still ~0.06** — no improvement over Iter 6's 0.0668 from batch 2
4. **lr_node=0.005 hurts** — Iter 11 had worst R² (0.017); high lr_node destabilizes training
5. **Longer training + sub_norm gives best alpha** — Iters 9 and 12 both achieved alpha=0.62

**Principle Updates:**
- CONFIRMED: coeff_MLP_sub_norm=1.0 is essential (enables correct MLP function shapes)
- NEW: MLP_node becomes active when coeff_MLP_sub_norm=1.0 is set (surprising causal link)
- REFUTED: "Higher lr_node helps MLP_node" — actually hurts overall R²
- PARTIAL: "No L1 on MLP_node helps" — R² similar but dynamics prediction worse

**Block 1 Completed** — 12 iterations done. Best R²=0.0668 (Iter 6). MLP functions now correct shape. R² plateau at ~0.06 suggests need for different approach in Block 2.

>>> BLOCK 1 END <<<

---

## Block 2: Breaking the R² Plateau

### Batch 4 (Iter 13-16): New Approaches to Break Plateau

Base config from Iter 9: lr_k=0.005, lr_node=0.001, lr_sub=0.0005, data_augmentation_loop=2000, coeff_MLP_sub_norm=1.0, coeff_MLP_node_L1=1.0

**Slot 0 (id=13)**: Recurrent training (multi-step rollout)
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, recurrent_training=true, time_step=4
- Mutation: recurrent_training: false -> true, time_step: 1 -> 4
- Parent rule: Node 9 (best alpha/outliers), testing if multi-step rollout breaks k degeneracy

**Slot 1 (id=14)**: Add k_floor penalty to prevent outliers
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: coeff_k_floor: 0.0 -> 1.0, k_floor_threshold: -3.0 -> -2.0
- Parent rule: Node 9, testing if k_floor reduces outliers and improves R²

**Slot 2 (id=15)**: Smaller MLP architecture (implicit regularization)
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, hidden_dim_sub=32, n_layers_sub=2, hidden_dim_node=32, n_layers_node=2
- Mutation: hidden_dim_sub: 64 -> 32, n_layers_sub: 3 -> 2, hidden_dim_node: 64 -> 32, n_layers_node: 3 -> 2
- Parent rule: Node 9, testing if smaller MLPs prevent over-fitting and improve k recovery

**Slot 3 (id=16)**: Stronger monotonicity constraint
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_MLP_sub_diff=10
- Mutation: coeff_MLP_sub_diff: 5 -> 10
- Parent rule: Node 9, testing principle "stronger MLP_sub monotonicity improves k learning"

Rationale: Block 1 achieved R²~0.06 plateau with correct MLP shapes. This batch explores 4 different approaches to break the plateau: (1) recurrent training to enforce trajectory consistency, (2) k_floor to prevent outlier log_k values, (3) smaller MLPs for implicit regularization, (4) stronger monotonicity constraint.

---

## Iter 13: failed
Node: id=13, parent=9
Mode/Strategy: exploit (recurrent training)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=0.0, recurrent_training=true, time_step=4
Metrics: rate_constants_R2=0.0563, trimmed_R2=0.9323, n_outliers=28, slope=0.9613, test_R2=-170.32, test_pearson=0.4600, final_loss=25697.81, alpha=0.6134
Visual: MLP_sub=partial: c^1 linear correct, c^2 sublinear (below GT quadratic), MLP_node=bad: flat lines at 0 (dead)
Mutation: recurrent_training: false -> true, time_step: 1 -> 4
Parent rule: Node 9, testing if multi-step rollout breaks k degeneracy
Observation: Recurrent training did NOT improve R² (0.056 vs 0.054 baseline), 3.5x slower (131 min). MLP_node regressed to dead state. test_pearson improved to 0.46 but k recovery same.
Next: parent=14

## Iter 14: partial
Node: id=14, parent=9
Mode/Strategy: exploit (k_floor penalty)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.5076, trimmed_R2=0.9330, n_outliers=33, slope=0.9764, test_R2=-3255043.93, test_pearson=-0.0167, final_loss=25658.69, alpha=0.5810
Visual: MLP_sub=partial: c^1 matches GT well, c^2 sublinear (below GT quadratic), MLP_node=bad: flat lines at 0 (dead)
Mutation: coeff_k_floor: 0.0 -> 1.0, k_floor_threshold: -3.0 -> -2.0
Parent rule: Node 9, testing if k_floor prevents outliers and improves R²
Observation: **MAJOR BREAKTHROUGH** R² jumped from 0.06 to 0.51! k_floor penalty at threshold=-2.0 prevents outlier log_k values from drifting below true minimum. Trimmed R² unchanged (0.93), but raw R² massively improved. MLP_node still dead.
Next: parent=14

## Iter 15: failed
Node: id=15, parent=9
Mode/Strategy: explore (smaller MLP)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=0.0, hidden_dim_sub=32, n_layers_sub=2, hidden_dim_node=32, n_layers_node=2
Metrics: rate_constants_R2=0.0109, trimmed_R2=0.9457, n_outliers=36, slope=0.9368, test_R2=-1889072.42, test_pearson=-0.0418, final_loss=31982.66, alpha=0.5969
Visual: MLP_sub=partial: c^1 linear, c^2 quadratic near GT, MLP_node=bad: flat lines at 0 (dead)
Mutation: hidden_dim_sub: 64 -> 32, n_layers_sub: 3 -> 2, hidden_dim_node: 64 -> 32, n_layers_node: 3 -> 2
Parent rule: Node 9, testing if smaller MLPs prevent overfitting
Observation: Smaller MLP WORST result (R²=0.011). Reduced capacity hurt k recovery significantly. MLP_node still dead. NOT a viable direction.
Next: parent=14

## Iter 16: failed
Node: id=16, parent=9
Mode/Strategy: principle-test
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=0.0, coeff_MLP_sub_diff=10
Metrics: rate_constants_R2=0.0567, trimmed_R2=0.9413, n_outliers=33, slope=0.9707, test_R2=-379396.92, test_pearson=0.0897, final_loss=23428.23, alpha=0.6411
Visual: MLP_sub=partial: c^1 linear, c^2 sublinear (below GT), MLP_node=bad: flat lines at 0 (dead)
Mutation: coeff_MLP_sub_diff: 5 -> 10. Testing principle: "stronger MLP_sub monotonicity improves k learning"
Parent rule: Node 9, testing stronger monotonicity constraint
Observation: Stronger monotonicity (diff=10) made no difference (R²=0.057 vs 0.054). MLP_node still dead. Principle NOT supported.
Next: parent=14

---

### Batch 4 Summary

**Results Table:**
| Slot | Iter | Mutation | R² | outliers | alpha | test_pearson |
|------|------|----------|-----|----------|-------|--------------|
| 0 | 13 | recurrent time_step=4 | 0.0563 | 28 | 0.61 | 0.46 |
| 1 | 14 | coeff_k_floor=1.0, threshold=-2.0 | **0.5076** | 33 | 0.58 | -0.02 |
| 2 | 15 | smaller MLP (32/2) | 0.0109 | 36 | 0.60 | -0.04 |
| 3 | 16 | coeff_MLP_sub_diff=10 | 0.0567 | 33 | 0.64 | 0.09 |

**Key Findings:**
1. **coeff_k_floor=1.0 with k_floor_threshold=-2.0 is BREAKTHROUGH** — R² jumped from 0.06 to 0.51 (8.5x improvement!)
2. **Recurrent training (time_step=4) did NOT help** — similar R² but 3.5x slower
3. **Smaller MLP architecture HURT** — worst R² of 0.011
4. **Stronger monotonicity (diff=10) had no effect** — principle not supported

**Critical Observation:**
- MLP_node is DEAD (flat at 0) in ALL 4 configs this batch, even though it was active in batch 3
- The key difference: batch 3 used data_augmentation_loop=2000 with coeff_MLP_sub_norm=1.0 starting fresh
- Batch 4 started from configs that may have had different initialization paths

**Principle Updates:**
- NEW: **coeff_k_floor=1.0 with k_floor_threshold=-2.0 dramatically improves R²** (evidence: Iter 14, R² 0.06→0.51)
- REFUTED: "Recurrent training breaks degeneracy plateau" — no R² improvement
- REFUTED: "Smaller MLP improves k recovery" — actually hurt significantly
- REFUTED: "Stronger monotonicity helps" — no effect

**Next Batch Strategy:**
Exploit the k_floor breakthrough by combining with other successful settings:
- Slot 0: k_floor + longer training (aug=3000)
- Slot 1: k_floor + no L1 on MLP_node (trying to activate it)
- Slot 2: k_floor + higher lr_k (test if we can push R² higher)
- Slot 3: k_floor + k_floor_threshold=-2.5 (test if tighter bound helps)

### Batch 5 (Iter 17-20): Exploiting k_floor Breakthrough

Base config from Iter 14: lr_k=0.005, lr_node=0.001, lr_sub=0.0005, data_augmentation_loop=2000, coeff_MLP_sub_norm=1.0, coeff_MLP_node_L1=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0

**Slot 0 (id=17)**: Longer training with k_floor
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=3000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: data_augmentation_loop: 2000 -> 3000
- Parent rule: Node 14 (k_floor breakthrough), testing if more training pushes R² higher

**Slot 1 (id=18)**: Remove MLP_node L1 with k_floor
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=0.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: coeff_MLP_node_L1: 1.0 -> 0.0
- Parent rule: Node 14, testing if removing L1 activates MLP_node while keeping k_floor benefit

**Slot 2 (id=19)**: Higher lr_k with k_floor
- Config: seed=42, lr_k=0.01, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: lr_k: 0.005 -> 0.01
- Parent rule: Node 14, testing if faster k learning with floor constraint pushes R² higher

**Slot 3 (id=20)**: Tighter k_floor threshold
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.5
- Mutation: k_floor_threshold: -2.0 -> -2.5
- Parent rule: Node 14, testing principle "tighter k_floor bound reduces outliers further"

Rationale: Iter 14's k_floor=1.0 with threshold=-2.0 achieved breakthrough R²=0.51. This batch explores 4 variations to push performance higher: (1) longer training, (2) remove MLP_node L1 to try activating it, (3) higher lr_k with the safety of k_floor, (4) tighter threshold.

---

## Iter 17: partial
Node: id=17, parent=14
Mode/Strategy: exploit (longer training)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=3000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.6382, trimmed_R2=0.9590, n_outliers=17, slope=0.9551, test_R2=-366074.72, test_pearson=0.1057, final_loss=29556.97, alpha=0.7848
Visual: MLP_sub=good: c^1 linear, c^2 quadratic matching GT well, MLP_node=good: linear with correct slopes (λ=0.001, 0.002)
Mutation: data_augmentation_loop: 2000 -> 3000
Parent rule: Node 14, testing if more training pushes R² higher
Observation: **MAJOR IMPROVEMENT** R² improved from 0.51 to 0.64 (+25%). Outliers reduced from 33 to 17 (48% reduction). Alpha improved to 0.78 (closer to 1.0). MLP_node became ACTIVE with correct linear behavior! Longer training with k_floor is synergistic.
Next: parent=17

## Iter 18: partial
Node: id=18, parent=14
Mode/Strategy: exploit (remove L1)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=0.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.6418, trimmed_R2=0.9424, n_outliers=24, slope=0.9648, test_R2=-417515.29, test_pearson=0.0427, final_loss=25123.83, alpha=0.6418
Visual: MLP_sub=good: c^1 linear, c^2 quadratic matching GT well, MLP_node=good: linear with correct slopes
Mutation: coeff_MLP_node_L1: 1.0 -> 0.0
Parent rule: Node 14, testing if removing L1 activates MLP_node while keeping k_floor benefit
Observation: **BEST R²=0.6418** — slightly better than Iter 17 but with standard training length. More outliers (24 vs 17) and lower alpha (0.64 vs 0.78). Removing L1 helps R² but loses some alpha improvement.
Next: parent=17

## Iter 19: partial
Node: id=19, parent=14
Mode/Strategy: exploit (higher lr_k)
Config: seed=42, lr_k=0.01, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.4700, trimmed_R2=0.9497, n_outliers=26, slope=0.9610, test_R2=-151583.48, test_pearson=0.0759, final_loss=24936.36, alpha=0.5739
Visual: MLP_sub=good: c^1 linear, c^2 quadratic, MLP_node=good: linear with correct slopes
Mutation: lr_k: 0.005 -> 0.01
Parent rule: Node 14, testing if faster k learning with floor constraint pushes R² higher
Observation: Higher lr_k=0.01 HURT R² (0.47 vs 0.51). Even with k_floor, higher lr_k causes instability. lr_k=0.005 remains optimal.
Next: parent=17

## Iter 20: partial
Node: id=20, parent=14
Mode/Strategy: principle-test
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=2000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.5
Metrics: rate_constants_R2=0.3732, trimmed_R2=0.9230, n_outliers=35, slope=0.9838, test_R2=-2406895.59, test_pearson=0.1174, final_loss=27298.13, alpha=0.5712
Visual: MLP_sub=good: c^1 linear, c^2 quadratic, MLP_node=good: linear with correct slopes
Mutation: k_floor_threshold: -2.0 -> -2.5. Testing principle: "tighter k_floor threshold further reduces outliers"
Parent rule: Node 14, testing tighter k_floor threshold
Observation: Tighter threshold=-2.5 HURT R² significantly (0.37 vs 0.51). More outliers (35 vs 33). The penalty is too weak with threshold below true min. threshold=-2.0 matching log_k_min is optimal.
Next: parent=17

---

### Batch 5 Summary

**Results Table:**
| Slot | Iter | Mutation | R² | outliers | alpha | slope |
|------|------|----------|-----|----------|-------|-------|
| 0 | 17 | aug=3000 (longer) | **0.6382** | **17** | **0.78** | 0.96 |
| 1 | 18 | L1=0.0 (no node L1) | **0.6418** | 24 | 0.64 | 0.96 |
| 2 | 19 | lr_k=0.01 (higher) | 0.4700 | 26 | 0.57 | 0.96 |
| 3 | 20 | threshold=-2.5 (tighter) | 0.3732 | 35 | 0.57 | 0.98 |

**Key Findings:**
1. **Longer training (aug=3000) with k_floor achieved R²=0.64** — 25% improvement over Iter 14 (0.51), outliers reduced 48%
2. **Removing L1 also achieved R²=0.64** — slightly higher R² but fewer alpha improvement
3. **Higher lr_k=0.01 HURT** — R² dropped to 0.47 even with k_floor safety
4. **Tighter threshold=-2.5 HURT significantly** — R² dropped to 0.37, threshold should match log_k_min

**Critical Insight:**
- Iter 17 (aug=3000) is the best overall: highest R², fewest outliers (17), best alpha (0.78)
- MLP_node is now ACTIVE in all 4 configs with correct linear homeostasis behavior
- The combination of k_floor + longer training is synergistic

**Principle Updates:**
- STRENGTHENED: "Longer training helps" — now with k_floor, aug=3000 gives significant improvement
- NEW: "k_floor_threshold should match log_k_min" — threshold=-2.0 optimal, tighter is worse
- CONFIRMED: "lr_k=0.005 is optimal" — even with k_floor protection, lr_k=0.01 hurts

**Best Node:** Iter 17 (R²=0.6382, outliers=17, alpha=0.78)

### Batch 6 (Iter 21-24): Pushing Beyond R²=0.64

Base config from Iter 17: lr_k=0.005, lr_node=0.001, lr_sub=0.0005, data_augmentation_loop=3000, coeff_MLP_sub_norm=1.0, coeff_MLP_node_L1=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0

**Slot 0 (id=21)**: Even longer training
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: data_augmentation_loop: 3000 -> 4000
- Parent rule: Node 17 (best R², alpha, outliers), testing if even more training pushes R² above 0.7

**Slot 1 (id=22)**: Combine aug=3000 + L1=0.0 (best of both)
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=3000, coeff_MLP_node_L1=0.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: coeff_MLP_node_L1: 1.0 -> 0.0 (from Iter 17 baseline)
- Parent rule: Node 17, combining best aspects of Iter 17 (aug=3000) and Iter 18 (L1=0.0)

**Slot 2 (id=23)**: Lower lr_sub with longer training
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0002, batch_size=8, n_epochs=1, data_augmentation_loop=3000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: lr_sub: 0.0005 -> 0.0002
- Parent rule: Node 17, testing if slower MLP_sub learning lets k dominate and improves recovery

**Slot 3 (id=24)**: Stronger k_floor penalty
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=3000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=2.0, k_floor_threshold=-2.0
- Mutation: coeff_k_floor: 1.0 -> 2.0. Testing principle: "stronger k_floor penalty reduces outliers further"
- Parent rule: Node 17, testing if doubling k_floor penalty reduces outliers from 17

Rationale: Iter 17 achieved R²=0.64 with 17 outliers and alpha=0.78. This batch explores 4 variations: (1) even longer training (aug=4000), (2) combining best aspects of Iter 17 and 18, (3) slower MLP_sub to let k dominate, (4) stronger k_floor to reduce outliers.

---

## Iter 21: partial
Node: id=21, parent=17
Mode/Strategy: exploit (even longer training)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.6896, trimmed_R2=0.9621, n_outliers=16, slope=0.9762, test_R2=-4706.46, test_pearson=0.3319, final_loss=33292.95, alpha=0.8543
Visual: MLP_sub=good: c^1 linear close to GT, c^2 quadratic matching GT well at lower concentrations, MLP_node=bad: flat lines at 0, homeostasis not learned
Mutation: data_augmentation_loop: 3000 -> 4000
Parent rule: Node 17 (best), testing if even more training pushes R² above 0.7
Observation: **NEW BEST R²=0.6896** (+8% over Iter 17). Outliers reduced from 17→16, alpha improved 0.78→0.85. Longer training continues to help. However MLP_node remains flat despite L1=1.0. test_pearson improved to 0.33 (best so far).
Next: parent=21

## Iter 22: partial
Node: id=22, parent=17
Mode/Strategy: exploit (combine aug=3000 + L1=0.0)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=3000, coeff_MLP_node_L1=0.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.4193, trimmed_R2=0.9533, n_outliers=29, slope=0.9633, test_R2=-419437.11, test_pearson=0.1897, final_loss=33671.34, alpha=0.6669
Visual: MLP_sub=good: c^1 linear, c^2 quadratic but slightly compressed vs GT, MLP_node=bad: flat lines at 0, homeostasis not learned
Mutation: coeff_MLP_node_L1: 1.0 -> 0.0
Parent rule: Node 17, combining best aspects of Iter 17 (aug=3000) and Iter 18 (L1=0.0)
Observation: **WORSE than expected** R²=0.42 (vs 0.64 for Iter 17 and 0.64 for Iter 18). Combination failed — more outliers (29), worse alpha (0.67). L1=0.0 with aug=3000 is harmful; they don't combine well.
Next: parent=21

## Iter 23: partial
Node: id=23, parent=17
Mode/Strategy: explore (lower lr_sub)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0002, batch_size=8, n_epochs=1, data_augmentation_loop=3000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.6581, trimmed_R2=0.9602, n_outliers=19, slope=0.9743, test_R2=-929868.84, test_pearson=0.1322, final_loss=27791.38, alpha=0.7753
Visual: MLP_sub=good: c^1 linear close to GT, c^2 quadratic matching GT well, MLP_node=bad: flat lines at 0, homeostasis not learned
Mutation: lr_sub: 0.0005 -> 0.0002
Parent rule: Node 17, testing if slower MLP_sub learning lets k dominate and improves recovery
Observation: Lower lr_sub slightly HURT R² (0.66 vs 0.64). More outliers (19 vs 17). Slowing MLP_sub doesn't help when k_floor is active; the constraint is more effective than lr tuning.
Next: parent=21

## Iter 24: partial
Node: id=24, parent=17
Mode/Strategy: principle-test
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=3000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=2.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.5589, trimmed_R2=0.9628, n_outliers=24, slope=0.9796, test_R2=-850537.65, test_pearson=0.0164, final_loss=30359.13, alpha=0.7537
Visual: MLP_sub=good: c^1 linear close to GT, c^2 quadratic matching GT well, MLP_node=bad: flat lines at 0, homeostasis not learned
Mutation: coeff_k_floor: 1.0 -> 2.0. Testing principle: "stronger k_floor penalty reduces outliers further"
Parent rule: Node 17, testing if doubling k_floor penalty reduces outliers from 17
Observation: Stronger k_floor=2.0 HURT R² (0.56 vs 0.64). More outliers (24 vs 17). The penalty is too strong — it interferes with k optimization. k_floor=1.0 is optimal.
Next: parent=21

---

### Batch 6 Summary

**Results Table:**
| Slot | Iter | Mutation | R² | outliers | alpha | slope |
|------|------|----------|-----|----------|-------|-------|
| 0 | 21 | aug=4000 (longer) | **0.6896** | **16** | **0.85** | 0.98 |
| 1 | 22 | L1=0.0 (no node L1) | 0.4193 | 29 | 0.67 | 0.96 |
| 2 | 23 | lr_sub=0.0002 (lower) | 0.6581 | 19 | 0.78 | 0.97 |
| 3 | 24 | k_floor=2.0 (stronger) | 0.5589 | 24 | 0.75 | 0.98 |

**Key Findings:**
1. **Longer training (aug=4000) achieved NEW BEST R²=0.6896** — 8% improvement over Iter 17 (0.64). Alpha improved to 0.85, outliers=16.
2. **Combining L1=0.0 + aug=3000 FAILED** — R²=0.42, worse than either alone. Unexpected negative interaction.
3. **Lower lr_sub=0.0002 slightly worse** — R²=0.66, doesn't help when k_floor is active.
4. **Stronger k_floor=2.0 HURT** — R²=0.56, penalty too strong. k_floor=1.0 is optimal.

**Critical Insight:**
- Training duration is the most effective lever: aug=3000→4000 gave 8% R² improvement
- MLP_node is FLAT (not learning) in all 4 configs despite correct MLP_sub shapes — investigate in next batch
- test_pearson improved significantly for Iter 21 (0.33 vs 0.13 for others) — better dynamics fit

**Principle Updates:**
- STRENGTHENED: "Longer training helps" — aug=4000 > aug=3000, diminishing returns not yet reached
- NEW: "k_floor=1.0 is optimal" — k_floor=2.0 too strong, hurts R²
- REFUTED: "L1=0.0 + longer training combines well" — combination was harmful
- REFUTED: "Lower lr_sub helps k dominate" — no effect when k_floor is active

**Best Node:** Iter 21 (R²=0.6896, outliers=16, alpha=0.85)

>>> BLOCK END <<<

---

## Block 3: Pushing Beyond R²=0.7

### Batch 7 (Iter 25-28): Longer training and MLP_node activation

Base config from Iter 21: lr_k=0.005, lr_node=0.001, lr_sub=0.0005, data_augmentation_loop=4000, coeff_MLP_sub_norm=1.0, coeff_MLP_node_L1=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0

**Slot 0 (id=25)**: Even longer training
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=5000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: data_augmentation_loop: 4000 -> 5000
- Parent rule: Node 21 (best R²=0.6896), testing if even more training pushes R² above 0.75

**Slot 1 (id=26)**: Higher lr_node to activate MLP_node
- Config: seed=42, lr_k=0.005, lr_node=0.002, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: learning_rate_node: 0.001 -> 0.002
- Parent rule: Node 21, testing if faster MLP_node learning activates homeostasis (flat in all Block 2 runs)

**Slot 2 (id=27)**: Different seed to break degeneracy
- Config: seed=123, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: seed: 42 -> 123
- Parent rule: Node 21, testing if MLP_node flatness is seed-specific initialization issue

**Slot 3 (id=28)**: Smaller batch size for more gradient updates
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=4, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: batch_size: 8 -> 4. Testing principle: "smaller batch with more updates helps convergence"
- Parent rule: Node 21, testing if more frequent weight updates improves R²

Rationale: Iter 21 achieved R²=0.69 with alpha=0.85 but MLP_node remained flat. This batch explores: (1) longer training (aug=5000), (2) higher lr_node to activate homeostasis, (3) different seed to break initialization degeneracy, (4) smaller batch for more gradient updates.

---

## Iter 25: partial
Node: id=25, parent=21
Mode/Strategy: exploit (even longer training)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=5000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.6518, trimmed_R2=0.9649, n_outliers=18, slope=0.9691, test_R2=-1.17, test_pearson=0.6681, final_loss=37004.79, alpha=0.9494
Visual: MLP_sub=partial: c^1 linear close to GT, c^2 sublinear (diverges from GT quadratic at high c), MLP_node=bad: flat lines at 0 (homeostasis not learned)
Mutation: data_augmentation_loop: 4000 -> 5000
Parent rule: Node 21 (best R²=0.6896), testing if longer training pushes R² above 0.75
Observation: R²=0.65 is LOWER than Iter 21 (0.69) — aug=5000 hit diminishing returns or overfitting. However, alpha=0.95 is BEST EVER and test_pearson=0.67 improved significantly. More outliers (18 vs 16). Trading off R² for better MLP_sub scale.
Next: parent=21

## Iter 26: partial
Node: id=26, parent=21
Mode/Strategy: exploit (higher lr_node)
Config: seed=42, lr_k=0.005, lr_node=0.002, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.6382, trimmed_R2=0.9623, n_outliers=18, slope=0.9771, test_R2=-21.67, test_pearson=0.5955, final_loss=32163.51, alpha=0.8611
Visual: MLP_sub=partial: c^1 linear close to GT, c^2 sublinear (below GT quadratic), MLP_node=bad: flat lines at 0 (homeostasis not learned despite higher lr_node)
Mutation: learning_rate_node: 0.001 -> 0.002
Parent rule: Node 21, testing if faster MLP_node learning activates homeostasis
Observation: lr_node=0.002 did NOT activate MLP_node. R²=0.64 slightly worse than Iter 21 (0.69). test_pearson=0.60 good. MLP_node remains dead even with 2x learning rate — the issue is not lr_node magnitude.
Next: parent=21

## Iter 27: partial
Node: id=27, parent=21
Mode/Strategy: explore (different seed)
Config: seed=123, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.6141, trimmed_R2=0.9637, n_outliers=15, slope=0.9683, test_R2=-651163.61, test_pearson=0.1543, final_loss=30894.26, alpha=0.8435
Visual: MLP_sub=partial: c^1 linear close to GT, c^2 sublinear (below GT quadratic), MLP_node=bad: flat lines at 0 (homeostasis not learned)
Mutation: seed: 42 -> 123
Parent rule: Node 21, testing if MLP_node flatness is seed-specific
Observation: Different seed (123) did NOT activate MLP_node. R²=0.61 worse than seed=42 (0.69). Fewest outliers (15) but worst test_pearson (0.15). MLP_node degeneracy is not seed-specific — it's structural.
Next: parent=21

## Iter 28: partial
Node: id=28, parent=21
Mode/Strategy: principle-test
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=4, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.5995, trimmed_R2=0.9688, n_outliers=17, slope=0.9905, test_R2=-1112792.70, test_pearson=0.1196, final_loss=34344.87, alpha=0.8218
Visual: MLP_sub=partial: c^1 linear close to GT, c^2 sublinear with slight wiggle at mid-concentration, MLP_node=bad: flat lines at 0 (homeostasis not learned)
Mutation: batch_size: 8 -> 4. Testing principle: "smaller batch with more updates helps convergence"
Parent rule: Node 21, testing if more frequent weight updates improves R²
Observation: Smaller batch_size=4 HURT R² (0.60 vs 0.69). Worst test_pearson (0.12). More gradient noise didn't help. Principle REFUTED: smaller batch does NOT help convergence in this regime.
Next: parent=21

---

### Batch 7 Summary

**Results Table:**
| Slot | Iter | Mutation | R² | outliers | alpha | test_pearson |
|------|------|----------|-----|----------|-------|--------------|
| 0 | 25 | aug=5000 (longer) | 0.6518 | 18 | **0.95** | **0.67** |
| 1 | 26 | lr_node=0.002 (higher) | 0.6382 | 18 | 0.86 | 0.60 |
| 2 | 27 | seed=123 (different) | 0.6141 | **15** | 0.84 | 0.15 |
| 3 | 28 | batch_size=4 (smaller) | 0.5995 | 17 | 0.82 | 0.12 |

**Key Findings:**
1. **Longer training (aug=5000) DID NOT improve R²** — R²=0.65 is WORSE than Iter 21 (0.69). However, alpha=0.95 is best ever!
2. **Higher lr_node=0.002 did NOT activate MLP_node** — R²=0.64, MLP_node still dead
3. **Different seed (123) did NOT break MLP_node degeneracy** — R²=0.61, fewest outliers (15) but worse R²
4. **Smaller batch_size=4 HURT performance** — R²=0.60, worst in batch

**Critical Insights:**
- **Iter 21 remains the best R²=0.6896** — none of the batch 7 experiments improved upon it
- **aug=5000 shows diminishing returns** — R² dropped but alpha and test_pearson improved
- **MLP_node is structurally dead** — not a seed issue, not an lr_node issue; needs different intervention
- **All configs show c^2 sublinear** — MLP_sub not learning proper quadratic for |s|=2

**Principle Updates:**
- REFUTED: "Longer training always helps" — aug=5000 hurt R² vs aug=4000
- REFUTED: "Higher lr_node activates MLP_node" — lr_node=0.002 made no difference
- REFUTED: "MLP_node degeneracy is seed-specific" — different seed didn't help
- REFUTED: "Smaller batch size helps convergence" — batch_size=4 hurt R²

**Best Node Remains:** Iter 21 (R²=0.6896, outliers=16, alpha=0.85)

### Batch 8 (Iter 29-32): Alternative approaches to break plateau

Base config from Iter 21: lr_k=0.005, lr_node=0.001, lr_sub=0.0005, data_augmentation_loop=4000, coeff_MLP_sub_norm=1.0, coeff_MLP_node_L1=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0

**Slot 0 (id=29)**: Remove MLP_node L1 with long training
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=0.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: coeff_MLP_node_L1: 1.0 -> 0.0
- Parent rule: Node 21, testing if removing L1 penalty helps MLP_node learn when combined with aug=4000

**Slot 1 (id=30)**: Higher coeff_MLP_sub_norm
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=2.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: coeff_MLP_sub_norm: 1.0 -> 2.0
- Parent rule: Node 21, testing if stronger MLP_sub scale pinning improves alpha and R²

**Slot 2 (id=31)**: Lower lr_k for finer convergence
- Config: seed=42, lr_k=0.003, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: lr_k: 0.005 -> 0.003
- Parent rule: Node 21, testing if finer k updates near convergence improves R²

**Slot 3 (id=32)**: Stronger sub_diff monotonicity
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0, coeff_MLP_sub_diff=10
- Mutation: coeff_MLP_sub_diff: 5 -> 10. Testing principle: "Stronger monotonicity fixes c^2 sublinear shape"
- Parent rule: Node 21, testing if stronger monotonicity constraint fixes the c^2 sublinear issue

Rationale: All batch 7 experiments failed to improve over Iter 21 and MLP_node remained dead. This batch explores: (1) remove L1 with long training, (2) stronger sub_norm, (3) lower lr_k for finer convergence, (4) stronger monotonicity to fix c^2 shape.

---

## Iter 29: partial
Node: id=29, parent=21
Mode/Strategy: exploit (remove L1 with long training)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=0.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.5066, trimmed_R2=0.9672, n_outliers=19, slope=0.9681, test_R2=-349940.41, test_pearson=0.1001, final_loss=31636.78, alpha=0.8307
Visual: MLP_sub=good: c^1 linear, c^2 quadratic matching GT well, MLP_node=good: linear with correct slopes (type 0 λ=0.001, type 1 λ=0.002)
Mutation: coeff_MLP_node_L1: 1.0 -> 0.0
Parent rule: Node 21, testing if removing L1 penalty helps MLP_node learn when combined with aug=4000
Observation: R²=0.51 is WORSE than Iter 21 (0.69). Removing L1 hurt even with aug=4000 (Iter 22 with L1=0+aug=3000 also hurt). MLP_node is ACTIVE with correct linear homeostasis shapes. L1=0 activates MLP_node but hurts R².
Next: parent=21

## Iter 30: partial
Node: id=30, parent=21
Mode/Strategy: exploit (stronger sub_norm)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=2.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.6191, trimmed_R2=0.9536, n_outliers=14, slope=0.9874, test_R2=-1192827.45, test_pearson=0.0100, final_loss=36455.95, alpha=0.7869
Visual: MLP_sub=good: c^1 linear, c^2 quadratic matching GT well, MLP_node=good: linear with correct slopes (type 0 λ=0.001, type 1 λ=0.002)
Mutation: coeff_MLP_sub_norm: 1.0 -> 2.0
Parent rule: Node 21, testing if stronger MLP_sub scale pinning improves alpha and R²
Observation: R²=0.62 is BEST THIS BATCH but still below Iter 21 (0.69). Fewest outliers=14 (tied best). Slope=0.99 excellent (closest to 1.0). But alpha=0.79 worse than Iter 21 (0.85). Stronger sub_norm HURT alpha (counterintuitive). MLP_node IS ACTIVE!
Next: parent=21

## Iter 31: partial
Node: id=31, parent=21
Mode/Strategy: explore (lower lr_k)
Config: seed=42, lr_k=0.003, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.5298, trimmed_R2=0.9680, n_outliers=21, slope=0.9658, test_R2=-983723.51, test_pearson=0.1096, final_loss=33699.31, alpha=0.8074
Visual: MLP_sub=good: c^1 linear, c^2 quadratic matching GT well, MLP_node=good: linear with correct slopes (type 0 λ=0.001, type 1 λ=0.002)
Mutation: lr_k: 0.005 -> 0.003
Parent rule: Node 21, testing if finer k updates near convergence improves R²
Observation: R²=0.53 is WORSE than Iter 21 (0.69). Lower lr_k=0.003 slowed convergence too much. lr_k=0.005 remains optimal. More outliers (21). MLP_node IS ACTIVE with correct linear shapes.
Next: parent=21

## Iter 32: partial
Node: id=32, parent=21
Mode/Strategy: principle-test
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0, coeff_MLP_sub_diff=10
Metrics: rate_constants_R2=0.4090, trimmed_R2=0.9556, n_outliers=21, slope=0.9686, test_R2=-740318.00, test_pearson=0.2092, final_loss=34107.95, alpha=0.8172
Visual: MLP_sub=good: c^1 linear, c^2 quadratic matching GT well, MLP_node=good: linear with correct slopes (type 0 λ=0.001, type 1 λ=0.002)
Mutation: coeff_MLP_sub_diff: 5 -> 10. Testing principle: "Stronger monotonicity fixes MLP_sub c^2 shape"
Parent rule: Node 21, testing if stronger monotonicity constraint fixes the c^2 sublinear issue
Observation: R²=0.41 is WORST in batch, well below Iter 21 (0.69). Stronger monotonicity HURT R² significantly. test_pearson=0.21 is best in batch (but still poor). Principle REFUTED: stronger sub_diff does NOT improve R² or c^2 shape.
Next: parent=21

---

### Batch 8 Summary

**Results Table:**
| Slot | Iter | Mutation | R² | outliers | alpha | slope | test_pearson |
|------|------|----------|-----|----------|-------|-------|--------------|
| 0 | 29 | L1=0.0 | 0.5066 | 19 | 0.83 | 0.97 | 0.10 |
| 1 | 30 | sub_norm=2.0 | **0.6191** | **14** | 0.79 | **0.99** | 0.01 |
| 2 | 31 | lr_k=0.003 | 0.5298 | 21 | 0.81 | 0.97 | 0.11 |
| 3 | 32 | sub_diff=10 | 0.4090 | 21 | 0.82 | 0.97 | **0.21** |

**Key Findings:**
1. **Removing L1 with aug=4000 STILL hurts R²** — R²=0.51, confirms L1=0.0 + long training don't combine well
2. **Stronger sub_norm=2.0 achieved best R² this batch (0.62)** but still below Iter 21 (0.69). Fewest outliers (14) and best slope (0.99)
3. **Lower lr_k=0.003 slowed convergence too much** — R²=0.53, lr_k=0.005 remains optimal
4. **Stronger monotonicity (sub_diff=10) HURT R² significantly** — R²=0.41 is worst in batch

**IMPORTANT VISUAL FINDING:**
- **MLP_node is now ACTIVE in ALL 4 configs!** Shows correct linear homeostasis with type differentiation
- This is a change from batch 7 where MLP_node was flat in all configs
- The learned MLP_node shows correct λ=0.001 for type 0 and λ=0.002 for type 1
- MLP_sub shapes are good: c^1 linear, c^2 quadratic

**Critical Insights:**
- **Iter 21 remains best R²=0.6896** — no batch 8 experiments improved upon it
- **MLP_node activation doesn't correlate with better R²** — all batch 8 configs have active MLP_node but lower R² than batch 7
- **sub_norm=2.0 gave best batch results** — fewest outliers, best slope, but still below Iter 21

**Principle Updates:**
- CONFIRMED: "L1=0.0 + longer training don't combine well" — Iter 29 confirms Iter 22 finding
- REFUTED: "Lower lr_k=0.003 gives finer convergence" — convergence was too slow
- REFUTED: "Stronger sub_diff=10 fixes c^2 shape" — actually hurt R² significantly
- NEW: "MLP_node can activate without improving R²" — batch 8 shows active MLP_node but worse R² than batch 7

**Best Node Remains:** Iter 21 (R²=0.6896, outliers=16, alpha=0.85)

### Batch 9 (Iter 33-36): Testing alternative approaches at R²=0.69 plateau

Base config from Iter 21: lr_k=0.005, lr_node=0.001, lr_sub=0.0005, data_augmentation_loop=4000, coeff_MLP_sub_norm=1.0, coeff_MLP_node_L1=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0

**Slot 0 (id=33)**: Combine sub_norm=2.0 with intermediate training
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=3500, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=2.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: coeff_MLP_sub_norm: 1.0 -> 2.0, data_augmentation_loop: 4000 -> 3500
- Parent rule: Node 21, testing if sub_norm=2.0 (best outliers in batch 8) works better with intermediate training

**Slot 1 (id=34)**: Recurrent training with optimal config
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0, recurrent_training=true, time_step=2
- Mutation: recurrent_training: false -> true, time_step: 1 -> 2
- Parent rule: Node 21, testing if multi-step rollout breaks R² plateau (previously tested without k_floor in Iter 13)

**Slot 2 (id=35)**: Higher lr_sub to compensate for constraint
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: learning_rate_sub: 0.0005 -> 0.001
- Parent rule: Node 21, testing if higher lr_sub helps MLP_sub learn faster

**Slot 3 (id=36)**: Intermediate training (boundary probe)
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=3500, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: data_augmentation_loop: 4000 -> 3500. Testing principle: "aug=4000 is optimal"
- Parent rule: Node 21, boundary probe to confirm aug=4000 optimality

Rationale: Batch 8 showed sub_norm=2.0 gave fewest outliers (14) and best slope (0.99). This batch explores: (1) sub_norm=2.0 with less training to avoid overfitting, (2) recurrent training with full optimal config, (3) higher lr_sub, (4) intermediate training to probe aug optimum.

---

## Iter 33: partial
Node: id=33, parent=21
Mode/Strategy: exploit (sub_norm=2.0 + intermediate training)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=3500, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=2.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.5213, trimmed_R2=0.9520, n_outliers=21, slope=0.9568, test_R2=-1083810.44, test_pearson=0.0253, final_loss=32653.05, alpha=0.7654
Visual: MLP_sub=partial: c^1 linear close to GT, c^2 sublinear (diverges from GT quadratic at high c), MLP_node=bad: flat lines at 0 (homeostasis not learned)
Mutation: coeff_MLP_sub_norm: 1.0 -> 2.0, data_augmentation_loop: 4000 -> 3500
Parent rule: Node 21, testing if sub_norm=2.0 works better with intermediate training
Observation: R²=0.52 is worse than both Iter 21 (0.69) and Iter 30 (0.62 with sub_norm=2.0+aug=4000). Combining sub_norm=2.0 with shorter training didn't help. MLP_node remained flat. More outliers (21 vs 14 for Iter 30).
Next: parent=35

## Iter 34: partial
Node: id=34, parent=21
Mode/Strategy: exploit (recurrent training)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0, recurrent_training=true, time_step=2
Metrics: rate_constants_R2=0.4779, trimmed_R2=0.9706, n_outliers=21, slope=0.9674, test_R2=-3.80, test_pearson=0.6179, final_loss=35816.24, alpha=0.8012
Visual: MLP_sub=partial: c^1 linear close to GT, c^2 sublinear (diverges from GT quadratic), MLP_node=bad: flat lines at 0 (homeostasis not learned)
Mutation: recurrent_training: false -> true, time_step: 1 -> 2
Parent rule: Node 21, testing if multi-step rollout breaks R² plateau with optimal config
Observation: R²=0.48 is WORSE than Iter 21 (0.69). Recurrent training HURT R² even with k_floor (confirming Iter 13 result). test_pearson=0.62 is good but k recovery worse. 2.3x slower (148 min vs 65 min). MLP_node still flat. Recurrent training principle REFUTED again.
Next: parent=35

## Iter 35: partial
Node: id=35, parent=21
Mode/Strategy: explore (higher lr_sub)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.7262, trimmed_R2=0.9649, n_outliers=15, slope=0.9877, test_R2=-733282.72, test_pearson=0.1417, final_loss=35897.11, alpha=0.8277
Visual: MLP_sub=partial: c^1 linear close to GT, c^2 sublinear (diverges from GT quadratic at high c), MLP_node=bad: flat lines at 0 (homeostasis not learned)
Mutation: learning_rate_sub: 0.0005 -> 0.001
Parent rule: Node 21, testing if higher lr_sub helps MLP_sub learn faster
Observation: **NEW BEST R²=0.7262!** (+5% over Iter 21). Higher lr_sub=0.001 improved R² from 0.69 to 0.73. Fewest outliers (15, tied with Iter 27). Slope=0.99 excellent. MLP_node still flat but R² improved! Key finding: lr_sub was too low previously.
Next: parent=35

## Iter 36: partial
Node: id=36, parent=21
Mode/Strategy: principle-test
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.0005, batch_size=8, n_epochs=1, data_augmentation_loop=3500, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.5440, trimmed_R2=0.9607, n_outliers=22, slope=0.9744, test_R2=-190398.32, test_pearson=0.1425, final_loss=33193.24, alpha=0.8144
Visual: MLP_sub=partial: c^1 linear close to GT, c^2 sublinear (diverges from GT quadratic), MLP_node=bad: flat lines at 0 (homeostasis not learned)
Mutation: data_augmentation_loop: 4000 -> 3500. Testing principle: "aug=4000 is optimal"
Parent rule: Node 21, boundary probe to confirm aug=4000 optimality
Observation: R²=0.54 is WORSE than Iter 21 (0.69). Reducing aug from 4000→3500 hurt R² significantly. Principle CONFIRMED: aug=4000 is optimal (or at least better than 3500). MLP_node still flat.
Next: parent=35

---

### Batch 9 Summary

**Results Table:**
| Slot | Iter | Mutation | R² | outliers | alpha | slope | test_pearson |
|------|------|----------|-----|----------|-------|-------|--------------|
| 0 | 33 | sub_norm=2.0 + aug=3500 | 0.5213 | 21 | 0.77 | 0.96 | 0.03 |
| 1 | 34 | recurrent time_step=2 | 0.4779 | 21 | 0.80 | 0.97 | **0.62** |
| 2 | 35 | lr_sub=0.001 (2x) | **0.7262** | **15** | 0.83 | **0.99** | 0.14 |
| 3 | 36 | aug=3500 (boundary) | 0.5440 | 22 | 0.81 | 0.97 | 0.14 |

**Key Findings:**
1. **BREAKTHROUGH: Higher lr_sub=0.001 achieved NEW BEST R²=0.7262** (+5% over Iter 21's 0.69)
2. **Recurrent training with optimal config STILL hurt R²** — confirms this approach doesn't work (Iter 13, 34)
3. **sub_norm=2.0 + shorter training hurt R²** — needs aug=4000 to work
4. **aug=3500 boundary probe confirms aug=4000 is better** — principle confirmed

**Critical Insight:**
- **lr_sub was the missing piece!** Doubling from 0.0005 → 0.001 broke the R² plateau
- Faster MLP_sub learning allows better coordination with k learning
- All 4 configs have MLP_node flat, but Iter 35 achieved best R² anyway

**Principle Updates:**
- NEW: **lr_sub=0.001 is better than 0.0005** — significant R² improvement
- CONFIRMED: "Recurrent training doesn't help" — Iter 34 confirms with optimal config
- CONFIRMED: "aug=4000 is optimal" — aug=3500 hurt R²

**Best Node: Iter 35 (R²=0.7262, outliers=15, alpha=0.83, slope=0.99)**

>>> BLOCK 3 END, 12/12 iterations complete <<<

---

## Block 4: Exploiting lr_sub=0.001 Breakthrough

### Batch 10 (Iter 37-40): Pushing Beyond R²=0.73

Base config from Iter 35: lr_k=0.005, lr_node=0.001, lr_sub=0.001, data_augmentation_loop=4000, coeff_MLP_sub_norm=1.0, coeff_MLP_node_L1=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0

**Slot 0 (id=37)**: Combine lr_sub=0.001 with sub_norm=2.0
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=2.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: coeff_MLP_sub_norm: 1.0 -> 2.0
- Parent rule: Node 35 (new best R²=0.7262), testing if stronger sub_norm combines with higher lr_sub for more outlier reduction

**Slot 1 (id=38)**: Even higher lr_sub
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.002, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: learning_rate_sub: 0.001 -> 0.002
- Parent rule: Node 35, testing if further increasing lr_sub pushes R² higher

**Slot 2 (id=39)**: lr_sub=0.001 + lr_node=0.002
- Config: seed=42, lr_k=0.005, lr_node=0.002, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: learning_rate_node: 0.001 -> 0.002
- Parent rule: Node 35, testing if higher lr_node helps with the new lr_sub setting

**Slot 3 (id=40)**: lr_sub=0.001 + L1=0.0
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=0.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: coeff_MLP_node_L1: 1.0 -> 0.0. Testing principle: "L1=0.0 helps with higher lr_sub"
- Parent rule: Node 35, testing if removing MLP_node L1 works better with higher lr_sub (unlike at lr_sub=0.0005)

Rationale: Iter 35's lr_sub=0.001 achieved new best R²=0.7262. This batch explores 4 variations: (1) combine with sub_norm=2.0 for outlier reduction, (2) try lr_sub=0.002 to push further, (3) test lr_node interaction, (4) test if L1=0 behaves differently at higher lr_sub.

---

## Iter 37: partial
Node: id=37, parent=35
Mode/Strategy: exploit (lr_sub=0.001 + sub_norm=2.0)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=2.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.5882, trimmed_R2=0.9554, n_outliers=16, slope=0.9758, test_R2=-671363.78, test_pearson=0.1650, final_loss=32234.77, alpha=0.8816, MLP_node_slope_0=0.000000, MLP_node_gt_slope_0=-0.001000, MLP_node_slope_1=0.000000, MLP_node_gt_slope_1=-0.002000, embedding_cluster_acc=0.5100, embedding_n_clusters=1
Visual: MLP_sub=partial: c^1 linear, c^2 sublinear, MLP_node=bad: flat lines at 0 (homeostasis not learned)
Mutation: coeff_MLP_sub_norm: 1.0 -> 2.0
Parent rule: Node 35, testing if stronger sub_norm combines with higher lr_sub
Observation: R²=0.59 is WORSE than Iter 35 (0.73). sub_norm=2.0 hurt R² even with lr_sub=0.001. Alpha=0.88 is best this batch but R² dropped significantly. sub_norm=2.0 consistently hurts R² despite improving alpha.
Next: parent=35

## Iter 38: partial
Node: id=38, parent=35
Mode/Strategy: exploit (even higher lr_sub)
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.002, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.5176, trimmed_R2=0.9593, n_outliers=19, slope=0.9624, test_R2=-993.35, test_pearson=0.4198, final_loss=30664.99, alpha=0.8018, MLP_node_slope_0=0.000000, MLP_node_gt_slope_0=-0.001000, MLP_node_slope_1=0.000000, MLP_node_gt_slope_1=-0.002000, embedding_cluster_acc=0.5000, embedding_n_clusters=1
Visual: MLP_sub=partial: c^1 linear, c^2 sublinear, MLP_node=bad: flat lines at 0 (homeostasis not learned)
Mutation: learning_rate_sub: 0.001 -> 0.002
Parent rule: Node 35, testing if further increasing lr_sub pushes R² higher
Observation: R²=0.52 is WORSE than Iter 35 (0.73). lr_sub=0.002 is TOO HIGH - hurts R² significantly. Test_pearson=0.42 is best but k recovery worse. lr_sub=0.001 is optimal, not 0.002.
Next: parent=35

## Iter 39: partial
Node: id=39, parent=35
Mode/Strategy: explore (lr_node=0.002 with lr_sub=0.001)
Config: seed=42, lr_k=0.005, lr_node=0.002, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.6537, trimmed_R2=0.9665, n_outliers=20, slope=0.9966, test_R2=-582138.95, test_pearson=0.2212, final_loss=37049.07, alpha=0.8148, MLP_node_slope_0=0.000000, MLP_node_gt_slope_0=-0.001000, MLP_node_slope_1=0.000000, MLP_node_gt_slope_1=-0.002000, embedding_cluster_acc=0.5200, embedding_n_clusters=1
Visual: MLP_sub=partial: c^1 linear, c^2 sublinear, MLP_node=bad: flat lines at 0 (homeostasis not learned despite higher lr_node)
Mutation: learning_rate_node: 0.001 -> 0.002
Parent rule: Node 35, testing if higher lr_node helps with the new lr_sub setting
Observation: R²=0.65 is WORSE than Iter 35 (0.73) but second best this batch. Slope=1.00 (perfect) is best ever! lr_node=0.002 still didn't activate MLP_node. Higher lr_node doesn't help MLP_node learn.
Next: parent=35

## Iter 40: partial
Node: id=40, parent=35
Mode/Strategy: principle-test
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=0.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
Metrics: rate_constants_R2=0.6622, trimmed_R2=0.9721, n_outliers=21, slope=0.9650, test_R2=-1095706.73, test_pearson=0.2118, final_loss=37006.91, alpha=0.8305, MLP_node_slope_0=0.000000, MLP_node_gt_slope_0=-0.001000, MLP_node_slope_1=0.000000, MLP_node_gt_slope_1=-0.002000, embedding_cluster_acc=0.5100, embedding_n_clusters=1
Visual: MLP_sub=partial: c^1 linear, c^2 sublinear, MLP_node=bad: flat lines at 0 (homeostasis still not learned even with L1=0)
Mutation: coeff_MLP_node_L1: 1.0 -> 0.0. Testing principle: "L1=0.0 + longer training hurts R²"
Parent rule: Node 35, testing if removing MLP_node L1 works better with higher lr_sub
Observation: R²=0.66 is WORSE than Iter 35 (0.73). L1=0.0 STILL hurts R² even with lr_sub=0.001 (confirming principle 7). MLP_node remains flat despite L1=0. Best trimmed_R2=0.97. The L1=0 + long training combination consistently hurts.
Next: parent=35

---

### Batch 10 Summary

**Results Table:**
| Slot | Iter | Mutation | R² | outliers | alpha | slope | test_pearson |
|------|------|----------|-----|----------|-------|-------|--------------|
| 0 | 37 | sub_norm=2.0 | 0.5882 | **16** | **0.88** | 0.98 | 0.17 |
| 1 | 38 | lr_sub=0.002 | 0.5176 | 19 | 0.80 | 0.96 | **0.42** |
| 2 | 39 | lr_node=0.002 | 0.6537 | 20 | 0.81 | **1.00** | 0.22 |
| 3 | 40 | L1=0.0 | 0.6622 | 21 | 0.83 | 0.97 | 0.21 |

**Key Findings:**
1. **NONE of the variations beat Iter 35's R²=0.7262** — lr_sub=0.001 alone remains the best config
2. **sub_norm=2.0 HURT R² despite improving alpha** — R² dropped from 0.73 to 0.59, but alpha improved to 0.88
3. **lr_sub=0.002 is TOO HIGH** — R² dropped from 0.73 to 0.52, lr_sub=0.001 is optimal
4. **lr_node=0.002 didn't activate MLP_node** — MLP_node still flat, slope=1.00 is best ever
5. **L1=0.0 STILL hurts with lr_sub=0.001** — principle 7 confirmed again (L1=0 + long training hurts)

**Critical Insights:**
- **Iter 35 remains the best overall config** — no combination of lr_sub=0.001 with other changes improved R²
- **All MLP_node remained flat** — MLP_node not learning homeostasis is NOT the limiting factor for k recovery
- **Alpha vs R² tradeoff**: sub_norm=2.0 improves alpha but hurts R²; sub_norm=1.0 optimal for R²
- **lr_sub optimal at 0.001**: lower (0.0005) and higher (0.002) both hurt R²

**Principle Updates:**
- CONFIRMED: "L1=0.0 + longer training hurts R²" — Iter 40 adds evidence
- NEW: "lr_sub=0.002 is too high" — optimal is 0.001, not higher
- CONFIRMED: "sub_norm=2.0 improves alpha but hurts R²" — Iter 37 adds evidence

**Best Node Remains: Iter 35 (R²=0.7262, outliers=15, alpha=0.83, slope=0.99)**

### Batch 11 (Iter 41-44): Alternative approaches to break R²=0.73

Base config from Iter 35: lr_k=0.005, lr_node=0.001, lr_sub=0.001, data_augmentation_loop=4000, coeff_MLP_sub_norm=1.0, coeff_MLP_node_L1=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0

**Slot 0 (id=41)**: Different seed to verify reproducibility
- Config: seed=123, lr_k=0.005, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: seed: 42 -> 123
- Parent rule: Node 35, testing if Iter 35's R²=0.7262 is seed-dependent or reproducible

**Slot 1 (id=42)**: Slightly longer training (aug=4500)
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4500, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: data_augmentation_loop: 4000 -> 4500
- Parent rule: Node 35, probing if slightly more training can push R² beyond 0.73 (known aug=5000 hurt)

**Slot 2 (id=43)**: Reduced monotonicity constraint
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0, coeff_MLP_sub_diff=3
- Mutation: coeff_MLP_sub_diff: 5 -> 3
- Parent rule: Node 35, testing if less monotonicity constraint lets MLP_sub learn better

**Slot 3 (id=44)**: Slightly higher lr_k with lr_sub=0.001
- Config: seed=42, lr_k=0.007, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: learning_rate_k: 0.005 -> 0.007. Testing principle: "lr_k=0.005 is optimal"
- Parent rule: Node 35, testing if slightly faster k learning helps with lr_sub=0.001 (lr_k=0.01 too high previously)

Rationale: Batch 10 showed that lr_sub=0.001 combinations all hurt R². This batch explores: (1) seed reproducibility, (2) fine-tune training length, (3) less monotonicity, (4) slightly higher lr_k.

---

## Iter 41: partial
Node: id=41, parent=35
Mode/Strategy: exploit
Config: seed=123, lr_k=0.005, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0
Metrics: rate_constants_R2=0.4872, trimmed_R2=0.9558, n_outliers=21, slope=0.9638, test_R2=-1188216.2, test_pearson=0.0306, final_loss=36369.22, alpha=0.7109, MLP_node_slope_0=-0.0000, MLP_node_gt_slope_0=-0.0010, MLP_node_slope_1=-0.0000, MLP_node_gt_slope_1=-0.0020, embedding_cluster_acc=0.51, embedding_n_clusters=1
Visual: MLP_sub=partial: c^1 reasonably linear (α=0.64) but c^2 flat not quadratic (α=0.011), MLP_node=inactive: flat lines at 0
Mutation: seed: 42 -> 123
Parent rule: test reproducibility of Iter 35 best result
Observation: **SEED SENSITIVITY CONFIRMED** — R² dropped from 0.73 to 0.49 with different seed; Iter 35 result may have been lucky; alpha dropped to 0.71
Next: parent=42

## Iter 42: partial
Node: id=42, parent=35
Mode/Strategy: exploit
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4500, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0
Metrics: rate_constants_R2=0.6896, trimmed_R2=0.9674, n_outliers=16, slope=0.9771, test_R2=-92583.7, test_pearson=0.3066, final_loss=33202.49, alpha=0.9382, MLP_node_slope_0=0.0000, MLP_node_gt_slope_0=-0.0010, MLP_node_slope_1=0.0000, MLP_node_gt_slope_1=-0.0020, embedding_cluster_acc=0.51, embedding_n_clusters=1
Visual: MLP_sub=partial: c^1 close to linear (α=0.88) but c^2 compressed (α=0.015), MLP_node=inactive: flat lines at 0
Mutation: data_augmentation_loop: 4000 -> 4500
Parent rule: probe aug=4500 boundary (slightly longer training)
Observation: Aug=4500 gives R²=0.69 same as Iter 17 (aug=4000 w/o lr_sub=0.001); **BEST ALPHA=0.94** in this batch; aug beyond 4000 shows diminishing returns for R² but improves alpha
Next: parent=42

## Iter 43: partial
Node: id=43, parent=35
Mode/Strategy: explore
Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, coeff_MLP_sub_diff=3
Metrics: rate_constants_R2=0.6080, trimmed_R2=0.9733, n_outliers=18, slope=0.9894, test_R2=-6691.75, test_pearson=0.3337, final_loss=30423.01, alpha=0.8867, MLP_node_slope_0=0.0000, MLP_node_gt_slope_0=-0.0010, MLP_node_slope_1=0.0000, MLP_node_gt_slope_1=-0.0020, embedding_cluster_acc=0.50, embedding_n_clusters=1
Visual: MLP_sub=partial: c^1 close to linear (α=0.85), c^2 compressed (α=0.015), MLP_node=inactive: flat lines at 0
Mutation: coeff_MLP_sub_diff: 5 -> 3
Parent rule: explore lower monotonicity constraint (let MLP learn more freely)
Observation: Less monotonicity (sub_diff=3) hurt R² from 0.73 to 0.61; strong monotonicity (sub_diff=5) confirmed as better; best trimmed_R²=0.97
Next: parent=42

## Iter 44: partial
Node: id=44, parent=35
Mode/Strategy: principle-test
Config: seed=42, lr_k=0.007, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4000, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0
Metrics: rate_constants_R2=0.5931, trimmed_R2=0.9666, n_outliers=19, slope=0.9867, test_R2=-37174.42, test_pearson=0.2004, final_loss=33509.80, alpha=0.7889, MLP_node_slope_0=0.0000, MLP_node_gt_slope_0=-0.0010, MLP_node_slope_1=0.0000, MLP_node_gt_slope_1=-0.0020, embedding_cluster_acc=0.50, embedding_n_clusters=1
Visual: MLP_sub=partial: c^1 close to linear (α=0.79), c^2 compressed (α=0.013), MLP_node=inactive: flat lines at 0
Mutation: learning_rate_k: 0.005 -> 0.007. Testing principle: "lr_k=0.005 is appropriate"
Parent rule: test if slightly higher lr_k=0.007 helps with lr_sub=0.001
Observation: lr_k=0.007 hurts R² (0.59 vs 0.73), confirming lr_k=0.005 is optimal; principle CONFIRMED
Next: parent=42

---

### Batch 11 Summary

**Results Table:**
| Slot | Iter | Mutation | R² | outliers | alpha | slope | test_pearson |
|------|------|----------|-----|----------|-------|-------|--------------|
| 0 | 41 | seed=123 | 0.4872 | 21 | 0.71 | 0.96 | 0.03 |
| 1 | 42 | aug=4500 | 0.6896 | **16** | **0.94** | 0.98 | **0.31** |
| 2 | 43 | sub_diff=3 | 0.6080 | 18 | 0.89 | **0.99** | 0.33 |
| 3 | 44 | lr_k=0.007 | 0.5931 | 19 | 0.79 | 0.99 | 0.20 |

**Key Findings:**
1. **SEED SENSITIVITY IS MAJOR** — R² dropped from 0.73 to 0.49 with seed=123; Iter 35's R²=0.73 was partially lucky
2. **aug=4500 matches aug=4000** — R²=0.69 same as original aug=4000 runs; best alpha=0.94 in batch
3. **Less monotonicity hurts R²** — sub_diff=3 gave R²=0.61 vs sub_diff=5's 0.73
4. **lr_k=0.007 hurts** — confirms lr_k=0.005 is optimal

**Critical Insights:**
- **True baseline R² is ~0.49-0.69, not 0.73** — seed variability explains some of Iter 35's apparent breakthrough
- **Iter 42 (aug=4500) is most stable** — R²=0.69 with best alpha=0.94, likely more representative than seed=42 luck
- **All configs still have MLP_node flat** — homeostasis learning is not the bottleneck

**Principle Updates:**
- NEW: **Seed variability significant (~0.2 R² range)** — results should be interpreted with seed uncertainty
- CONFIRMED: "lr_k=0.005 is optimal" — lr_k=0.007 hurt R²
- CONFIRMED: "sub_diff=5 is optimal" — lower monotonicity hurts

**Best Node This Batch: Iter 42 (R²=0.6896, outliers=16, alpha=0.94, slope=0.98)**

### Batch 12 (Iter 45-48): Exploring alternative paths

Base config from Iter 42: lr_k=0.005, lr_node=0.001, lr_sub=0.001, data_augmentation_loop=4500, coeff_MLP_sub_norm=1.0, coeff_MLP_node_L1=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0, coeff_MLP_sub_diff=5

**Slot 0 (id=45)**: Stronger monotonicity for stability
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4500, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0, coeff_MLP_sub_diff=7
- Mutation: coeff_MLP_sub_diff: 5 -> 7
- Parent rule: Node 42, probing slightly stronger monotonicity for more stable MLP_sub

**Slot 1 (id=46)**: Slightly lower lr_k for stability
- Config: seed=42, lr_k=0.004, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4500, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: learning_rate_k: 0.005 -> 0.004
- Parent rule: Node 42, testing if slightly lower lr_k gives more stable k recovery

**Slot 2 (id=47)**: Wider MLP_sub architecture
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.001, batch_size=8, n_epochs=1, data_augmentation_loop=4500, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0, hidden_dim_sub=128
- Mutation: hidden_dim_sub: 64 -> 128
- Parent rule: Node 42, exploring if more MLP_sub capacity helps learn c^s better

**Slot 3 (id=48)**: Larger batch size for gradient stability
- Config: seed=42, lr_k=0.005, lr_node=0.001, lr_sub=0.001, batch_size=16, n_epochs=1, data_augmentation_loop=4500, coeff_MLP_node_L1=1.0, coeff_MLP_sub_norm=1.0, coeff_k_floor=1.0, k_floor_threshold=-2.0
- Mutation: batch_size: 8 -> 16. Testing principle: "batch_size=8 is optimal"
- Parent rule: Node 42, testing if larger batches provide more stable gradients

Rationale: Batch 11 revealed significant seed sensitivity (~0.2 R² variance). This batch explores stabilization: (1) stronger monotonicity, (2) lower lr_k, (3) wider MLP_sub, (4) larger batch size.

