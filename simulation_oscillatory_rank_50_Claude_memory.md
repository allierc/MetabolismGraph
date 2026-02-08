# Working Memory: simulation_oscillatory_rank_50 (parallel)

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | n_metabolites | n_reactions | n_frames | noise | eff_rank | Best R² | Optimal lr_k | Optimal lr_node | Key finding |
| ----- | ------------- | ----------- | -------- | ----- | -------- | ------- | ------------ | --------------- | ----------- |

### Established Principles

### Open Questions
- What is the optimal lr_k for high-rank (rank~50) regime?
- How does lr_node/lr_sub balance affect rate constant recovery?
- Does coeff_MLP_sub_norm help break scale ambiguity?

---

## Previous Block Summary

---

## Current Block (Block 1)

### Block Info
- Regime: simulation_oscillatory_rank_50 (S frozen, learning k)
- Simulation: 100 metabolites, 256 reactions, 2880 frames, log_k in [-2.0, -1.0]
- Primary metric: rate_constants_R2

### Hypothesis
Starting exploration based on previous findings from original oscillatory config (lr_k=0.005, lr_node=0.001, lr_sub=0.0005). Need to verify these translate to higher-rank regime.

### Iterations This Block

#### Iter 1-4 (Batch 1 - Initial spread)
Planned mutations:
- Slot 0 (baseline): lr_k=0.005, lr_node=0.001, lr_sub=0.0005 (unchanged from base)
- Slot 1 (explore lr_k high): lr_k=0.01, lr_node=0.001, lr_sub=0.0005
- Slot 2 (explore lr_node high): lr_k=0.005, lr_node=0.002, lr_sub=0.0005
- Slot 3 (explore lr_sub low): lr_k=0.005, lr_node=0.001, lr_sub=0.0002

Rationale: Spread across the 3 key learning rate dimensions to find initial signal. Baseline uses previously-optimal params; slot 1 tests if higher lr_k helps; slot 2 tests if MLP_node needs faster learning; slot 3 tests if slower MLP_sub learning prevents compensation.

### Emerging Observations
- **All 4 initial configs failed** (rate_constants_R2 < 0.05, test_R2 massively negative)
- **MLP_node is completely dead** (flat at 0) across all configs — coeff_MLP_node_L1=1.0 may be too strong
- **MLP_sub c^2 curve is linear** instead of quadratic — wrong shape being learned
- **alpha ~0.45-0.52** (should be ~1.0) — scale ambiguity present
- Learning rates from original oscillatory config (rank~24) do not transfer to rank~50 regime
- test_pearson near 0 indicates training did not learn dynamics at all

#### Batch 1 Results (Iter 1-4)
| Slot | lr_k | lr_node | lr_sub | R² | trimmed_R² | outliers | alpha |
|------|------|---------|--------|-----|------------|----------|-------|
| 0 | 0.005 | 0.001 | 0.0005 | 0.044 | 0.923 | 43 | 0.52 |
| 1 | 0.01 | 0.001 | 0.0005 | 0.027 | 0.913 | 47 | 0.47 |
| 2 | 0.005 | 0.002 | 0.0005 | 0.044 | 0.922 | 45 | 0.47 |
| 3 | 0.005 | 0.001 | 0.0002 | 0.031 | 0.916 | 53 | 0.45 |

Best so far: Slot 0 and 2 tied at R²=0.044 (both failed)

#### Batch 2 Planned (Iter 5-8)
- Slot 0: coeff_MLP_node_L1: 1.0 -> 0.1 (let MLP_node learn)
- Slot 1: coeff_MLP_sub_norm: 0.0 -> 1.0 (pin MLP_sub scale)
- Slot 2: lr_k: 0.005 -> 0.002 (more conservative)
- Slot 3: data_augmentation_loop: 1000 -> 2000 (longer training)

