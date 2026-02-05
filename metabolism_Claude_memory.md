# Working Memory: metabolism_Claude

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | n_metabolites | n_reactions | n_frames | noise | eff_rank | Best R² | Optimal lr_S | Optimal L1 | Key finding |
| ----- | ------------- | ----------- | -------- | ----- | -------- | ------- | ------------ | ---------- | ----------- |

### Established Principles

### Open Questions

---

## Previous Block Summary

---

## Current Block (Block 1)

### Block Info
- Simulation: 100 metabolites, 256 reactions, 2880 frames
- Target: Recover stoichiometric matrix S (sparse, integer entries {-2,-1,0,1,2})
- Baseline stoichiometry_R2: 0.037 (with no regularization)

### Hypothesis
Adding regularization (L1 for sparsity, integer penalty, mass conservation) will improve stoichiometry recovery by encoding prior knowledge about the structure of S.

### Iterations This Block
| Iter | lr | lr_S | L1 | integer | mass | sto_R2 | pearson | Notes |
|------|-----|------|-----|---------|------|--------|---------|-------|
| 0 (baseline) | 1e-3 | 1e-3 | 0 | 0 | 0 | 0.037 | 0.205 | No regularization |
| 1 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | - | - | Add all regularization |

### Emerging Observations
- Baseline with no regularization gives very poor S recovery (R2=0.037)
- Need to explore regularization strengths and their interactions

