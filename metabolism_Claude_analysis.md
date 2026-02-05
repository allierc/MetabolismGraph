# Experiment Log: metabolism_Claude

## Block 1

### Iteration 0 (Baseline)
**Config**: lr=1e-3, lr_S=1e-3, coeff_S_L1=0, coeff_S_integer=0, coeff_mass=0
**Results**: stoichiometry_R2=0.037, test_pearson=0.205, final_loss=59.92
**Analysis**: No regularization leads to very poor stoichiometry recovery. The model can fit dynamics but doesn't learn the correct sparse integer structure of S.

### Iteration 1
**Hypothesis**: Adding regularization will improve stoichiometry recovery by encoding prior knowledge:
- L1 promotes sparsity (S is ~96% zero)
- Integer penalty encourages values near {-2,-1,0,1,2}
- Mass conservation constrains column sums to be zero

**Config changes**:
- coeff_S_L1: 0 → 0.001
- coeff_S_integer: 0 → 0.001
- coeff_mass_conservation: 0 → 0.001

**Expected outcome**: Improved stoichiometry_R2 as the model is guided toward physically meaningful S values.

