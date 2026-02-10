# Activity Rank Exploration

## Goal
Increase activity rank(99%) from baseline of 24 (out of 100 metabolites).

## Baseline Config
- n_metabolites=100, n_reactions=256, n_frames=2880, delta_t=0.1
- cycle_fraction=1.0, cycle_length=3 (all 3-cycles, A+B->2B)
- log_k in [-2.5, -1.0], use_mass_action=true, flux_limit=false
- n_metabolite_types=2, lambda=[0.001, 0.002], baselines=[4.0, 6.0]
- concentration_min=1.0, concentration_max=9.0
- **Baseline activity rank(99%) = 24**

## Test Results (single parameter changes)

| Test | Change | Activity Rank | Delta | Notes |
|------|--------|:---:|:---:|-------|
| baseline | (none) | 24 | -- | reference |
| 1 | cycle_fraction 1.0 -> 0.7 | CRASH | -- | Random rxns have coeff=2, c^2*c^2 explodes without flux_limit |
| 2 | log_k_min -2.5 -> -3.5 | 7 | -17 | Slow reactions (k~0.0003) are inert, reducing active count |
| 3 | n_met_types 4 + lambda=[.001,.005,.01,.02] | 19 | -5 | Stronger homeostasis dampens oscillatory modes |
| 4 | lambda 10x [0.01, 0.02] (2 types) | 12 | -12 | Even stronger damping kills more modes |
| 5a | log_k_min -2.5 -> -1.5 | CRASH | -- | Too many fast reactions blow up Euler |
| **5b** | **log_k_min -2.5 -> -2.0** | **50** | **+26** | **BEST. Narrower range = all reactions active** |
| 6a | cycle_length 3 -> 4 | 11 | -13 | Fewer cycles (64 vs 85), fewer modes |
| 6b | cycle_length 3 -> 2 | 2 | -22 | Symmetric pairs = 1D oscillation, all correlate |
| **7** | **n_reactions 256 -> 512** | **47** | **+23** | **More reactions = more overlapping cycles** |
| 8a | n_metabolites 100 -> 200 | 26/200 | +2 abs | Dilutes reaction budget (13% vs 24% ratio) |
| 8b | n_metabolites 100 -> 50 | 25/50 | +1 abs | 50% ratio but same absolute rank |
| **9** | **lambda [0.001,0.002] -> [0.0,0.0]** | **37** | **+13** | **Even tiny homeostasis damps 13 modes** |

## Key Findings

### Three winning parameters (each tested in isolation):
1. **log_k_min = -2.0** (rank 50): Eliminating slow reactions ensures all 256 reactions actively drive dynamics
2. **n_reactions = 512** (rank 47): More reactions = more overlapping cycles per metabolite = higher connectivity
3. **lambda = [0, 0]** (rank 37): Zero homeostasis removes all linear damping

### Pattern: Activity rank is controlled by the number of ACTIVELY CONTRIBUTING reactions per metabolite
- Slow k kills reactions → rank drops (test 2)
- Fewer reactions → rank drops (test 7 shows converse)
- Homeostasis opposes reaction dynamics → effective reaction count drops → rank drops
- Shorter cycles = more correlated pairs → rank drops (test 6b)
- cycle_length=3 is near-optimal for the autocatalytic topology

### Critical insight on rate constant range:
- log_k in [-2.5, -1.0] spans 1.5 decades → ~40% of reactions have k < 0.01, contributing weakly
- log_k in [-2.0, -1.0] spans 1.0 decade → all reactions have k >= 0.01, all contribute
- log_k in [-1.5, -1.0] spans 0.5 decades → all very fast, but total rates overflow Euler step

### Absolute rank vs metabolite count:
- 50 mets: rank 25 (50%)
- 100 mets: rank 24 (24%)
- 200 mets: rank 26 (13%)
- Absolute rank ~25 is independent of n_metabolites, suggesting it's limited by the REACTION topology (85 cycles, 256 reactions), not the metabolite pool size.

## Next: Test combinations of winning parameters
