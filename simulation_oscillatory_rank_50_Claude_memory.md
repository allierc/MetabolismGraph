# Working Memory: simulation_oscillatory_rank_50 (parallel)

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | n_metabolites | n_reactions | n_frames | noise | eff_rank | Best R² | Optimal lr_k | Optimal lr_sub | Key finding |
| ----- | ------------- | ----------- | -------- | ----- | -------- | ------- | ------------ | -------------- | ----------- |
| 1 | 100 | 256 | 2880 | none | ~50 | 0.0668 | 0.005 | 0.0005 | coeff_MLP_sub_norm=1.0 essential for correct MLP shapes |
| 2 | 100 | 256 | 2880 | none | ~50 | 0.6896 | 0.005 | 0.0005 | k_floor=1.0 + aug=4000 breakthrough! |
| 3 | 100 | 256 | 2880 | none | ~50 | 0.7262 | 0.005 | 0.001 | lr_sub=0.001 broke R² plateau! |
| 4 | 100 | 256 | 2880 | none | ~50 | **0.7358** | 0.005 | 0.001 | sub_diff=7 NEW BEST! |
| 5 | 100 | 256 | 2880 | none | ~50 | 0.7358 | 0.005 | 0.001 | sub_diff=7 confirmed optimal, more seed-robust |

### Established Principles
1. **coeff_MLP_sub_norm=1.0 is essential** — enables correct MLP shapes: c^2 becomes quadratic, MLP_node becomes active (evidence: Iter 6 best, all batch 3 configs work)
2. **Longer training helps R² and alpha (up to aug=4000-4500)** — aug=2000→3000→4000 consistently improves R² (evidence: Iter 8, 9, 12, 17, 21, 42 - VERY STRONG). But aug=5000 hurt R² (Iter 25, 49 - CONFIRMED TWICE). aug=3500 also hurt (Iter 36).
3. **lr_k=0.005 is appropriate** — lower lr_k=0.003/0.004 too slow, higher lr_k=0.007/0.01 hurts R² (evidence: Iter 7, 19, 31, 44, 46)
4. **lr_node=0.005 hurts R²** — high lr_node destabilizes training, R² dropped to 0.017 (evidence: Iter 11)
5. **coeff_k_floor=1.0 is CRITICAL and OPTIMAL** — R² jumped from 0.06 to 0.64 (10x improvement!), k_floor=2.0 too strong (evidence: Iter 14, 17, 18, 24 - VERY STRONG)
6. **k_floor_threshold should match log_k_min** — threshold=-2.5 (below true min) hurt R² from 0.51 to 0.37 (evidence: Iter 20)
7. **L1=0.0 + longer training DON'T combine** — Iter 22 (aug=3000), Iter 29 (aug=4000), Iter 40 (lr_sub=0.001) all hurt R² (evidence: CONFIRMED 3x)
8. **MLP_node activation doesn't correlate with R² improvement** — batch 8 all had active MLP_node but worse R² than batch 7 (evidence: Iter 29-32)
9. **lr_sub=0.001 is optimal** — BREAKTHROUGH, R² improved from 0.69 to 0.73 (evidence: Iter 35); lr_sub=0.002 too high (Iter 38)
10. **sub_norm=2.0 improves alpha but hurts R²** — alpha improved to 0.88 but R² dropped (evidence: Iter 30, 37)
11. **Seed sensitivity is significant (~0.2 R² variance)** — Iter 35 (seed=42) got R²=0.73, Iter 41 (seed=123) got R²=0.49 with identical config (evidence: Iter 41)
12. **sub_diff=7 is optimal** — stronger monotonicity improves R² from 0.69 to 0.74 (evidence: Iter 45 - NEW BEST!); sub_diff=8 too strong (Iter 50), sub_diff=3 too weak (Iter 43)
13. **batch_size=8 is optimal** — batch_size=16 hurts R² (evidence: Iter 48)
14. **Wider/Deeper MLP_sub hurts R²** — hidden_dim_sub=128 allows degenerate solutions (Iter 47), n_layers_sub=4 also hurts (Iter 52)
15. **sub_diff=7 improves seed robustness** — R² with seed=123 improved from 0.49 (sub_diff=5) to 0.66 (sub_diff=7) — gap reduced from 0.24 to 0.08 (evidence: Iter 51 vs Iter 41) - NEW

### Refuted Hypotheses
- "MLP_node needs higher lr_node to learn" — FALSE, lr_node=0.005 hurts (Iter 11), lr_node=0.002 no effect (Iter 26, 39)
- "coeff_MLP_node_L1 prevents MLP_node learning" — FALSE, MLP_node activates with sub_norm=1.0 regardless of L1 setting
- "Recurrent training (time_step>1) breaks degeneracy" — FALSE, no R² improvement, slower (Iter 13, 34 - CONFIRMED TWICE)
- "Smaller MLP architecture helps k recovery" — FALSE, worst R²=0.011 (Iter 15)
- "Stronger monotonicity (sub_diff=10) helps" — FALSE, R² dropped to 0.41 (Iter 32)
- "Tighter k_floor_threshold improves k recovery" — FALSE, threshold=-2.5 hurt R² (Iter 20)
- "Higher lr_k with k_floor safety net works" — FALSE, lr_k=0.01 still hurts (Iter 19), lr_k=0.007 also hurts (Iter 44)
- "Lower lr_sub helps k dominate" — FALSE, no effect when k_floor is active (Iter 23), higher lr_sub actually helps (Iter 35)
- "Stronger k_floor (coeff=2.0) reduces outliers" — FALSE, hurts R² (Iter 24)
- "aug=5000 continues to improve R²" — FALSE, R² dropped from 0.69 to 0.65 (Iter 25), dropped 0.74 to 0.70 (Iter 49)
- "Different seed breaks MLP_node degeneracy" — FALSE, seed=123 same flat MLP_node (Iter 27)
- "Smaller batch_size helps convergence" — FALSE, batch_size=4 hurt R² (Iter 28)
- "Lower lr_k=0.003/0.004 gives finer convergence" — FALSE, convergence too slow (Iter 31, 46)
- "Stronger sub_norm=2.0 improves alpha" — FALSE, alpha=0.79 worse than sub_norm=1.0 (Iter 30)
- "sub_norm=2.0 + shorter training works" — FALSE, R²=0.52 worse than aug=4000 (Iter 33)
- "lr_sub=0.002 improves over 0.001" — FALSE, R² dropped from 0.73 to 0.52 (Iter 38)
- "Combining lr_sub=0.001 with other improvements beats baseline" — FALSE, all combinations hurt (Iter 37-40)
- "Less monotonicity (sub_diff=3) lets MLP learn better" — FALSE, R² dropped from 0.73 to 0.61 (Iter 43)
- "Wider MLP_sub (hidden_dim=128) helps" — FALSE, R² dropped to 0.56 (Iter 47)
- "Larger batch_size=16 gives more stable gradients" — FALSE, R² dropped to 0.56 (Iter 48)
- "sub_diff=8 further improves over sub_diff=7" — FALSE, R² dropped from 0.74 to 0.59 (Iter 50)
- "Deeper MLP_sub (n_layers=4) helps" — FALSE, R² dropped to 0.55 (Iter 52)

### Open Questions
- Is R²=0.74 a fundamental limit for this regime?
- How much of the variance is seed-dependent vs method-dependent?
- Would sub_diff=6 be better than 7?
- Can ensemble averaging across seeds improve stability?
- Would lr_sub=0.0015 (between 0.001 and 0.002) help?

---

## Block 5 Progress (Iter 49-60) - IN PROGRESS

### Best Config So Far (Iter 45)
- lr_k=0.005, lr_node=0.001, lr_sub=0.001
- batch_size=8, data_augmentation_loop=4500
- coeff_MLP_sub_diff=7, coeff_MLP_sub_norm=1.0, coeff_MLP_node_L1=1.0
- coeff_k_floor=1.0, k_floor_threshold=-2.0
- R²=0.7358, outliers=15, alpha=0.90, slope=0.97

### UCB Tree Status
Best node: Node 45 (R²=0.7358, alpha=0.90, outliers=15)
Parent for next batch: Node 45

### Batch 13 (Iter 49-52) Results
| Slot | Iter | Mutation | R² | outliers | alpha | slope |
|------|------|----------|-----|----------|-------|-------|
| 0 | 49 | aug=5000 | 0.6957 | **12** | 0.89 | 0.98 |
| 1 | 50 | sub_diff=8 | 0.5905 | 21 | 0.86 | 0.97 |
| 2 | 51 | seed=123 | 0.6552 | **12** | 0.86 | 0.97 |
| 3 | 52 | n_layers_sub=4 | 0.5450 | 25 | 0.74 | 1.01 |

Key findings: None beat Iter 45. aug=5000 hurts, sub_diff=8 too strong, n_layers=4 hurts. sub_diff=7 improves seed robustness (R²=0.66 vs 0.49 for sub_diff=5).

### Batch 14 (Iter 53-56) Results
| Slot | Iter | Mutation | R² | outliers | alpha | slope |
|------|------|----------|-----|----------|-------|-------|
| 0 | 53 | aug=4250 | 0.6616 | 21 | 0.84 | 0.98 |
| 1 | 54 | sub_diff=6 | 0.5602 | 21 | 0.86 | 1.00 |
| 2 | 55 | seed=123+aug=4000 | **0.7009** | **18** | 0.81 | 0.98 |
| 3 | 56 | lr_sub=0.0015 | 0.5997 | 19 | 0.83 | 0.99 |

Key findings: None beat Iter 45. aug=4250 too short, sub_diff=6 too weak, lr_sub=0.0015 too high. seed=123+aug=4000 best of batch (R²=0.70).

### Batch 15 (Iter 57-60) Strategy
Given the tight optimization bounds discovered, exploring new dimensions:

- **Slot 0 (exploit)**: seed=123 + aug=3500 (even shorter training for seed=123)
- **Slot 1 (exploit)**: seed=99 (try a new seed with optimal config)
- **Slot 2 (explore)**: coeff_MLP_node_L1=0.5 (softer L1 constraint)
- **Slot 3 (principle-test)**: coeff_MLP_sub_norm=0.5 (weaker normalization). Testing principle: "coeff_MLP_sub_norm=1.0 is essential"
