# Working Memory: simulation_oscillatory_rank_50 (parallel)

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | n_metabolites | n_reactions | n_frames | noise | eff_rank | Best R² | Optimal lr_k | Optimal lr_sub | Key finding |
| ----- | ------------- | ----------- | -------- | ----- | -------- | ------- | ------------ | -------------- | ----------- |
| 1 | 100 | 256 | 2880 | none | ~50 | 0.0668 | 0.005 | 0.0005 | coeff_MLP_sub_norm=1.0 essential for correct MLP shapes |
| 2 | 100 | 256 | 2880 | none | ~50 | 0.6896 | 0.005 | 0.0005 | k_floor=1.0 + aug=4000 breakthrough! |
| 3 | 100 | 256 | 2880 | none | ~50 | **0.7262** | 0.005 | **0.001** | lr_sub=0.001 broke R² plateau! |
| 4 | 100 | 256 | 2880 | none | ~50 | 0.6896 | 0.005 | 0.001 | Seed sensitivity ~0.2 R²; true stable R²~0.69 |

### Established Principles
1. **coeff_MLP_sub_norm=1.0 is essential** — enables correct MLP shapes: c^2 becomes quadratic, MLP_node becomes active (evidence: Iter 6 best, all batch 3 configs work)
2. **Longer training helps R² and alpha (up to aug=4000-4500)** — aug=2000→3000→4000 consistently improves R² (evidence: Iter 8, 9, 12, 17, 21, 42 - VERY STRONG). But aug=5000 hurt R² (Iter 25). aug=3500 also hurt (Iter 36).
3. **lr_k=0.005 is appropriate** — lower lr_k=0.003 too slow, higher lr_k=0.007/0.01 hurts R² (evidence: Iter 7, 19, 31, 44)
4. **lr_node=0.005 hurts R²** — high lr_node destabilizes training, R² dropped to 0.017 (evidence: Iter 11)
5. **coeff_k_floor=1.0 is CRITICAL and OPTIMAL** — R² jumped from 0.06 to 0.64 (10x improvement!), k_floor=2.0 too strong (evidence: Iter 14, 17, 18, 24 - VERY STRONG)
6. **k_floor_threshold should match log_k_min** — threshold=-2.5 (below true min) hurt R² from 0.51 to 0.37 (evidence: Iter 20)
7. **L1=0.0 + longer training DON'T combine** — Iter 22 (aug=3000), Iter 29 (aug=4000), Iter 40 (lr_sub=0.001) all hurt R² (evidence: CONFIRMED 3x)
8. **MLP_node activation doesn't correlate with R² improvement** — batch 8 all had active MLP_node but worse R² than batch 7 (evidence: Iter 29-32)
9. **lr_sub=0.001 is optimal** — BREAKTHROUGH, R² improved from 0.69 to 0.73 (evidence: Iter 35); lr_sub=0.002 too high (Iter 38)
10. **sub_norm=2.0 improves alpha but hurts R²** — alpha improved to 0.88 but R² dropped (evidence: Iter 30, 37)
11. **Seed sensitivity is significant (~0.2 R² variance)** — Iter 35 (seed=42) got R²=0.73, Iter 41 (seed=123) got R²=0.49 with identical config (evidence: Iter 41 - NEW)
12. **sub_diff=5 is optimal** — less monotonicity (sub_diff=3) hurts R² (evidence: Iter 43 - NEW)

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
- "aug=5000 continues to improve R²" — FALSE, R² dropped from 0.69 to 0.65 (Iter 25)
- "Different seed breaks MLP_node degeneracy" — FALSE, seed=123 same flat MLP_node (Iter 27)
- "Smaller batch_size helps convergence" — FALSE, batch_size=4 hurt R² (Iter 28)
- "Lower lr_k=0.003 gives finer convergence" — FALSE, convergence too slow (Iter 31)
- "Stronger sub_norm=2.0 improves alpha" — FALSE, alpha=0.79 worse than sub_norm=1.0 (Iter 30)
- "sub_norm=2.0 + shorter training works" — FALSE, R²=0.52 worse than aug=4000 (Iter 33)
- "lr_sub=0.002 improves over 0.001" — FALSE, R² dropped from 0.73 to 0.52 (Iter 38)
- "Combining lr_sub=0.001 with other improvements beats baseline" — FALSE, all combinations hurt (Iter 37-40)
- "Less monotonicity (sub_diff=3) lets MLP learn better" — FALSE, R² dropped from 0.73 to 0.61 (Iter 43)

### Open Questions
- Is R²=0.69-0.73 a fundamental limit for this regime?
- How much of the variance is seed-dependent vs method-dependent?
- Would a fundamentally different approach (different architecture, loss function) help?
- Can ensemble averaging across seeds improve stability?

---

## Block 4 Progress (Iter 37-48) - IN PROGRESS

### Batch 10 Results (Iter 37-40) - COMPLETE
| Iter | R² | outliers | alpha | slope | Key change | Result |
|------|-----|----------|-------|-------|------------|--------|
| 37 | 0.5882 | **16** | **0.88** | 0.98 | sub_norm=2.0 | Hurt R², best alpha |
| 38 | 0.5176 | 19 | 0.80 | 0.96 | lr_sub=0.002 | TOO HIGH, worst R² |
| 39 | 0.6537 | 20 | 0.81 | **1.00** | lr_node=0.002 | Perfect slope, R² dropped |
| 40 | 0.6622 | 21 | 0.83 | 0.97 | L1=0.0 | Still hurts R² |

### Batch 11 Results (Iter 41-44) - COMPLETE
| Iter | R² | outliers | alpha | slope | Key change | Result |
|------|-----|----------|-------|-------|------------|--------|
| 41 | 0.4872 | 21 | 0.71 | 0.96 | seed=123 | **SEED SENSITIVE** — R² dropped 0.24! |
| 42 | **0.6896** | **16** | **0.94** | 0.98 | aug=4500 | Stable, best alpha ever |
| 43 | 0.6080 | 18 | 0.89 | 0.99 | sub_diff=3 | Less monotonicity hurts |
| 44 | 0.5931 | 19 | 0.79 | 0.99 | lr_k=0.007 | Higher lr_k hurts |

**Key Insight:** Iter 35's R²=0.73 was seed-dependent. True stable baseline is ~0.69 (Iter 42 with aug=4500 and best alpha=0.94).

### UCB Tree Status
Best stable node: Node 42 (R²=0.6896, alpha=0.94, outliers=16)
Parent for next batch: Node 42

### Next Batch (Iter 45-48) Strategy

Focus on stabilizing R² and exploring variations from Iter 42:

- **Slot 0 (exploit)**: Iter 42 baseline + sub_diff=7 (probe stronger monotonicity for stability)
- **Slot 1 (exploit)**: Iter 42 baseline + lr_k=0.004 (slightly lower lr_k for more stability)
- **Slot 2 (explore)**: Iter 42 + hidden_dim_sub=128 (wider MLP_sub capacity)
- **Slot 3 (principle-test)**: Iter 42 + batch_size=16 (larger batches for gradient stability). Testing principle: "batch_size=8 is optimal"
