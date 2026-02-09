# Working Memory: simulation_oscillatory_rank_50 (parallel)

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | n_metabolites | n_reactions | n_frames | noise | eff_rank | Best R² | Optimal lr_k | Optimal lr_sub | Key finding |
| ----- | ------------- | ----------- | -------- | ----- | -------- | ------- | ------------ | -------------- | ----------- |
| 1 | 100 | 256 | 2880 | none | ~50 | 0.0668 | 0.005 | 0.0005 | coeff_MLP_sub_norm=1.0 essential for correct MLP shapes |
| 2 | 100 | 256 | 2880 | none | ~50 | 0.6896 | 0.005 | 0.0005 | k_floor=1.0 + aug=4000 breakthrough! |
| 3 | 100 | 256 | 2880 | none | ~50 | 0.7262 | 0.005 | 0.001 | lr_sub=0.001 broke R² plateau! |
| 4 | 100 | 256 | 2880 | none | ~50 | 0.7358 | 0.005 | 0.001 | sub_diff=7 NEW BEST! |
| 5 | 100 | 256 | 2880 | none | ~50 | 0.7358 | 0.005 | 0.001 | seed=99 R²=0.72, sub_norm confirmed essential |
| 6 | 100 | 256 | 2880 | none | ~50 | 0.7358 | 0.005 | 0.001 | HIGH VARIANCE: replica of Iter45 got R²=0.66 |
| 7 | 100 | 256 | 2880 | none | ~50 | **0.7640** | 0.005 | 0.001 | seed=77+sub_diff=8 R²=0.7640 NEW BEST! |

### Established Principles
1. **coeff_MLP_sub_norm=1.0 is essential** — enables correct MLP shapes: c^2 becomes quadratic, MLP_node becomes active (evidence: Iter 6 best, all batch 3 configs work, Iter 60 CONFIRMED)
2. **Longer training helps R² and alpha (up to aug=4000-4500)** — aug=2000→3000→4000 consistently improves R² (evidence: Iter 8, 9, 12, 17, 21, 42 - VERY STRONG). But aug=5000 hurt R² (Iter 25, 49, 84 - CONFIRMED 3x). aug=3500 also hurt (Iter 36). aug=4750 hurt (Iter 67). aug=4000 hurt (Iter 76 - CONFIRMED)
3. **lr_k=0.005 is optimal** — lower lr_k=0.003/0.004/0.0045 too slow, higher lr_k=0.007/0.01 hurts R² (evidence: Iter 7, 19, 31, 44, 46, 70)
4. **lr_node=0.005 hurts R²** — high lr_node destabilizes training, R² dropped to 0.017 (evidence: Iter 11)
5. **coeff_k_floor=1.0 is CRITICAL** — R² jumped from 0.06 to 0.64 (10x improvement!), k_floor=2.0 too strong (evidence: Iter 14, 17, 18, 24 - VERY STRONG). k_floor=1.5 gives R²=0.70 (Iter 64). k_floor=1.25 HURT R² to 0.56 (Iter 65 - NON-MONOTONIC!)
6. **k_floor_threshold should match log_k_min** — threshold=-2.5 (below true min) hurt R² from 0.51 to 0.37 (evidence: Iter 20)
7. **L1=0.0 + longer training DON'T combine** — Iter 22 (aug=3000), Iter 29 (aug=4000), Iter 40 (lr_sub=0.001) all hurt R² (evidence: CONFIRMED 3x)
8. **MLP_node activation doesn't correlate with R² improvement** — batch 8 all had active MLP_node but worse R² than batch 7 (evidence: Iter 29-32)
9. **lr_sub=0.001 is optimal** — BREAKTHROUGH, R² improved from 0.69 to 0.73 (evidence: Iter 35); lr_sub=0.002 too high (Iter 38), lr_sub=0.0015 too high (Iter 56), lr_sub=0.0012 too high (Iter 74 - NEW)
10. **sub_norm=2.0 improves alpha but hurts R²** — alpha improved to 0.88 but R² dropped (evidence: Iter 30, 37)
11. **Seed sensitivity is significant (~0.2 R² variance)** — Iter 35 (seed=42) got R²=0.73, Iter 41 (seed=123) got R²=0.49 with identical config (evidence: Iter 41)
12. **sub_diff optimal is SEED-DEPENDENT** — sub_diff=7 for seed=42 (Iter 45), sub_diff=8 for seed=77 (Iter 82). sub_diff=8 hurt seed=42 (Iter 50) but helped seed=77!
13. **batch_size=8 is optimal** — batch_size=16 hurts R² (evidence: Iter 48)
14. **Wider/Deeper MLP_sub hurts R²** — hidden_dim_sub=128 allows degenerate solutions (Iter 47), n_layers_sub=4 also hurts (Iter 52, 79 - CONFIRMED)
15. **sub_diff=7 improves seed robustness** — R² with seed=123 improved from 0.49 (sub_diff=5) to 0.66 (sub_diff=7) — gap reduced from 0.24 to 0.08 (evidence: Iter 51 vs Iter 41)
16. **L1=1.0 is optimal** — L1=0.5 hurts R² (evidence: Iter 59)
17. **seed=99 gives R²=0.72** — better than seed=123 (R²=0.70) but below seed=42 (R²=0.74) (evidence: Iter 58)
18. **lr_node=0.001 is optimal** — lr_node=0.0005 hurts R² (0.62 vs 0.74) (evidence: Iter 63)
19. **seed=99 requires aug>=4500** — aug=4250 dropped R² from 0.72 to 0.43 (evidence: Iter 62)
20. **k_floor response is NON-MONOTONIC** — k_floor=1.25 gave R²=0.56, worse than BOTH k_floor=1.0 (R²=0.74) and k_floor=1.5 (R²=0.70) (evidence: Iter 65)
21. **seed=99 + k_floor=1.5 incompatible** — R²=0.67, worse than seed=42 + k_floor=1.5 (R²=0.70) (evidence: Iter 66)
22. **hidden_dim_node=64 required** — hidden_dim_node=32 hurts R² significantly (0.47 vs 0.66) (evidence: Iter 71)
23. **Training has HIGH VARIANCE** — exact same config can give R²=0.74 (Iter 45) or R²=0.66 (Iter 69), R²=0.75 (Iter 80) or R²=0.66 (Iter 81) (evidence: Iter 69, 81)
24. **seed=123 can reach R²=0.72 with optimal config** — better than previous 0.66 (evidence: Iter 73 vs Iter 51)
25. **seed=77 is BEST seed** — R²=0.7479 (Iter 80), R²=0.7640 with sub_diff=8 (Iter 82) — beats seed=42's best of 0.7358!
26. **seed=77 + sub_diff=8 is NEW BEST CONFIG** — R²=0.7640, outliers=15, alpha=0.87 (evidence: Iter 82 - NEW)

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
- "aug=5000 continues to improve R²" — FALSE, R² dropped from 0.69 to 0.65 (Iter 25), dropped 0.74 to 0.70 (Iter 49), dropped 0.75 to 0.69 (Iter 84 - CONFIRMED 3x)
- "Different seed breaks MLP_node degeneracy" — FALSE, seed=123 same flat MLP_node (Iter 27)
- "Smaller batch_size helps convergence" — FALSE, batch_size=4 hurt R² (Iter 28)
- "Lower lr_k=0.003/0.004/0.0045 gives finer convergence" — FALSE, convergence too slow (Iter 31, 46, 70)
- "Stronger sub_norm=2.0 improves alpha" — FALSE, alpha=0.79 worse than sub_norm=1.0 (Iter 30)
- "sub_norm=2.0 + shorter training works" — FALSE, R²=0.52 worse than aug=4000 (Iter 33)
- "lr_sub=0.002 improves over 0.001" — FALSE, R² dropped from 0.73 to 0.52 (Iter 38)
- "Combining lr_sub=0.001 with other improvements beats baseline" — FALSE, all combinations hurt (Iter 37-40)
- "Less monotonicity (sub_diff=3) lets MLP learn better" — FALSE, R² dropped from 0.73 to 0.61 (Iter 43)
- "Wider MLP_sub (hidden_dim=128) helps" — FALSE, R² dropped to 0.56 (Iter 47)
- "Larger batch_size=16 gives more stable gradients" — FALSE, R² dropped to 0.56 (Iter 48)
- "Deeper MLP_sub (n_layers=4) helps" — FALSE, R² dropped to 0.55 (Iter 52), 0.47 with 32 outliers (Iter 79)
- "Weaker sub_norm=0.5 helps" — FALSE, R² dropped from 0.74 to 0.60 (Iter 60)
- "L1=0.5 better than L1=1.0" — FALSE, R² dropped from 0.74 to 0.60 (Iter 59)
- "lr_node=0.0005 helps when MLP_node inactive" — FALSE, R² dropped to 0.62 (Iter 63)
- "seed=99 benefits from shorter training" — FALSE, aug=4250 dropped R² to 0.43 (Iter 62)
- "seed=7 as good as seed=42" — FALSE, R²=0.688 vs 0.74 (Iter 61)
- "Intermediate k_floor=1.25 optimal" — FALSE, R²=0.56 worse than both extremes (Iter 65)
- "aug=4750 within safe range" — FALSE, R² dropped to 0.66 despite alpha=0.96 (Iter 67)
- "sub_diff=6 + k_floor=1.5 compensate" — FALSE, R²=0.61 (Iter 68)
- "sub_norm=1.5 better than 1.0" — FALSE, R²=0.55 vs 0.66 (Iter 72)
- "hidden_dim_node=32 simpler is better" — FALSE, R²=0.47 significantly worse (Iter 71)
- "lr_sub=0.0012 helps MLP_sub learn c^2" — FALSE, R²=0.58 vs 0.66 (Iter 74)
- "sub_diff=9 forces c^2 quadratic" — FALSE, R²=0.68 vs 0.66, more outliers (Iter 75)
- "aug=4000 avoids overfitting" — FALSE, R²=0.64 vs 0.66 (Iter 76)
- "seed=42 is optimal" — FALSE, seed=77 got R²=0.7479 (Iter 80), then R²=0.7640 with sub_diff=8 (Iter 82)
- "k_floor=1.5 is reproducible" — FALSE, same config gave R²=0.70 (Iter 64) and R²=0.51 (Iter 78)
- "sub_diff=8 hurts all seeds" — FALSE, hurts seed=42 (Iter 50) but helps seed=77 (Iter 82)
- "seed=78 is good (adjacent to seed=77)" — FALSE, R²=0.39, 23 outliers (Iter 83 - NEW)

### Open Questions
- Is R²=0.77+ achievable? (current best R²=0.7640 with seed=77+sub_diff=8)
- How much of the variance is seed-dependent vs method-dependent?
- Can ensemble averaging across seeds improve stability?
- What causes k_floor non-monotonic response (1.25 < 1.0 and 1.5)?
- Why does MLP_sub c^2 consistently fail to learn quadratic shape in most runs?
- Is there a way to break the c^2 linear failure mode?
- Are there more "golden seeds" like 77 to discover?

---

## Block 8 Progress (Iter 85-96) - IN PROGRESS

### Best Config So Far (Iter 82 - GLOBAL BEST!)
- seed=77, lr_k=0.005, lr_node=0.001, lr_sub=0.001
- batch_size=8, data_augmentation_loop=4500
- coeff_MLP_sub_diff=8, coeff_MLP_sub_norm=1.0, coeff_MLP_node_L1=1.0
- coeff_k_floor=1.0, k_floor_threshold=-2.0
- R²=0.7640, outliers=15, alpha=0.87, slope=0.97

### Batch 21 (Iter 81-84) Results [FINAL]
| Slot | Iter | Mutation | R² | outliers | alpha | slope |
|------|------|----------|-----|----------|-------|-------|
| 0 | 81 | replicate seed=77 | 0.6609 | 16 | 0.80 | 0.99 |
| 1 | 82 | sub_diff=8 | **0.7640** | 15 | 0.87 | 0.97 |
| 2 | 83 | seed=78 | 0.3870 | 23 | 0.81 | 0.97 |
| 3 | 84 | aug=5000 | 0.6887 | 18 | 0.75 | 0.95 |

Key findings: seed=77+sub_diff=8 achieved R²=0.7640 (NEW GLOBAL BEST!). Replicate of seed=77 gave R²=0.66 (HIGH VARIANCE). seed=78 poor (R²=0.39). aug=5000 confirmed bad (3rd time).

### Batch 22 (Iter 85-88) Strategy
Given seed=77+sub_diff=8 discovery (NEW BEST R²=0.7640):

- **Slot 0 (exploit)**: Replicate seed=77+sub_diff=8 — verify reproducibility (parent=82)
- **Slot 1 (explore)**: seed=76 + sub_diff=8 — explore adjacent seed (parent=82)
- **Slot 2 (explore)**: seed=79 + sub_diff=8 — explore another adjacent seed (parent=82)
- **Slot 3 (principle-test)**: seed=42 + sub_diff=8 — test if sub_diff=8 helps seed=42 with lr_sub=0.001 (parent=82). Testing principle: "sub_diff=8 hurts seed=42"
