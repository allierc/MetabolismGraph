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
| 7 | 100 | 256 | 2880 | none | ~50 | 0.7640 | 0.005 | 0.001 | seed=77+sub_diff=8 R²=0.7640 NEW BEST! |
| 8 | 100 | 256 | 2880 | none | ~50 | 0.8512 | 0.005 | 0.001 | seed=79+aug=5000 R²=0.8512 NEW BEST! |
| 9 | 100 | 256 | 2880 | none | ~50 | 0.8512 | 0.005 | 0.001 | seed=77+sub_diff=6 R²=0.80, 2ND BEST CONFIG! |
| 10 | 100 | 256 | 2880 | none | ~50 | **0.8691** | 0.005 | 0.001 | **GLOBAL BEST!** seed=77+sub_diff=6+aug=5500 R²=0.8691 |
| 11 | 100 | 256 | 2880 | none | ~50 | **0.8691** | 0.005 | 0.001 | k_floor=1.5+sub_diff=7 R²=0.8458 (2nd best), variance ~0.30 confirmed |
| 12 | 100 | 256 | 2880 | none | ~50 | **0.8691** | 0.005 | 0.001 | k_floor=1.5 doesn't reduce variance (~0.21), sub_diff=6 prefers k_floor=1.0 |

### Established Principles
1. **coeff_MLP_sub_norm=1.0 is essential** — enables correct MLP shapes: c^2 becomes quadratic, MLP_node becomes active (evidence: Iter 6 best, all batch 3 configs work, Iter 60 CONFIRMED)
2. **Longer training helps R² (SEED-DEPENDENT)** — seed=77 benefits from aug=5500 (R²=0.87 Iter 116!), seed=79 benefits from aug=5000 but NOT aug=5500. aug=5500 is MAX for seed=77 (aug=5750 HURTS, Iter 118, 126)
3. **lr_k=0.005 is optimal** — lower lr_k=0.003/0.004/0.0045 too slow, higher lr_k=0.007/0.01 hurts R² (evidence: Iter 7, 19, 31, 44, 46, 70)
4. **lr_node=0.005 hurts R²** — high lr_node destabilizes training, R² dropped to 0.017 (evidence: Iter 11)
5. **coeff_k_floor=1.0-1.5 is OPTIMAL range for sub_diff=7** — R² jumped from 0.06 to 0.64 with k_floor=1.0 (10x improvement!), k_floor=2.0 comparable to k_floor=1.5 given variance. **k_floor=1.5 with sub_diff=7 gives R²=0.85 (Iter 132)**. **sub_diff=6 prefers k_floor=1.0 (Iter 134 R²=0.51 vs Iter 116 R²=0.87)**
6. **k_floor_threshold should match log_k_min** — threshold=-2.5 (below true min) hurt R² from 0.51 to 0.37 (evidence: Iter 20)
7. **L1=0.0 + longer training DON'T combine** — Iter 22 (aug=3000), Iter 29 (aug=4000), Iter 40 (lr_sub=0.001) all hurt R² (evidence: CONFIRMED 3x)
8. **MLP_node activation doesn't correlate with R² improvement** — batch 8 all had active MLP_node but worse R² than batch 7 (evidence: Iter 29-32)
9. **lr_sub=0.001 is optimal** — BREAKTHROUGH, R² improved from 0.69 to 0.73 (evidence: Iter 35); lr_sub=0.002 too high (Iter 38), lr_sub=0.0015 too high (Iter 56), lr_sub=0.0012 too high (Iter 74 - NEW)
10. **sub_norm=2.0 improves alpha but hurts R²** — alpha improved to 0.88 but R² dropped (evidence: Iter 30, 37)
11. **Seed sensitivity is significant (~0.21-0.30 R² variance)** — same config can give R²=0.51-0.85 (evidence: Iter 129 vs 132, Iter 133 vs 132)
12. **sub_diff is SEED-SPECIFIC** — seed=77 optimal sub_diff=6-7, seed=42 prefers sub_diff=6 (R²=0.72 Iter 131 vs 0.57 Iter 123 with sub_diff=7), seed=79 prefers sub_diff=8
13. **batch_size=8 is optimal** — batch_size=16 hurts R² (evidence: Iter 48)
14. **Wider/Deeper MLP_sub hurts R²** — hidden_dim_sub=128 allows degenerate solutions (Iter 47), n_layers_sub=4 also hurts (Iter 52, 79 - CONFIRMED)
15. **L1=1.0 is optimal** — L1=0.5 hurts R² (evidence: Iter 59)
16. **seed=99 gives R²=0.72** — better than seed=123 (R²=0.70) but below seed=42 (R²=0.74) (evidence: Iter 58)
17. **lr_node=0.001 is optimal** — lr_node=0.0005 hurts R² (0.62 vs 0.74) (evidence: Iter 63)
18. **seed=99 requires aug>=4500** — aug=4250 dropped R² from 0.72 to 0.43 (evidence: Iter 62)
19. **k_floor response is NON-MONOTONIC** — k_floor=1.25 gave R²=0.56, worse than BOTH k_floor=1.0 (R²=0.74) and k_floor=1.5 (R²=0.70) (evidence: Iter 65)
20. **hidden_dim_node=64 required** — hidden_dim_node=32 hurts R² significantly (0.47 vs 0.66) (evidence: Iter 71)
21. **Training has EXTREME VARIANCE (~0.21-0.30)** — exact same config can give R²=0.51 (Iter 129) or R²=0.85 (Iter 132)
22. **seed=77 is BEST seed** — R²=0.7479 (Iter 80), R²=0.7640 with sub_diff=8 (Iter 82), R²=0.80 with sub_diff=6 (Iter 106), **R²=0.8691** with aug=5500 (Iter 116), R²=0.8458 with k_floor=1.5 (Iter 132)
23. **seed=77 + sub_diff=6 + aug=5500 is GLOBAL BEST config** — R²=0.8691, outliers=10, alpha=0.93 (evidence: Iter 116)
24. **seed=79 is high-variance seed** — R²=0.7484 (Iter 87), R²=0.8512 (Iter 96), but R²=0.64 (Iter 113 same config!) — variance ~0.21!
25. **sub_diff=9 hurts ALL golden seeds** — seed=42 (refuted), seed=77 (0.80→0.62 Iter 108), seed=79 (0.75→0.64) — 3x confirmed
26. **aug=5500 is optimal ceiling** — HELPS seed=77 (R²=0.87!), HURTS seed=79 (R²=0.61 Iter 98). aug=5750 HURTS BOTH sub_diff=6 (Iter 118) AND sub_diff=7 (Iter 126)
27. **k_floor=1.5 + sub_diff=7 can achieve R²=0.85** — 2nd best result ever (Iter 132)! But variance still ~0.21
28. **k_floor response is sub_diff-dependent** — sub_diff=7 tolerates k_floor=1.5-2.0, sub_diff=6 prefers k_floor=1.0 (Iter 134 R²=0.51 vs Iter 116 R²=0.87)

### Refuted Hypotheses
- "MLP_node needs higher lr_node to learn" — FALSE, lr_node=0.005 hurts (Iter 11), lr_node=0.002 no effect (Iter 26, 39)
- "coeff_MLP_node_L1 prevents MLP_node learning" — FALSE, MLP_node activates with sub_norm=1.0 regardless of L1 setting
- "Recurrent training (time_step>1) breaks degeneracy" — FALSE, no R² improvement, slower (Iter 13, 34 - CONFIRMED TWICE)
- "Smaller MLP architecture helps k recovery" — FALSE, worst R²=0.011 (Iter 15)
- "Stronger monotonicity (sub_diff=10) helps" — FALSE, R² dropped to 0.41 (Iter 32)
- "Tighter k_floor_threshold improves k recovery" — FALSE, threshold=-2.5 hurt R² (Iter 20)
- "Higher lr_k with k_floor safety net works" — FALSE, lr_k=0.01 still hurts (Iter 19), lr_k=0.007 also hurts (Iter 44)
- "Lower lr_sub helps k dominate" — FALSE, no effect when k_floor is active (Iter 23), higher lr_sub actually helps (Iter 35)
- "Stronger k_floor (coeff=2.0) reduces outliers" — WEAKENED, k_floor=2.0 NOT catastrophically worse (Iter 136 R²=0.69)
- "k_floor=1.5 reduces variance" — FALSE, Iter 133 R²=0.64 vs Iter 132 R²=0.85 (variance ~0.21 persists)
- "k_floor=1.5 helps sub_diff=6" — FALSE, Iter 134 R²=0.51 vs Iter 116 R²=0.87
- "aug=5500 hurts ALL seeds" — **REFUTED** — seed=77 benefits from aug=5500 (R²=0.87 Iter 116!)
- "Different seed breaks MLP_node degeneracy" — FALSE, seed=123 same flat MLP_node (Iter 27)
- "Smaller batch_size helps convergence" — FALSE, batch_size=4 hurt R² (Iter 28)
- "Lower lr_k=0.003/0.004/0.0045 gives finer convergence" — FALSE, convergence too slow (Iter 31, 46, 70)
- "Stronger sub_norm=2.0 improves alpha" — FALSE, alpha=0.79 worse than sub_norm=1.0 (Iter 30)
- "sub_norm=2.0 + shorter training works" — FALSE, R²=0.52 worse than aug=4000 (Iter 33)
- "lr_sub=0.002 improves over 0.001" — FALSE, R² dropped from 0.73 to 0.52 (Iter 38)
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
- "sub_diff=6 + k_floor=1.5 compensate" — FALSE, R²=0.61 (Iter 68) and R²=0.51 (Iter 134)
- "sub_norm=1.5 better than 1.0" — FALSE, R²=0.55 vs 0.66 (Iter 72)
- "hidden_dim_node=32 simpler is better" — FALSE, R²=0.47 significantly worse (Iter 71)
- "lr_sub=0.0012 helps MLP_sub learn c^2" — FALSE, R²=0.58 vs 0.66 (Iter 74)
- "sub_diff=9 forces c^2 quadratic" — FALSE, R²=0.68 vs 0.66, more outliers (Iter 75)
- "seed=42 is optimal" — FALSE, seed=77 got R²=0.87 (Iter 116)!
- "k_floor=1.5 is reproducible" — FALSE, same config gave R²=0.85 (Iter 132) and R²=0.64 (Iter 133)
- "sub_diff=8 hurts seed=42" — REFUTED! Iter 88 got R²=0.72 with seed=42+sub_diff=8
- "seed=78 is good (adjacent to seed=77)" — FALSE, R²=0.39, 23 outliers (Iter 83)
- "seed=76 is good (adjacent to seed=77)" — FALSE, R²=0.51, 22 outliers (Iter 86)
- "sub_diff=9 helps golden seeds" — FALSE, hurt seed=79 (0.75→0.64), seed=77 (0.80→0.62 Iter 108), seed=42
- "seed=80 is good (adjacent to seed=79)" — FALSE, R²=0.57, not a golden seed (Iter 91)
- "seed=81 is good (near seed=79)" — FALSE, R²=0.55 (Iter 95)
- "sub_diff=7 works with aug=5000" — FALSE FOR seed=79, R² dropped from 0.85 to 0.47 (Iter 100); BUT TRUE FOR seed=77, R²=0.76 (Iter 104)
- "sub_diff=8 universally optimal for golden seeds" — FALSE, seed=77+sub_diff=6 R²=0.87 > sub_diff=8 (Iter 116)
- "seed=79 is most RELIABLE golden seed" — FALSE, seed=77 variance ~0.30, seed=79 variance ~0.21 over multiple runs
- "sub_diff=6 transfers across seeds" — FALSE, seed=79 got R²=0.65, seed=42 got R²=0.59 vs seed=77's R²=0.87 (Iter 111, 112, 116)
- "sub_diff=7 helps seed=79" — FALSE, R²=0.57 vs sub_diff=8's R²=0.64 (Iter 115 vs 113)
- "sub_diff=7 is more robust than sub_diff=6" — **WEAKENED** — Iter 129 got R²=0.51, same variance range as sub_diff=6
- "sub_diff=7 transfers to seed=42" — FALSE, R²=0.57 (Iter 123)
- "sub_diff=8 works for seed=77" — FALSE, R²=0.51 (Iter 124) vs sub_diff=7's R²=0.73-0.81
- "sub_diff=7 transfers to seed=79" — FALSE, R²=0.54 (Iter 127)
- "aug=5750 tolerates sub_diff=7" — FALSE, R²=0.68 (Iter 126) vs aug=5500's R²=0.78 (Iter 125)
- "k_floor=1.0 is universally optimal" — **WEAKENED** — k_floor=1.5+sub_diff=7 got R²=0.85 (Iter 132)

### Open Questions
- Is R²=0.90+ achievable? (current best R²=0.8691 with seed=77+sub_diff=6+aug=5500)
- Can lower sub_diff (sub_diff=5) work?
- Why is MLP_node completely inactive (slope=0) in ALL recent runs despite coeff_MLP_sub_norm=1.0?

---

## Block 12 Progress (Iter 133-144)

### Best Configs
1. **GLOBAL BEST**: Iter 116 — seed=77+sub_diff=6+aug=5500+k_floor=1.0 → R²=0.8691, outliers=10, α=0.93
2. **2ND BEST**: Iter 132 — seed=77+sub_diff=7+k_floor=1.5+aug=5500 → R²=0.8458, outliers=12, α=0.87
3. **seed=42 best**: Iter 131 — seed=42+sub_diff=6+aug=5500+k_floor=1.0 → R²=0.7208, outliers=12, α=0.92

### Batch 34 Results (Iter 133-136) — k_floor=1.5 variance confirmed

| Iter | Seed | sub_diff | k_floor | aug | R² | outliers | α | Observation |
|------|------|----------|---------|-----|-----|----------|-----|-------------|
| 133 | 77 | 7 | 1.5 | 5500 | 0.6350 | 18 | 0.92 | Replicate of Iter 132 (R²=0.85), extreme variance! |
| 134 | 77 | 6 | 1.5 | 5500 | 0.5122 | 21 | 0.80 | sub_diff=6+k_floor=1.5 underperforms |
| 135 | 42 | 6 | 1.5 | 5500 | 0.6770 | 11 | 0.97 | k_floor=1.5 slightly worse for seed=42 |
| 136 | 77 | 7 | 2.0 | 5500 | 0.6918 | 20 | 0.75 | k_floor=2.0 NOT catastrophically worse |

**Key findings from Batch 34:**
1. **k_floor=1.5 variance persists** — Iter 133 R²=0.64 vs Iter 132 R²=0.85 (variance ~0.21)
2. **sub_diff=6 requires k_floor=1.0** — Iter 134 R²=0.51 with k_floor=1.5 vs Iter 116 R²=0.87 with k_floor=1.0
3. **k_floor=2.0 is within variance range** — R²=0.69 (Iter 136), not catastrophically worse than k_floor=1.5
4. **Variance is DOMINANT** — config differences matter less than training stochasticity

### Batch 35 (Iter 137-140) Strategy

Return to k_floor=1.0 baseline and explore sub_diff boundaries:
- **Slot 0 (exploit)**: seed=77+sub_diff=6+k_floor=1.0+aug=5500 — replicate global best config (Iter 116)
- **Slot 1 (exploit)**: seed=77+sub_diff=7+k_floor=1.0+aug=5500 — test sub_diff=7 with k_floor=1.0
- **Slot 2 (explore)**: seed=42+sub_diff=6+k_floor=1.0+aug=5500 — replicate seed=42 best (Iter 131)
- **Slot 3 (principle-test)**: seed=77+sub_diff=5+k_floor=1.0+aug=5500 — test lower sub_diff boundary. Testing principle: "sub_diff must be >=6"
