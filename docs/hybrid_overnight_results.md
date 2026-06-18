# Hybrid overnight results (Thu Jun 18 14:05:06 EDT 2026)
- **identifiability ecoli_core_hybrid** | joint rank 215/215, null 0, cond 3.8e+03 | lin recovery R2 Vmax=1.000 Km=1.000 joint=1.000 (resid 0.117)
- **identifiability glyco_hybrid** | joint rank 72/72, null 0, cond 6.6e+04 | lin recovery R2 Vmax=1.000 Km=1.000 joint=1.000 (resid 0.081)
- **identifiability yeast_hybrid** | joint rank 429/429, null 0, cond 6.9e+04 | lin recovery R2 Vmax=1.000 Km=1.000 joint=1.000 (resid 0.087)
- **Phase 1 conclusion** | joint (Vmax,Km) is RANK-identifiable (full rank, 0 null, linearized R2=1.0 all rungs) BUT the Km block is practically ill-conditioned (cond 3.8e3-6.9e4): edges whose substrate runs in saturation (c>>Km) weakly constrain Km. A naive unregularized Gauss-Newton diverges on those directions -> recovery needs regularization / well-excited-edge selection (Phase 2-3). Predicts: Vmax recoverable, Km the weak link -- consistent with the oracle (Vmax learnable when Km given).
- **ecoli_core_hybrid_oracle** [15:03] | final Vmax R2=0.112, Km R2=1.000 (phase-A end Vmax R2=0.112) | k_recovery: raw R2=0.112  trimmed R2=0.112  outliers=50/71 (70.4%) | rollout Pearson per-met=0.906 / pooled=0.970
- **ecoli_core_hybrid_joint** [15:05] | final Vmax R2=0.042, Km R2=-0.029 (phase-A end Vmax R2=0.042) | k_recovery: raw R2=0.042  trimmed R2=0.042  outliers=53/71 (74.6%) | rollout Pearson per-met=0.906 / pooled=0.969
- **ecoli_core_hybrid** [16:01] | final Vmax R2=0.080, Km R2=0.008 (phase-A end Vmax R2=0.154) | k_recovery: raw R2=0.079  trimmed R2=0.079  outliers=52/71 (73.2%) | rollout Pearson per-met=0.902 / pooled=0.966
- **ecoli_core_hybrid_slowramp** [16:04] | final Vmax R2=0.048, Km R2=0.004 (phase-A end Vmax R2=0.074) | k_recovery: raw R2=0.048  trimmed R2=0.048  outliers=51/71 (71.8%) | rollout Pearson per-met=0.898 / pooled=0.964
- **ecoli_core_hybrid_fastramp** [17:01] | final Vmax R2=0.040, Km R2=-0.001 (phase-A end Vmax R2=0.073) | k_recovery: raw R2=0.040  trimmed R2=0.040  outliers=50/71 (70.4%) | rollout Pearson per-met=0.906 / pooled=0.969
- **glyco_hybrid** [17:01] | final Vmax R2=0.020, Km R2=-0.055 (phase-A end Vmax R2=0.019) | k_recovery: raw R2=0.020  trimmed R2=0.020  outliers=21/30 (70.0%) | rollout Pearson per-met=0.916 / pooled=0.994
- **glyco_hybrid_oracle** [17:57] | final Vmax R2=0.018, Km R2=1.000 (phase-A end Vmax R2=0.018) | k_recovery: raw R2=0.020  trimmed R2=0.020  outliers=20/30 (66.7%) | rollout Pearson per-met=0.906 / pooled=0.993
- **yeast_hybrid** [18:06] | final Vmax R2=0.045, Km R2=0.000 (phase-A end Vmax R2=0.039) | k_recovery: raw R2=0.045  trimmed R2=0.045  outliers=106/120 (88.3%) | rollout Pearson per-met=0.839 / pooled=0.969
- **yeast_hybrid_oracle** [19:03] | final Vmax R2=0.043, Km R2=1.000 (phase-A end Vmax R2=0.043) | k_recovery: raw R2=0.043  trimmed R2=0.043  outliers=109/120 (90.8%) | rollout Pearson per-met=0.830 / pooled=0.967

ALL DONE Thu Jun 18 19:03:22 EDT 2026
- **SGD-vs-LSQ scale ecoli_core_hybrid_oracle** | Adam on log_k only (exact Km+S, no homeostasis) -> Vmax R2=1.000  vs  LSQ ceiling ~1.000. SGD recovers when isolated.
- **SGD-vs-LSQ scale ecoli_core_hybrid_oracle** | Adam on log_k only (exact Km+S, no homeostasis) -> Vmax R2=-26.505  vs  LSQ ceiling ~1.000. SGD gradient-starved.
- **SGD-vs-LSQ scale ecoli_core_hybrid_oracle** | Adam on log_k only (exact Km+S, no homeostasis) -> Vmax R2=0.224  vs  LSQ ceiling ~1.000. SGD gradient-starved.
- **joint-LSQ+homeo ecoli_core_hybrid_oracle** | Vmax R2=1.000, lambda R2=-6966680.996, cond 4.9e+02, resid 2.3e-04 -> reaction<->homeostasis SEPARABLE by LSQ -> SGD was just ill-conditioned/slow
