# Hybrid structured-MM GNN — multi-day research campaign

Goal: get the GNN to recover Michaelis–Menten kinetics (per-reaction Vmax **and** per-edge Km)
on real-topology networks, by giving it the exact saturating functional family
`g = [c/(Km+c)]^|s|` and learning the scale (Vmax) + the shape parameter (Km). This is the
constructive counterpart to the negative result (a generic MLP_sub cannot factor shape×scale).

**Mechanism:** each phase runs a batch of jobs with a watcher; when the watcher completes it
notifies me, I analyze, update this tracker, and launch the next phase. Resumable via per-config
`log/<cfg>/OVERNIGHT_DONE` markers and per-phase result files under `docs/`.

Guardrails: ≤2 GPU jobs/GPU; station shared with Cedric; time-boxed (`max_train_hours`);
never `CUDA_VISIBLE_DEVICES`; never commit graphs_data/ or log/; checkpoint frequently.

---

## CURRENT STATE
- **Active:** Phase 0 (strategy baseline) — orchestrator PID 14240, 2 GPU lanes.
- **Phase 1 DONE** (see RESULTS): joint (Vmax,Km) is rank-identifiable (full rank, 0 null,
  linearized R²=1.0 at all rungs) BUT the Km block is practically ill-conditioned (cond
  3.8e3–6.9e4) — Km of saturated edges (c≫Km) is weakly constrained. Naive unregularized
  Gauss-Newton diverges on those directions. ⇒ **Vmax recoverable, Km is the weak link.**
- **Next gate:** when P0 finishes, compare oracle/joint/curriculum. Then Phase 2 (Km warm-start +
  *regularized* solve restricted to well-excited edges) — `scripts/joint_gn_solve.py` needs a
  trust-region + Tikhonov/SVD-truncation before it gives the honest ceiling.

---

## Phase 0 — strategy baseline  [RUNNING]
9 configs reusing OU data: per-rung **oracle** (Km=GT frozen), **joint** (Km from epoch 0),
**curriculum** (freeze→ramp); ecoli also fast/slow-ramp. Results → `hybrid_overnight_results.md`.
**Decision gate:**
- oracle Vmax→1 ⇒ exact shape makes the scale learnable; the barrier is shape-learning → P2/P3.
- oracle Vmax stalls <1 ⇒ even the exact shape isn't enough for this GD setup → optimizer/loss audit.
- joint vs curriculum ⇒ does freeze→ramp beat learning shape+scale together?

## Phase 1 — joint identifiability theory  [RUNNING, CPU]
Extend `design_matrix2` to the **joint (Vmax, Km)** Jacobian `A=[∂ḋ/∂logVmax | ∂ḋ/∂logKm]`.
Rank, conditioning, null space, and a linearized Gauss-Newton recovery around GT. Predicts the
achievable ceiling for ANY method on the exact family and exposes any Vmax↔Km degeneracy.
**Deliverable:** `scripts/joint_identifiability.py` + figure + ceiling numbers.

## Phase 2 — Km warm-start / pretraining (step 1 done properly)  [PLANNED]
Implement a cheap per-edge Km estimate from data (method-of-moments on saturation, or a few
Gauss-Newton steps on (Vmax,Km) given S+shape), then run curriculum from that warm start.
**Hypothesis:** a good Km init turns joint ~0 into ~oracle ⇒ the barrier is basin/initialization.

## Phase 3 — regularization & curriculum refinement  [PLANNED]
From P0/P1: scale-pinning reg to break Vmax↔Km degeneracy; alternate minimization (fix Km →
LSQ Vmax; fix Vmax → GD Km); finer curriculum sweep. Find the winning recipe per rung.

## Phase 4 — rollout & generalization  [PLANNED]
Best recipe → held-out-stimulus rollout (target per-met Pearson >0.7); add AR curriculum if
long-horizon rollout is unstable.

## Phase 5 — scale to REAL E. coli data  [PLANNED]
Apply structured-MM + winning recipe to the real Link-2015 data (181 fully-observed iJO1366
reactions). No GT Vmax → judged by held-out one-step + rollout prediction. Headline real test.

## Phase 6 — write-up  [PLANNED]
Assemble figures (evolution, summary, identifiability, real-data) and write the hybrid +
real-data sections into `metabolism.tex`.

---

## RESULTS LOG
(appended as phases complete)
