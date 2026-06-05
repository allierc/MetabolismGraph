# MetabolismGraph — repo guide for Claude

GNN **inverse modelling** of metabolic networks: given metabolite concentration
time-series and the bipartite metabolite–reaction graph, recover per-reaction
rate constants $k_j$, the substrate kinetic law (MLP$_{sub}$), and per-metabolite
homeostasis (MLP$_{node}$). Validated on synthetic data; pushing toward real data.

## Run
- Env: `/workspace/.conda_envs/neural-graph-linux/bin/python` (editable install).
- Entry: `python GNN_Main.py -o {generate,train,test} <config>` (configs in `config/`).
  `<config>` sets `dataset:` (→ `graphs_data/<dataset>/`) and the log dir is `log/<config>/`.
- 2× RTX A6000. **Never set `CUDA_VISIBLE_DEVICES`** (breaks `set_device` → CPU fallback).
  `device: auto` always picks the freest GPU (usually cuda:0); set `device: cuda:1` to use GPU1.

## CRITICAL gotchas (check these before trusting any result)
1. **Be dubious of good results — check for leakage first.** History: a Phase-2
   "supervised contrastive loss" read ground-truth type labels `x[:,6]` during
   training (label leak); removed. Good numbers are guilty until proven clean.
2. **`data_test` rollout is BUGGY:** it calls `model(dataset)` with **no `stimulus`**
   (graph_trainer.py ~L1383), so it evaluates the model without its external
   boundary drive → the rollout diverges (the `R²=-390` / `-3.2` glycolysis numbers
   were artifacts). Use `figures/glyco_rollout.py` (passes `stimulus=stim[t]`) for the
   real rollout. Feeding the boundary stimulus is legitimate (it's a given input,
   like a velocity drive), not a leak.
3. **Global vs per-metabolite R² differ a lot.** Global R² (pooled over all
   metabolites×time) is variance-weighted → dominated by a few high-variance
   metabolites and flattering (can be 0.99). Per-metabolite R² (computed per
   metabolite, then averaged) is normalized by each metabolite's own variance, so
   near-constant metabolites get hugely negative R² and drag the mean down. **Always
   report per-metabolite (honest) alongside global.** A big gap = the fit is carried
   by a few easy metabolites.
4. **In-sample risk:** `n_runs=1` ⇒ train and rollout use the SAME single trajectory.
   A high rollout R² can be memorization. The **leak-resistant** test is parameter
   recovery: `figures/k_recovery.py <config>` (learned vs true $\log k$). You can't
   recover the right $V_{max}/K_m$ by memorizing a trajectory. **Glycolysis as of
   2026-06: rollout global R²=0.99 but Vmax R²≈0.00 (slope 0.06, 73% outliers) ⇒
   degenerate (right dynamics, wrong mechanism), will NOT generalize.**
5. **k-recovery reporting rule:** report raw + trimmed R² + %outliers; hard rule
   **outliers must be ≤10%** (`|corrected log10 k − true| > 0.3 dex`).

## Working points / status
- Toy model `k_recovery_winner` (rank-50 oscillatory, S given): leak-free k recovery
  **R²=0.75±0.04** across seeds (raw 0.74 / trimmed 0.98 / 4.3% outliers). NOT the
  published 0.87 (that was a lucky single draw).
- AR rollout curriculum (opt-in via `training.n_steps_schedule`; cx-ported: horizon
  ramp + LR co-ramp + soft tail-loss + grad clip) — fixes rollout *instability* but
  glycolysis Vmax recovery is still ~0 (degenerate). Single-step k-recovery path is
  untouched when `n_steps_schedule` is empty.
- Real data (nmeth.3584): 4 organisms probed for amenability. Low-rank + network
  coverage generalize; dc/dt smoothness does not — **Y. lipolytica** is the only
  smooth (learnable) one → best real-fit target.

## Conventions
- Figures: re-runnable script in `figures/<name>.py` (recapitulate the analysis, don't
  copy temp training plots). Panels: **bold letter top-left, no panel titles, large
  font**. Colors: GT=green, predicted=black.
- Docs are two-tier: `docs/lab_notebook.md` = raw running log (verbose, hourly);
  `docs/metabolism.tex/.pdf` = polished, semi-publishable.
- Weekend autonomous loop: `WEEKEND_RUN.md` is the playbook; an hourly cron runs it.
