# MetabolismGraph — experiment lab notebook

Detailed running log of the autonomous weekend runs. Raw and chronological:
every hour the loop appends what it checked, launched, found, and decided —
including dead ends. The **polished, semi-publishable** version of the science
lives in `docs/metabolism.pdf`; this file is the unfiltered notebook behind it.

Format: one dated entry per cycle. Be concrete (configs, metrics, file paths),
honest about failures, and note the *decision* taken.

---

## 2026-06-05 — weekend kickoff

**Goal.** (1) Reproduce the toy-model rate-constant recovery and quantify its
robustness by CV; (2) build intuition on whether the GNN inverse approach is
amenable to real data, across the three real datasets (yeast glycolysis kinetic
model, yeast-GEM, real *E. coli* metabolomics). Work by the scientific method:
state a hypothesis, run it, record validate/falsify. Both GPUs (RTX A6000);
`device: cuda:1` to place on GPU1 (auto always picks GPU0). Env
`/workspace/.conda_envs/neural-graph-linux/bin/python`.

**H1 — reproduce the leak-free winner (`k_recovery_winner`, seed 77).**
Fresh data gen + train, single-step k-recovery, S given. Final **k R² = 0.742**.
→ reproduces a working point but *below* the published 0.87. Bug notes: pinning
`CUDA_VISIBLE_DEVICES` breaks `set_device` (CPU fallback); the published 0.87
used `n_epochs=1` (claude override), not the file's 10.

**H2 — CV robustness.** Seeds 42/79/7/123 reuse the same data, vary training
seed. Finals: 77=0.742, 42=0.705, 79=0.808, 7=0.738 → **0.748 ± 0.037** (seed
123 stopped at 30%, 0.72 rising). → *validated*: reproducible, tight spread
(±0.04, not the ±0.2 the old logs implied), but lands ~0.75 not 0.87. The 0.87
was a lucky draw. Figure `figures/k_recovery.py`: raw R²=0.74 but **trimmed
R²=0.98, 11/256 outliers = 4.3%** (passes the 10%-outlier hard rule), slope 1.00
— the bulk recovers near-perfectly; a small hard-reaction tail drags the raw R².

**H5/H6/H7 — real-data amenability (CPU probes, `figures/amenability.py`).**
Ran identical probes on all four organisms in nmeth.3584 (E. coli, B. subtilis,
Y. lipolytica, hybrid).
- Low-rank (rank₉₉/N): 41 / 41 / 31 / 16 % → *validated*, all low-rank.
- Network coverage (KEGG ∩ iJO1366): 77 / 67 / 63 / 46 % → *validated*, S is buildable.
- dc/dt smoothness (median lag-1 autocorr): 0.27 / 0.41 / **0.62** / 0.24 →
  *does NOT generalise*; only **Y. lipolytica is smooth** (learnable dc/dt).
→ Decision: the best real-data fit target is **Y. lipolytica**, not E. coli
(best coverage but noisiest).

**H4 — does the autoregressive curriculum fix the glycolysis rollout?**
Single-step glycolysis fit diverges (rollout R²=−390, Pearson 0.19). Ported the
connectome-cx curriculum (per-epoch horizon ramp 10→300, soft tail-loss, LR
co-ramp, grad clip) as an opt-in trainer path (active iff
`training.n_steps_schedule` set; single-step k-recovery path untouched).
Smoke-validated, then swept 5 configs (base / tail10 / clip05 / steep / lrramp),
data reused from `graphs_data/glycolysis_yeast`.
Results (rollout, `-o test … best`):
| config | rollout R² | Pearson | note |
|---|---|---|---|
| base   | −3.2 | **0.795** | best |
| tail10 | −11.0 | 0.765 | |
| clip05 | −1040 | 0.243 | lowest *train* loss, worst rollout |
| steep / lrramp | running | | |
→ *partly*: curriculum lifts Pearson 0.19→0.80 (shape recovered) but R² stays
<0 (amplitude drifts). **Falsified**: lowest training loss ≠ best rollout
(clip05). Open: close the amplitude/scale gap on the stiff MM rollout.

---
<!-- hourly entries appended below -->

## 2026-06-05 (late) — glycolysis rollout: eval bug + DEGENERATE fit (skeptical check)

**Two figures requested** (rollout GT-vs-learned; learned k vs GT). Building them
surfaced two problems — investigated because a good result must be doubted.

**(1) `data_test` rollout metric is BUGGY.** It calls `model(dataset)` with no
`stimulus` (graph_trainer.py L1383), evaluating the model without its boundary
drive. Reproduced exactly: my free rollout WITHOUT stimulus gives per-met
R²=−3.39 / Pearson 0.79 ≈ the official −3.2/0.795. So the whole H4 "rollout
R²=−390/−3.2, Pearson 0.19→0.80" narrative was an **eval artifact**. The correct
rollout (`figures/glyco_rollout.py`, passes `stimulus=stim[t]`): glyco_ar_base
global R²=0.993 Pearson=0.997, per-met R²=0.145 Pearson=0.891.

**(2) But it's a DEGENERATE / in-sample result — NOT mechanism learning.**
Skeptical checks:
- *In-sample*: `n_runs=1`, so train and rollout use the SAME trajectory (rank 6/20).
  High in-sample rollout R² can be memorization. A held-out trajectory (seed 777)
  came out rank-1 (degenerate stimulus) → not a usable generalization test.
- *Leak-resistant test* = parameter recovery (`figures/k_recovery.py glyco_ar_base`):
  **Vmax R²=0.001, slope=0.06, 22/30 (73%) outliers → FAIL.** The model reproduces
  the dynamics while recovering ESSENTIALLY NONE of the true Vmax. Right dynamics,
  wrong mechanism = classic degeneracy. ⇒ it would NOT generalize.
- *Global ≫ per-met R²* (0.99 vs 0.15) confirms the rollout is carried by a few
  high-variance metabolites; most are not tracked.

**Verdict.** H4 corrected: the AR curriculum fixes rollout *instability* but does
NOT solve the inverse problem on real MM kinetics — the glycolysis fit is
degenerate (Vmax R²≈0). clip05 remains worst on BOTH metrics (per-met R²=−17.5,
Vmax fails too) — the "lowest train loss ≠ best rollout" falsification survives.
Config ranking by correct per-met R²: base 0.145 > tail10 0.058 ≫ clip05 −17.5.

**Next:** the real open problem is parameter identifiability on stiff MM dynamics,
not rollout stability. Needs: a genuine held-out trajectory with comparable rank,
and an identifiability fix (the model fits dc/dt with the wrong k via MLP_sub /
homeostasis compensation — the same scale/compensation degeneracy as the toy model,
but unbroken here). Created CLAUDE.md (repo gotchas) this session.

## 2026-06-05 (late) — MLP diagnostic: MLP_sub cannot represent c^2

Plotted both learned MLPs vs ground truth (figures/mlp_functions.py).
- **MLP_node (homeostasis) — EASY:** flat ~0 for the toy (true lambda=0), as expected.
- **MLP_sub (kinetic law) — the HARD one, and it FAILS the curvature.** Quantified
  growth over c in [1,9] (toy, mass-action):
    |s|=1: learned f(9)/f(1)=8.3  vs true 9^1=9    -> learned well.
    |s|=2: learned f(9)/f(1)=8.5  vs true 9^2=81   -> learned as ~LINEAR.
  MLP_sub barely distinguishes |s|=1 from |s|=2: it outputs ~the same near-linear
  curve for both, missing the quadratic entirely. So reactions with |s|=2 substrates
  get the WRONG functional form -> their k cannot be recovered -> this is the ROOT of
  the identifiability degeneracy (the 4.3% toy outliers and the glyco Vmax R2~0).
**Why:** c^s has high curvature / large dynamic range in linear space; an MLP with
bounded activations linearises it. In LOG space, log(c^s)=s*log(c) is LINEAR (slope s)
-> trivial. Candidate fix to discuss: feed log(c) / predict log(rate), or constrain
MLP_sub = c^{g(s)} with a tiny learned exponent map g(s). Figures: mlp_k_recovery_winner.png,
mlp_glyco.png. Discussing the fix with Cedric before implementing.

## 2026-06-05 (late) — MLP_sub alternatives implemented + launched

Implemented opt-in substrate-function variants (config.graph_model.substrate_func_type):
- **logspace**: g = exp(MLP([log c, |s|])) — c^s is linear in log-c, so curvature
  & dynamic range become trivial. The recommended first fix.
- **powerlaw**: g = c^{a(|s|)}, tiny learnable exponent map — exact for mass-action
  (sanity upper bound; will NOT fit MM saturation).
Default 'mlp' path unchanged; smoke-tested (LogSubstrate builds, finite output).
Configs k_logspace (cuda:0, RUNNING), k_powerlaw (queued) — toy, reuse k_recovery_winner data.

**Eval plan per experiment:** (1) figures/mlp_functions.py <cfg> — does MLP_sub now
bend like c^2 (|s|=2 growth ~81x vs the plain-MLP 8.5x)? (2) figures/k_recovery.py
<cfg> — does k recovery raw R² and the 4.3% outlier tail improve? The plain-MLP
baseline: |s|=2 collapses to ~linear, raw R²=0.74 / 4.3% outliers.
Then glyco_logspace on the SBML model (does log-space help the MM case / Vmax recovery?).

NOTE: station now SHARED with Cedric's zebrafish (connectome-cx) jobs — keep <=2
metabolism GPU jobs running concurrently; prefer the idle GPU.

## 2026-06-05 (hourly) — steep finished; degeneracy is universal; k_powerlaw launched

- **glyco_ar_steep** DONE (5 epochs). Correct rollout (with stimulus): per-met
  R²=0.521 Pearson=0.871, global R²=0.991 — the BEST AR rollout so far
  (base 0.145, tail10 0.058, clip05 −17.5 per-met). BUT Vmax recovery still FAILS:
  raw R²=0.003, 19/30 (63%) outliers, slope −0.07. ⇒ even the best-rolling config
  recovers NO mechanism. Rollout quality and Vmax recovery are decoupled — the
  degeneracy is universal across the AR sweep, not a tuning issue. Reinforces the
  PDF verdict (no PDF change; confirmation only).
- AR sweep ranking by per-met rollout R²: **steep 0.52 > base 0.145 > tail10 0.058
  ≫ clip05 −17.5**; all have Vmax R²≈0.
- **k_logspace** still early (2%, k R²=0.003 — first eval ~19800 iters). **lrramp**
  running (epoch 4/7). Launched **k_powerlaw** (cuda:1) — the exact-mass-action
  sanity bound for MLP_sub. Station clear of zebrafish this cycle.
- Next: k_logspace / k_powerlaw evals (mlp_functions.py: does MLP_sub bend like c²?
  k_recovery.py: does k improve over plain-MLP 0.74/4.3%?). If log-space breaks the
  degeneracy on the toy, build glyco_logspace next.

## 2026-06-05 (hourly) — MLP_sub fix is WORKING (interim, be-skeptical pending)

Mid-training k recovery (toy, S given), vs plain-MLP baseline raw R²=0.74:
- **k_logspace: k R²=0.871 @ 51%** (still climbing) — log-space MLP_sub.
- **k_powerlaw: k R²=0.801 @ 49%** — exact c^a(s) form.
Both already beat plain-MLP 0.74 at half an epoch. Strong signal that MLP_sub's
inability to represent c^2 was the bottleneck; reparameterising it (log-space /
power-law) recovers k far better. k-recovery is leak-resistant (can't memorise the
right k), so this is credible — but NOT yet final/verified.
DO NOT claim until: (1) runs finish, (2) figures/mlp_functions.py confirms MLP_sub
now bends like c^2 (|s|=2 growth ~81x, not 8.5x), (3) check outlier % (10% rule).
glyco_ar_lrramp still running (AR glyco; k R²~0 expected, MM not mass-action).
GPUs full (k_logspace cuda:0; k_powerlaw+lrramp cuda:1). No launches. Next cycle:
final eval + curvature confirmation; if confirmed, build glyco_logspace (MM case)
and promote a polished figure to the PDF.

## 2026-06-05 (hourly) — log-space helps k recovery, but my c^2 story was WRONG

FINISHED: k_logspace, k_powerlaw (toy). Skeptical confirmation:
- **k recovery (leak-resistant):** k_logspace raw R²=0.887, trimmed 0.983, 6/256
  (2.3%) outliers, slope 1.00, PASS. k_powerlaw raw 0.804, 3.1% outliers, PASS.
  Both beat plain-MLP (0.74 raw, 4.3% out). REAL improvement.
- **Curvature check (surprise):** MLP_sub |s|=2 growth over c in[1,9]: logspace 7.1,
  powerlaw 12.3 — vs true 81. Still essentially linear / ignores s. So the gain is
  NOT from fixing c^2.
- **ROOT CAUSE of the surprise:** the toy stoichiometry is **100% |s|=1** (510 edges,
  ZERO |s|>=2). So there are NO quadratic reactions in the toy. My earlier
  "MLP_sub collapses c^2 -> root of degeneracy" was probing UNTRAINED EXTRAPOLATION
  at |s|=2; it is NOT a real failure on the toy. **Self-correction: the toy
  improvement comes from log-space better-CONDITIONING the c^1 (linear) fit
  (log compresses dynamic range -> k more identifiable), not from curvature.**
- Implication: the c^2 / nonlinear-kinetics question is only testable where the
  kinetics are actually nonlinear -> GLYCOLYSIS (MM, saturating). Launched
  **glyco_logspace** (glyco_ar_base + substrate_func_type=logspace, cuda:0) to test
  whether log-space helps the REAL Vmax degeneracy (was R²~0). lrramp/steep/etc all
  still Vmax R²~0.
- Promoted to PDF: ledger rows (log-space improves toy k; toy-c^2-failure FALSIFIED).
  NOT adding the |s|=2 extrapolation figure (would mislead). glyco_logspace verdict
  next cycle.

## 2026-06-05 (hourly) — lrramp finished (AR sweep complete); glyco_logspace mid-run

- **glyco_ar_lrramp** DONE. Correct rollout: per-met R²=0.551 Pearson=0.879 (best of
  the sweep, edging steep 0.521). Vmax recovery: raw R²=0.006, 24/30 (80%) outliers,
  slope −0.14 → FAIL. Same degeneracy.
- **AR sweep COMPLETE (all 5).** Per-met rollout R²: lrramp 0.551 > steep 0.521 >
  base 0.145 > tail10 0.058 ≫ clip05 −17.5. **ALL have Vmax R²≈0** — degeneracy is
  universal, decoupled from rollout quality. No PDF change (confirms existing verdict).
- **glyco_logspace** still training (epoch 4/7) — the KEY test: does log-space MLP_sub
  break the glyco MM Vmax degeneracy? Verdict next cycle.
- GPU1 idle; held (glyco_logspace is the active key experiment; not half-starting the
  bigger Rung-3 Y. lipolytica real-data build unsupervised — that's the next major
  thread once the MLP_sub line concludes). No zebrafish this cycle.
- Open idea worth queuing: test log-space on a regime that ACTUALLY has |s|=2 reactions
  (toy is all |s|=1) to properly test the curvature hypothesis — needs a new dataset
  with |s|>=2 stoichiometry.
- 2026-06-05 (hourly): glyco_logspace on final epoch (6/7, T_epoch=300); KEY MM Vmax test finishes next cycle. GPU1 idle, held. No zebrafish.
