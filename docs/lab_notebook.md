# MetabolismGraph — experiment lab notebook

Detailed running log of the autonomous weekend runs. Raw and chronological:
every hour the loop appends what it checked, launched, found, and decided —
including dead ends. The **polished, semi-publishable** version of the science
lives in `docs/metabolism.pdf`; this file is the unfiltered notebook behind it.

Format: one dated entry per cycle. Be concrete (configs, metrics, file paths),
honest about failures, and note the *decision* taken.

---

## STANDING INSTRUCTIONS — read at EVERY checkpoint (Cedric, 2026-06-08, 2-day toy effort)

Work the **toy model** along THREE directions for two days. **SWEEP between them** each
cycle (rotate D1 -> D2 -> D3 -> D1 ...) so we don't get stuck and we see the same problem
from different perspectives. Scientific method throughout: **hypothesis -> verify ->
falsify**. Update BOTH `docs/metabolism.pdf` (polished) and this notebook every cycle.
**Re-read this notebook AND the PDF at the start of every checkpoint.**

- **D1 — recurrent training curriculum.** n_steps schedule: epoch 1 = single-step (t->t+1,
  n_steps=1, as now), then 100, 200, 300, ... up to 1000 (the curriculum used for the
  glycolysis kinetic model). HYPOTHESIS: fixes the rollout divergence (toy currently
  diverges at time~124). Metric: rollout Pearson (convergent-window + full).
- **D2 — log-exp transform for MLP_sub.** The toy is all |s|=1, so first generate a toy
  variant WITH genuine |s|=2 reactions (Cedric's call), then apply the log-space substrate
  transform (as used for glycolysis). HYPOTHESIS: log-space lets MLP_sub learn the quadratic
  c^2. Metric: MLP_sub |s|=2 curvature (growth->81) + k-recovery R²/outliers.
- **D3 — add an external time-varying STIMULUS to the training data**, given to the inverse
  problem (not learned). Mirror connectome-gnn: column 4 of x = stimulus (same convention
  here; Metabolism_Propagation already reads external_input = x[:,4] with external_input_mode).
  HYPOTHESIS: an external drive improves convergence/identifiability. Metric: k-recovery R²
  + rollout Pearson vs no-stimulus baseline.

Sweep state is tracked by the dated entries below (last entry says which direction was last
advanced; do the next in the rotation). Loop cadence: hourly.

**Result figures = ONE dashboard** (Cedric, 2026-06-08): show off each fit as a single 2x2
dashboard (`figures/toy_dashboard.py <cfg>`: a=MLP_sub, b=MLP_node, c=k-recovery, d=rollout),
not separate figures. Config-parameterised — use it for D1/D2/D3 results in the PDF too.

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

## 2026-06-05 (hourly) — DECISIVE: log-space does NOT fix glyco Vmax (degeneracy is identifiability)

glyco_logspace DONE (7 epochs). Leak-resistant Vmax recovery: raw R²=0.044, 23/30
(76.7%) outliers, slope −0.32 → STILL FAIL (baseline plain-MLP glyco_ar_base:
0.001/73%). Rollout per-met R²=0.474 Pearson=0.867 (fine, in-sample, as always).
**Verdict: reparameterising MLP_sub (log-space) does NOT break the glycolysis MM
Vmax degeneracy.** So the glyco failure is NOT an MLP_sub representation problem.

Synthesis of the MLP_sub line:
- TOY (mass-action, all |s|=1): log-space improves k 0.74→0.89 — CONDITIONING gain.
- GLYCO (real MM, saturating): log-space gives Vmax 0.001→0.044 — no real change.
⇒ The glyco degeneracy is **identifiability**, not representation: a single in-sample
trajectory (rank ~6/20) + homeostasis compensation + isoenzymes (3 HXK, 3 PDC, 3 TDH
sharing flux) admit many Vmax sets with the same dynamics. MLP_sub form is not the lever.

MLP_sub line CONCLUDED. Promoted to PDF: ledger row (glyco-logspace FALSIFIED).
Next threads (NOT auto-launched — need setup): (a) break the identifiability degeneracy
with MULTIPLE glyco trajectories (n_runs>1, diverse ICs) so Vmax becomes identifiable
[note: a single holdout at seed 777 was rank-1, so need rich/diverse ICs]; (b) Rung-3
Y. lipolytica real-data fit (the best real target). GPUs free; held pending direction.

## 2026-06-05 (hourly) — launched the |s|=2 curvature experiment (controlled)

GPUs were free (MLP_sub line concluded). Set up the queued curvature test properly:
the toy was all |s|=1, so log-space's "conditioning gain" couldn't be separated from a
real curvature benefit. init_reaction gives |s| in {1,2} for RANDOM (non-cycle)
reactions, so cycle_fraction<1 brings them in (no code change).
- Generated **mixed_s2** (k_recovery_winner + cycle_fraction=0.5, S given). |s| dist:
  {1:499, 2:187, 3:15, 4:12, 5:8, 6:2} → **31% of substrate edges are |s|>=2**. Now
  there IS real quadratic+ curvature to learn.
- Launched **mixed_s2** (plain-MLP, cuda:0) vs **mixed_s2_logspace** (log-space, cuda:1),
  same data, S given. HYPOTHESIS: if curvature is the issue, log-space should beat
  plain-MLP HERE (unlike the toy, where both were ~equal because all |s|=1). Eval next
  cycle(s): k_recovery.py (raw/trimmed/outliers) + mlp_functions.py (does MLP_sub now
  bend like c^2/c^3 for |s|>=2, growth toward 81 not 8.5?).
- Configs committed; data gitignored. No zebrafish this cycle.

## 2026-06-06 (hourly) — DEAD END then fix: |s|=2 regime exploded; flux_limit stabilised it

mixed_s2 / mixed_s2_logspace (first launch) both CRASHED: "SVD did not converge in
lstsq" at the first plot checkpoint, no models saved. Diagnosed: the data was
catastrophically unstable — 99.8% non-finite, blew up to 7e35 by frame 5. Cause:
|s|>=2 mass-action reactions with flux_limit=false → autocatalytic c^2 explosion
(the toy was stable only because it was all |s|=1). The lstsq crash was downstream
of NaN concentrations.
FIX: set flux_limit=true (bounds reaction velocity). Regenerated mixed_s2 → 100%
finite, range [0, 86], activity rank(99%)=17. Relaunched both (plain-MLP cuda:0,
log-space cuda:1); training cleanly past startup, no crash.
Caveat for the curvature test: flux_limit adds a saturation on the aggregate rate,
so MLP_sub only gets clean c^s gradient in the UNsaturated regime — still a valid
test (31% |s|>=2 reactions present), just not pure power-law. Eval next cycle:
does log-space beat plain-MLP HERE (k_recovery.py), and does MLP_sub bend like c^2
for |s|=2 (mlp_functions.py)?
- 2026-06-06 (hourly): |s|=2 curvature test mid-run (~55%): log-space k R²=0.250 vs plain-MLP 0.123 — log-space ~2x ahead WHERE curvature exists (toy tied at all-|s|=1). Both low (hard regime). Verify + curvature check (mlp_functions.py) at completion next cycle. No crash.

## 2026-06-06 (hourly) — |s|=2 curvature VERDICT: log-space > plain, but both undershoot

mixed_s2 (plain-MLP) vs mixed_s2_logspace DONE (stable, flux_limit=true, rank 17).
- k recovery: plain raw R²=0.158 trimmed 0.856 (37.5% out, FAIL); log-space raw 0.259
  trimmed 0.905 (40.6% out, FAIL). Log-space better on raw/trimmed; both fail 10% rule.
- CURVATURE growth over c[1,9] (true 9/81/729 for |s|=1/2/3):
    plain:     |s|=1 7.9   |s|=2 24.4   |s|=3 28.0
    log-space: |s|=1 8.6   |s|=2 32.3   |s|=3 56.6
  ⇒ WITH |s|>=2 signal, MLP_sub DOES learn some curvature (toy couldn't — it was all
  |s|=1, extrapolation). Log-space learns MORE. But BOTH undershoot badly (c^2: 32 vs
  81; c^3: 57 vs 729). Higher powers over a wide range stay hard.
VERDICT (partial): curvature hypothesis has real support (log-space>plain WHERE
curvature exists), but reparameterising MLP_sub is only a partial fix; c^3-c^4 unsolved.
Promoted to PDF: fig:curvature (figures/curvature_compare.py, 2-panel plain vs log-space)
+ paragraph + ledger row. Note: even the BEST (log-space) fails the 10% gate here, so
the |s|>=2 regime is genuinely hard regardless of MLP_sub form.
Queued threads unchanged (need steer): Y. lipolytica real fit; multi-traj glyco identifiability.

## 2026-06-06 (hourly) — IDENTIFIABILITY test: multi-trajectory training (generator fix)

Central open problem = identifiability (single in-sample trajectory admits many k;
glyco Vmax R²~0, mixed_s2 |s|>=2 fails 37-40% outliers). Direct test: train on
MULTIPLE diverse trajectories.
- Found generation bug: init_concentration computed ONCE (outside run loop), so
  n_runs>1 produced IDENTICAL trajectories (useless). FIX (data_generator.py): run 0
  keeps original IC (single-run datasets byte-unchanged), run>0 re-inits with seed+run
  -> diverse ICs. Verified: mixed_s2_multi has 4 finite runs with different ICs
  (run0 IC [5.9,1.1,4.2,..], run1 [2.0,1.4,3.2,..]) and ranges [0,86]..[0,146].
- Launched **mixed_s2_multi** (plain-MLP, n_runs=4, |s|>=2 regime, S given), cuda:0.
  HYPOTHESIS: if the mixed_s2 failure (raw k R²=0.16, 37% outliers, single traj) is
  identifiability, 4 diverse trajectories should improve k recovery / cut outliers.
  Baseline to beat: single-traj plain-MLP mixed_s2 = 0.158 raw / 37.5% out.
Eval next cycle. GPU change is opt-in-safe (run0 unchanged). No zebrafish this cycle.
- 2026-06-06 (hourly): identifiability test mid-run (54%): mixed_s2_multi (4 traj) k R²=0.135 — so far TRACKING the single-traj baseline (final 0.16), not obviously better. If it ends ~similar with ~37% outliers, more data did NOT break the degeneracy (=> failure is curvature/representation of c³+ , not pure identifiability). Final + outlier% next cycle. No crash.

## 2026-06-06 (hourly) — IDENTIFIABILITY verdict: FALSIFIED (it's representation, not data)

mixed_s2_multi (4 diverse trajectories, plain-MLP) DONE. Head-to-head k recovery:
  single-traj mixed_s2:       raw 0.158, trimmed 0.856, 96/256 (37.5%) outliers, FAIL
  multi-traj  mixed_s2_multi: raw 0.164, trimmed 0.850, 82/256 (32.0%) outliers, FAIL
⇒ 4x diverse trajectories barely helped (outliers 37.5→32%, raw flat). **"More
trajectories breaks the |s|>=2 degeneracy" is FALSIFIED.** The failure is
REPRESENTATION-limited (MLP_sub can't fit c^3-c^4 over the range, last cycle's
curvature result), not data/identifiability-limited. NOTE: this is the SYNTHETIC
high-|s| regime; the GLYCO MM degeneracy (isoenzymes) is a separate question, not
tested here (glyco multi-traj would need rich ICs; seed-777 holdout was rank-1).

⇒ Right next move: structured form. PowerLawSubstrate g=c^{a(s)} can represent ANY
exponent exactly. Tested on the toy (all |s|=1, gave 0.80) but NOT on |s|>=2.
Launched **mixed_s2_powerlaw** (cuda:0): if a(s) learns the true exponents, it should
fix the high-|s| reactions where the MLP/log-space undershoot. Eval next cycle:
k_recovery.py + does a(s) -> [1,2,3,..]? Promoted: ledger row (identifiability FALSIFIED).
- 2026-06-06 (hourly): structured power-law mid-run (61%): mixed_s2_powerlaw k R²=0.220 — between plain-MLP 0.16 and log-space 0.26 on |s|>=2 so far. NOT yet dramatically better despite c^{a(s)} being the exact mass-action family (likely: few |s|>=3 reactions => weak signal to learn a(3),a(4),...). Final + a(s)->[1,2,3,..]? check next cycle. No crash.

## 2026-06-06 (hourly) — CLOSING the arc: even exact power-law fails (high-|s| too rare)

mixed_s2_powerlaw DONE. k recovery: raw 0.255, trimmed 0.851, 79/256 (30.9%) outliers,
FAIL. So on |s|>=2: plain 0.158/37.5%, log-space 0.259/40.6%, power-law 0.255/30.9%.
Power-law has fewest outliers but still fails; NOT better than log-space despite being
the exact mass-action family.
DECISIVE a(s) check (PowerLawSubstrate g=c^{a(s)}, can represent any exponent exactly):
  learned a(s), s=1..6: [1.00, 1.45, 1.64, 1.71, 1.73, 1.75]
  true    a(s)=s      : [1,    2,    3,    4,    5,    6]
  |s| counts in data  : {1:499, 2:187, 3:15, 4:12, 5:8, 6:2}
⇒ a(s) SATURATES at ~1.7; it learns a(1)=1 perfectly (499 reactions) and a(2)=1.45
(undershoots, 187 rxns), and completely fails s>=3 (15/12/8/2 rxns — too rare).
SYNTHESIS of the whole MLP_sub/curvature/identifiability arc:
  - representation capacity is NOT the limit (power-law has it exactly),
  - data volume / more trajectories is NOT the limit (falsified, same S),
  - the limit is SIGNAL: high-order (|s|>=3) reactions are too RARE to identify their
    exponent. The ~31% outliers ARE the |s|>=2 reactions, unrecoverable across all 3
    parameterisations. Fix would need either more high-|s| reactions or an integer-
    exponent prior (a(s)->round), not a better generic function approximator.
Promoted to PDF: paragraph + ledger row (power-law a(s) saturation). This cleanly
concludes the synthetic MLP_sub investigation. Remaining big threads (Y. lipolytica real
fit; real-glyco identifiability) still need steer.

## 2026-06-06 (hourly) — positive control is confounded; holding for steer

Tried to build a clean positive control for the scarcity conclusion: a regime RICH in
|s|=2 to see if a(2) reaches 2.0 with more signal. Generated rich_s2 (cycle_fraction=0,
flux_limit=true): |s| counts {1:493, 2:403, 3:36, 4:23, 5:11, 6:3} (49% |s|>=2, |s|=2
DOUBLED vs mixed_s2's 187) — BUT activity rank(99%)=2. cycle_fraction couples dynamic
richness with the |s| distribution: cf=1 → rich dynamics, all |s|=1; cf=0 → many |s|=2
but trivial (rank-2) dynamics with ~no signal to identify k. So this is NOT a clean test
(low k signal would confound the a(s) result). Did NOT launch it.
Reflection: my earlier "integer-exponent prior" idea also wouldn't fix the diagnosed
problem — a(2)=1.45 saturates by MAGNITUDE (signal), and snapping to nearest integer
rounds it to 1 (worse), not 2.
STATUS: the synthetic MLP_sub / curvature / identifiability investigation is COMPLETE and
fully documented (PDF + ledger). A clean positive control needs a generator change to
decouple stoichiometry magnitude from cycle structure (bigger than config). The high-value
next steps all need steer: (a) decoupled high-|s| positive control; (b) Rung-3 Y. lipolytica
real fit (the best real target); (c) symbolic regression / structural integer-exponent
prior for the real (unknown-form) kinetics. Not launching marginal/confounded experiments
autonomously — holding. rich_s2.yaml kept as a record of the attempt.

## 2026-06-06 (hourly) — reproducibility integrity check (consolidation, no new experiment)

Synthetic investigation complete; held on new synthetic experiments (clean positive
control needs risky generator surgery to keep sub/all/S edge lists consistent — not
worth it autonomously vs the already strong a(s)-saturation evidence). Instead did a
low-risk consolidation: re-ran ALL figure scripts that feed the PDF, from saved
artifacts only.
  amenability.py, curvature_compare.py, k_recovery.py (winner + glyco_ar_base),
  mlp_functions.py, glyco_rollout.py  ->  6/6 OK, 0 fail.
So every PDF figure is reproducible (figures rule satisfied). Discarded byte-churn on
the committed PNGs (deterministic re-render) and removed stray untracked eval PNGs
(k_recovery_<cfg>.png from per-cycle evals, not referenced by the PDF).
STATUS unchanged: investigation done + reproducible; high-value next steps (Y. lipolytica
real fit; real-glyco identifiability; decoupled high-|s| positive control via a small
generator change) await steer.

## 2026-06-06 (hourly) — advancing Rung-3: yeast-GEM is the right network for Y. lipolytica

Idle GPUs; advanced the best real-data thread (Y. lipolytica fit) with a low-risk,
decisive analysis: which network to build S from? Amenability used iJO1366 (E. coli)
as a common cross-organism proxy, but Y. lipolytica is a YEAST.
  yeast-GEM (885 KEGG ids) covers 127/164 (77%) of Y. lipolytica measured ions
  iJO1366  (911 KEGG ids) covers 103/164 (63%)  <- the proxy used in the figure
⇒ yeast-GEM is the proper network and raises coverage 63%->77%. A real Y. lipolytica
fit is viable on it (77% of measured metabolites sit in a curated yeast reaction net).
This sets up thread (b): build C(t) for the 127 mapped ions + the yeast-GEM S submatrix
(needs: KEGG->yeast-GEM metabolite map, extract bipartite S for the subset, decide
relative-intensity normalisation). Those choices still benefit from steer, so I did the
go/no-go (viable) but did NOT auto-build the full pipeline. Promoted: 1-sentence refinement
of the amenability caveat + ledger row. No training this cycle.

## 2026-06-06 (hourly) — Rung-3 structural go/no-go: Y. lipolytica fit is WELL-POSED

Beyond raw coverage (77%), computed the actual subnetwork the GNN would fit. Parsed
yeast-GEM (2806 mets, 1723 with KEGG; 4131 rxns). Measured Y. lipolytica KEGG set
(269 ids) maps to 401 yeast-GEM metabolites; 740 reactions touch >=2 measured mets
(1212 touch >=1). => effective measured S ~ 401 met x 740 rxn — a substantial bipartite
graph (cf. toy 100x256, glyco 20x30). So the fit is STRUCTURALLY VIABLE/well-posed.
Surfaced the key remaining setup choice: 164 ions -> 401 yeast-GEM mets = ~2.4
COMPARTMENTAL copies per KEGG (cytosol/mito/...). Options: (a) aggregate compartments
-> ~127 unique nodes (1 measurement = 1 node); (b) keep 401 nodes with a many-to-one
observation map (several nodes share one measured intensity). Plus relative-intensity
normalisation + uneven 59-frame time. These are the steer points; the go/no-go itself
is GREEN. Promoted: ledger row (subnetwork well-posed). No training; idle GPUs.

## 2026-06-06 (hourly) — Rung-3 FULLY SCOPED: a clean 164-reaction first-fit exists

Observation completeness of the Y. lipolytica measured subnetwork in yeast-GEM:
  740 reactions touch >=2 measured metabolites, but only **164 are FULLY measured**
  (all participants observed); 576 are partially measured (median 50% of participants
  unobserved). 
KEY: restricting to the 164 fully-observed reactions gives a COMPLETE inverse problem
on real data (no latent species) -- directly analogous to the synthetic/glyco setups,
and the cleanest entry point for a first real fit. The full 740-rxn network (576
partial) is the harder latent-species follow-on (flagged in PDF as incomplete observation).
=> Rung-3 is now FULLY SCOPED with a tractable plan:
  (1) right network: yeast-GEM (77% coverage);
  (2) well-posed: 401 met x 740 rxn measured subnetwork;
  (3) clean entry: 164 fully-observed reactions -> complete inverse problem.
Remaining build decisions (steer): compartment aggregation (164 ions -> ~127 unique vs
401 compartmental nodes), relative-intensity normalisation, uneven 59-frame time. The
build itself (C(t) + S for the 164-rxn subnetwork) is the next concrete step on the
user's word. Promoted: ledger row. No training; idle GPUs. Structural scoping COMPLETE.
- 2026-06-06 (hourly): no runs, no clean autonomous experiment. Synthetic science COMPLETE; Rung-3 Y. lipolytica FULLY SCOPED (yeast-GEM 77%, 401x740 well-posed, clean 164-rxn fully-observed first fit). Genuine steer-gated resting point — next moves (build the 164-rxn fit / pick a thread) need direction + a multi-hour build; not manufacturing more analyses. Holding.
- 2026-06-06 (hourly): idle, steer-gated (4th holding cycle). Work complete & build-ready; loop now idle-spinning. Recommending user pause the hourly cron or pick a thread. Holding.

## 2026-06-06 (hourly) — Y. lipolytica build RECIPE ready (execution gated on domain validation)

Reverse-engineered the exact repo data format (from glycolysis_yeast) so the real build
is execute-ready:
  graphs_data/<ds>/  needs:
   - stoich_graph.pt = dict{ 'sub':(met_sub,rxn_sub,|coeff|), 'all':(met_all,rxn_all,signed),
       optional 'stimulus_sub' }  (contiguous met/rxn indices)
   - stoichiometry.pt = dense (N x M)
   - x_list_0.npy (T,N,8): col0=idx, col1-2=pos, col3=conc, col4=ext(0), col6=type(0)
   - y_list_0.npy (T,N,1) = dc/dt (finite diff of conc)
   - metadata.pt (species_names). gt_model.pt/stimulus.npy NOT needed (trainer skips
     k-comparison when gt_model absent -> only held-out prediction). Train with
     Metabolism_Propagation (learns MLP_sub, S given); no kinetic form assumed.
  BUILD STEPS: 164 fully-observed yeast-GEM reactions -> measured metabolite nodes ->
  contiguous bipartite edges (sub/all) + dense S; C(t)=measured Y.lipolytica intensities
  for those nodes, normalised, on a uniform grid; y=dc/dt.
WHY NOT EXECUTED autonomously: real data has NO ground truth, so a network-construction
error (reaction set / sign / metabolite identity / compartment aggregation) would be
INVISIBLE (can't be GT-validated, unlike every synthetic experiment this weekend). That
is a correctness requirement, not a preference -> needs Cedric's domain check of:
  (i) compartment aggregation (KEGG-merge ~127 vs 401 compartmental nodes -- changes the
      164-rxn set), (ii) intensity normalisation, (iii) which replicate / time handling.
STATE: recipe + format ready; one command from a dataset once (i)-(iii) are set. Holding
on execution. 5th idle cycle. (Recommended pausing the cron last cycle.)
- 2026-06-06 (hourly): PAUSED the hourly cron (job 1a6ded85) after 6 idle cycles. Work complete (synthetic science closed; Rung-3 fully scoped + build-recipe ready). No value in further idle spinning; recommended pausing twice. Loop easily resumable on Cedric's word. Final state pushed.

## 2026-06-08 — TWO-DAY TOY SWEEP kicked off (D1 launched)

Set up the 3-direction sweep (standing instructions above). Studied connectome-gnn
stimulus handling for D3: it uses **column 4 = stimulus** (neuron_state.py), and
MetabolismGraph already mirrors this (Metabolism_Propagation reads external_input=x[:,4],
external_input_mode) — so D3 has a native mechanism.
- **D1 LAUNCHED**: config/toy_recurrent.yaml = k_recovery_winner data + AR curriculum
  n_steps_schedule [1,100,200,...,1000], 10 epochs, batch 4, aug 1, grad_clip 1.0,
  tail_loss 0. Smoke OK ([AR] epoch 0: T_epoch=1, single-step as requested; no crash).
  HYP: the curriculum trains for multi-step stability -> fixes the rollout divergence
  (single-step baseline: rollout Pearson 0.74 convergent-window, then diverges at time~124).
  Eval next D1 cycle: figures/glyco_rollout.py toy_recurrent (rollout Pearson + does it
  still diverge?) + k-recovery (did the curriculum hurt parameter recovery?).
- NEXT in rotation: D2 (generate |s|=2 toy, log-space vs plain-MLP).
- Hourly cron re-established for the sweep.

## 2026-06-08 (cycle 2) — figures merged into ONE dashboard

Cedric: merge figures 1/2/3 into a single all-in-one dashboard, and use that
dashboard style from now on for showing off results.
- New `figures/toy_dashboard.py <cfg>` (config-parameterised): 2x2 —
  (a) MLP_sub vs c^|s|, (b) MLP_node, (c) k-recovery scatter, (d) rollout traces.
  Toy: raw R²=0.74 / trim 0.98 / 4% out, rollout Pearson(conv)=0.74 — matches the
  three former figures exactly (no number changed, just consolidated).
- PDF: replaced fig:mlp + fig:k_recovery + fig:toy_rollout with one
  fig:toy_dashboard (full width); updated the paragraph cross-refs to panel
  letters (a,b -> functions, c -> recovery, d -> rollout). Recompiled: 13 pp, no
  undefined refs. Convention recorded in CLAUDE.md + standing instructions so
  D1/D2/D3 results render as the same dashboard.
- D1 (toy_recurrent) still training in background.

## 2026-06-08 (cycle 3) — D2 advanced: log-space vs plain MLP on the |s|=2 toy

Rotation: D1 still training (final epoch T=1000), so advanced **D2**.
HYPOTHESIS (D2): the log-exp transform g=exp(MLP(log c,|s|)) lets MLP_sub learn the
quadratic c^2 that the plain MLP cannot.

Verified the |s|=2 toy is genuine first: `mixed_s2` stoichiometry has **93/256
reactions with a single substrate at order 2** (true c^2 terms), 182 in `rich_s2` —
so there IS quadratic signal to learn (unlike the all-|s|=1 toy where c^2 was
extrapolation). All three fits (mlp / logspace / powerlaw) were already trained
(single-epoch); this cycle = the evaluation.

Curvature growth f(9)/f(1) for MLP_sub (`figures/curvature_compare.py`, reproduced
exactly vs the prior PDF numbers — no drift):
| order | plain MLP | log-space | true |
|---|---|---|---|
| c^1 | 7.9 | 8.6 | 9 |
| c^2 | 24.4 | 32.3 | 81 |
| c^3 | 28.0 | 56.6 | 729 |

Dashboards (`figures/toy_dashboard.py mixed_s2{,_logspace}`):
| metric | plain | log-space |
|---|---|---|
| raw k R² | 0.16 | 0.26 |
| trimmed k R² | 0.86 | 0.91 |
| outliers | 38% | 41% |
| rollout Pearson (conv) | **0.27** | **0.76** |

→ **PARTIALLY VALIDATED.** Log-space learns MORE curvature at every order and
nearly TRIPLES the convergent-window rollout Pearson (0.27→0.76) — matching the
shape where the data lives stabilises integration. But it still badly undershoots
true c^2 (32 vs 81) and c^3 (57 vs 729), and both fail the 10%-outlier k-recovery
gate (~40%). FALSIFIED the strong form ("log-space solves c^2"): the residual
limit is **signal scarcity of high-|s| reactions** (already established: powerlaw
g=c^a(s), which can represent any exponent exactly, also saturates a(s) at
1.0/1.45/1.64/... vs true 1/2/3/...). The fix is a structural prior or a
high-|s|-rich regime, not a better approximator.

PDF: added the rollout-Pearson gain (0.27→0.76) to the curvature paragraph; the
curvature figure + numbers were already there and now reproduced. Recompiled, 13pp.
- NEXT in rotation: D3 (external time-varying stimulus in col 4, given to inverse problem).
- D1 (toy_recurrent) finishing its final epoch; evaluate next D1 cycle.
