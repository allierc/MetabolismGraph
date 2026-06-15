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

**ROLLOUT METRIC = per-metabolite Pearson, NOT pooled** (Cedric's catch, 2026-06-08):
pooled Pearson is flattering (dominated by between-metabolite levels; rewards staying
bounded). Always lead with per-met. Single-step=0.47, naive recurrent=0.18 (dynamically dead).

**24h FOCUS (Cedric, 2026-06-08 ~17:00 -> ~17:00 next day): sweep RECURRENT TRAINING SCHEMES.**
Pause the D1/D2/D3 rotation; spend the next 24h finding a recurrent scheme with a TRUE good
rollout (per-met > 0.47 AND stable AND k preserved). Track everything in
`docs/recurrent_sweep_log.md` (results table + backlog + running best). Cedric reviews at the
12h mark. Each cycle: evaluate finished runs by per-met Pearson + k-recovery, append results,
launch the next backlog schemes (adapt toward what works), keep <=2 jobs/GPU, commit.

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

## 2026-06-08 (cycle 4) — D1 EVALUATED (curriculum = degeneracy) + D3 launched

**D1 result (toy_recurrent finished, 10 epochs, T ramp 1->1000).** Evaluated via
the dashboard + a direct corr(true,learned log_k) probe (leak-resistant).
HYPOTHESIS (D1): the recurrent n_steps curriculum fixes the rollout divergence.
- VERIFIED for rollout: the free trajectory **no longer diverges** over the full
  window (no blow-up to t=200), convergent-window Pearson **0.74 -> 0.83**.
- BUT k-recovery **COLLAPSES**: raw R²=0.004, 90% outliers; corr(true,learned
  log_k) **0.87 -> 0.07** (learned std 0.69 vs true 0.29, mean shifted -1.5->-0.3).
  Verified NOT a correction bug (raw corr 0.07 too). And MLP_node, correctly ~0
  under single-step, **drifts to a spurious nonzero function** (dashboard panel b).
- => the multi-step objective admits a STABLE DEGENERATE solution: wrong k +
  spurious homeostasis reproduce the trajectory without the mechanism. Same
  right-dynamics/wrong-parameters degeneracy as glycolysis. **Stability and
  identifiability are in tension**; the naive curriculum buys one by sacrificing
  the other. FALSIFIED the implicit "curriculum = strictly better fit".
- Next D1 iteration (future cycle): anchor k from the single-step fit (freeze /
  regularize log_k) while ramping the horizon, to keep recovery AND get stability.
- Promoted to PDF: new paragraph in the toy section (the curriculum tradeoff).

**D3 advanced (implemented + launched).** Added an analytic external stimulus.
- Generator: new `external_input_type="modulation"` branch — per-input sinusoid
  u_k(t)=A sin(2*pi f_k t + phi_k) (distinct freq/phase) written to **x[:,4]**
  each frame, applied via `external_input_mode` (connectome convention). Added
  `external_input_amplitude` to SimulationConfig.
- `config/toy_stim.yaml` = k_recovery_winner + modulation: 20 inputs, amp 0.5,
  additive, single-step (clean controlled comparison to the no-stimulus baseline).
- Generated: col 4 carries the drive (20 active inputs, ±0.5, std 0.158);
  `external_input rank(99%)=20` — the known drive adds 20 independent modes.
  HYPOTHESIS (D3): the known time-varying drive enriches the trajectory and
  improves identifiability / k-recovery vs the no-stimulus baseline (0.74/0.07corr).
- toy_stim single-step training LAUNCHED (cuda:0), mid-flight; evaluate next D3 cycle.

## 2026-06-08 (cycle 5) — METRIC CORRECTION + recurrent-scheme SWEEP

**Cedric caught a flattering metric.** In Fig 2d the recurrent rollout looked
smooth and did NOT reproduce the GT oscillations, yet "rollout Pearson" was higher
(0.83 vs 0.74). Root cause: the dashboard reported **POOLED** Pearson (all 100
metabolites x time, ravelled), which is dominated by BETWEEN-metabolite level
differences and just rewards staying bounded. The honest **per-metabolite** Pearson
(each rollout vs its own true trace, then averaged):

| model | pooled | **per-met** | t_div |
|---|---|---|---|
| single-step (Fig1) | 0.742 | **0.474** | 1240 (time 124) |
| recurrent naive (Fig2) | 0.834 | **0.179** | 2001 (no divergence) |

=> The traces are CORRECT (not a plotting bug). The naive curriculum does NOT give
a good rollout — per-metabolite fidelity COLLAPSES 0.47->0.18: it replaces the
single-step model's correct-but-unstable oscillations with a smooth, bounded,
**dynamically dead** trajectory. The pooled metric hid this. (This is the
global-vs-per-met trap already flagged in CLAUDE.md, now biting the rollout.)
- FIXED: dashboard now reports "per-met / pooled" (per-met primary). Regenerated
  both figures. PDF: corrected the D1 paragraph + Fig1/Fig2 captions to the honest
  per-met numbers; Fig2 retitled "stable but dynamically dead".

**Lesson (Cedric): don't conclude too fast — try MANY recurrent schemes, put in
effort to get a TRUE good rollout** (high per-met fidelity AND stable AND k
preserved). Diagnosis of the collapse: long free-rollout horizon + high LR lets the
model avoid blow-up the easy way (damp everything smooth, degenerate) instead of the
hard way (learn real dynamics). Schemes launched (eval by per-met Pearson + k-recovery):
- S1 anchor-k: freeze log_k after the single-step warmup (new `anchor_k_after_warmup`
  trainer flag, lr('k')->0 for epoch>=1). RUNNING.
- S2 cap-200 / S3 cap-500: cap the training horizon (avoid the 1000-step flatness pressure).
- S4 lr-decay: ramp k/sub LR down over the curriculum.
- S5 tail-loss 0.5; S6 warmup-3 (3 single-step epochs); S7 kitchen-sink (anchor+cap200+lrdecay).
- **S8 = Cedric's zebrafish recipe** (connectome-gnn-cx): 21 epochs (n_step=1 warmup
  prepended), n_steps ramp to an 800-step PLATEAU held for ~11 epochs, LR decay
  5e-4->5e-5. The long low-LR plateau at a moderate horizon is the likely key. RUNNING.
- All on the 2 GPUs, <=2 jobs/GPU (two serial queues + zebra). Evaluate next cycle.

## 2026-06-08 (cycle 6) — log-exp panel-a test (NEGATIVE) + n_runs lever experiment (overnight)

**Q1 (Cedric): can the log-exp transform learn the c^2 quadratic in toy dashboard panel a?**
k_logspace (toy k_recovery_winner data, substrate_func_type=logspace) was already trained.
Dashboard panel (a): the learned |s|=2 curve stays FLAT, far below true c^2 (red dashed -> 80).
=> **NO.** The toy has ZERO |s|=2 data, so the |s| input to MLP_sub is untrained at |s|=2;
no parameterization (plain/logspace/powerlaw) can produce c^2 without quadratic signal. My
"log-space extrapolates in log c" idea is FALSIFIED: log-space linearizes the CONCENTRATION
dependence at fixed |s|, but the |s|-dependence is still learned and unconstrained for unseen |s|.
BUT log-space IS a lever on the other axes (toy):
| toy | plain MLP | log-space |
|---|---|---|
| k-recovery raw R2 | 0.74 | **0.89** |
| trimmed / %out | 0.98 / 4% | 0.98 / 2% |
| rollout | diverges@124 | **stable, no divergence** |
| rollout per-met Pearson | 0.47 | 0.32 |
To learn c^2 you need |s|=2 reactions in the data (mixed_s2, D2: logspace lifted curvature
24->32, still capped by high-|s| signal scarcity).

**Q2 (Cedric): is n_runs a lever? train toy with 10 runs of 2881 timepoints instead of 1.**
Clean controlled design: generate ONE 12-run dataset (same network S), train on 1 / 3 / 10 of
the runs, hold out run 11 for a true generalization test. Launched overnight (controller
/tmp/run_nruns.sh, results -> /tmp/nruns_results.txt). Configs: toy_runs_data (gen, n_runs=12),
toy_1run / toy_3runs / toy_10runs (dataset=toy_runs_data, n_runs=1/3/10). Eval per config:
k-recovery + in-sample rollout (dashboard) AND held-out run-11 rollout (glyco_rollout.py
<cfg> toy_holdout_run11). Hypothesis: more diverse runs raise the effective rank seen in
training -> better k-recovery and/or a stable, generalizing rollout. Baseline: 1 run = k R2 0.75,
rollout diverges. Generation done (12 runs), held-out staged, toy_1run training now. Eval next cycle.

## 2026-06-09 — SUMMARY: levers tested with inconclusive / negative results

Three levers were tried to get a TRUE good rollout (per-met Pearson > single-step's 0.47,
stable, k preserved) on the toy. None succeeded on the stimulus-free toy:

1. **Recurrent training curriculum (D1) — NEGATIVE, conclusive.** 8 schemes (naive 1->1000,
   horizon caps 200/500, LR-decay, soft tail, 3x warmup, anchor-k, kitchen-sink, and Cedric's
   zebrafish 800-plateau). ALL collapse to a smooth-degenerate fixed point: per-met 0.09-0.23
   (vs 0.47 single-step) AND k-recovery destroyed (raw R^2 0.00-0.07, 73-100% outliers). The
   more stable a scheme (no divergence), the WORSE its k -> "stability = degeneracy". Likely
   cause: the toy is an AUTONOMOUS linear system (all |s|=1, dc/dt=Mc) with no external drive
   to anchor a long rollout, so the multi-step loss damps the oscillations.

2. **Number of runs (1/3/10) — INCONCLUSIVE (confounded).** k-recovery 0.85/0.72/0.80 (non-
   monotonic); held-out rollout poor for all. But the training BUDGET was held fixed, so more
   runs = less training/run (the 3-run case looks undertrained). Not a clean test; needs the
   budget scaled with n_runs to conclude.

3. **log-exp transform to learn c^2 (panel a) — NEGATIVE, expected.** The toy has zero |s|=2
   data, so no parameterization can produce c^2. (Log-space DID help k-recovery 0.74->0.89 and
   stability, just not the quadratic.)

DECISION (Cedric): the evidence points to the missing external drive, so pivot to **recurrent
training WITH the external stimulus, 10 runs per test** (D1+D3 combined). Launched:
toy_stim10_data (11 runs, per-run modulation drive; run 10 held out) + 4 tests on 10 runs each
(single-step control, naive recurrent, zebrafish recipe, cap-200). Fixed the rollout eval to
FEED the given drive (col 4) at each step (was frozen at frame 0) and made the drive vary per
run. Eval by per-met Pearson + k-recovery + held-out run 10.

## 2026-06-10 — PRELIMINARY read of recurrent+stimulus (mid-training, Cedric asked)

The recurrent+stimulus runs are still training (single-step control finished Jun 9 and is
already in the PDF as fig:toy_stim; naive/cap200/zebra all alive on GPU but only ~1-2 of
10/10/21 epochs after 1d9h — slow because the long sequential rollout dominates). Ran the
dashboard on the **current mid-training checkpoints** of the two furthest-along (cap200, naive)
for an early signal. NUMBERS ARE PROVISIONAL.

| run (mid-training ~1-2/10 ep) | k-recovery raw R2 | trim / %out | rollout per-met Pearson | stable? |
|---|---|---|---|---|
| **cap200** (horizon capped at 200) | **0.80** | 0.97 / 3% | **0.30** | yes, no divergence, traces alive |
| naive (uncapped ramp 1->500) | 0.00 | 0.00 / 0% | nan (dead flat) | stable but dynamically dead |

- **cap200 is the FIRST recurrent scheme to PRESERVE k** (0.80, ~ the single-step+stim 0.78)
  while staying stable AND alive (panel d tracks the GT oscillations under the drive). Every
  stimulus-free recurrent scheme destroyed k (R2 0.00-0.07). => the external drive anchors k
  under recurrent training **provided the horizon is capped**. Rollout per-met 0.30 still below
  single-step's 0.47, but it's only 1-2 epochs in and may climb.
- **naive (uncapped) still collapses** to the dead degenerate fixed point (k destroyed, rollout
  flat; pooled=1.00 is the flat-trace artifact, per-met nan). Stimulus alone is NOT enough; the
  long uncapped horizon still drives the smooth-degenerate collapse.
- Dashboards saved: figures/metabolism/toy_dashboard_toy_stim10_{cap200,naive}.png.
- PDF: added a provisional sentence to the fig:toy_stim paragraph (mid-training caveat flagged).
- NOTE (device): the 3 jobs were NOT on CPU — they hold cuda:0 (cap200) / cuda:1 (naive+zebra,
  oversubscribed). 99% CPU is the sequential python rollout loop, not a fallback. Slowness is
  inherent to the long unrolled horizon, not the device. zebra not evaluated (epoch 1/21, too early).

### Optimization: torch.compile the rollout dx/dt core, then kill + relaunch (Cedric)

Root cause of the slowness: the AR rollout calls the model once per step on a TINY graph, so it's
CPU-launch-bound (hundreds of tiny kernels/step launched from python). The time loop is
autoregressive => CANNOT be vectorized away (true data dependency); the lever is per-step launch
overhead. The model.forward is plain tensor ops (cat/MLP/index_add_/softplus), NOT PyG
MessagePassing => very torch.compile-friendly.

Changes (live in the editable install):
- Metabolism_Propagation: extracted forward body into `compute_dxdt(x, stimulus)` (pure tensors,
  no pyg_Data wrapper); forward() now delegates to it. Numerically identical.
- graph_trainer: build `dxdt_core = torch.compile(model.compute_dxdt, dynamic=False)` when
  `training.compile_rollout` and cuda; route all 3 hot-loop forwards (AR / recurrent / single-step)
  through it, dropping the per-step pyg_Data rebuild + redundant x.clone().
- config: new `TrainingConfig.compile_rollout` (default false); set true in cap200/naive/zebra.

Verified before relaunch (cuda:0):
- forward vs compute_dxdt, eager vs compiled: max|diff| = 6e-8 (float32 roundoff) — identical.
- fwd+bwd benchmark (the real training cost): eager->compile-default = 378->243 ms @T=200 (1.55x),
  1575->809 ms @T=500 (1.95x). Win GROWS with horizon (good for zebra-800/naive-500).
- mode="reduce-overhead" (CUDA graphs) is CATASTROPHIC here (23s/72s) — recaptures every step
  because the rollout allocates a fresh x each step. Use default mode.

Killed the 3 CPU-launch-bound jobs (PIDs 3209235-37; backed up their best ckpts to
/tmp/prelaunch_*_best.pt) and relaunched optimized: cap200+naive on cuda:0, zebra (heaviest,
800-plateau) alone on cuda:1. compile engaged on all 3, no errors, ~50 it/s in the T=1 warmup.
Code changes NOT yet committed.

### 12h-bounded redesign (Cedric: "make versions that last 12h max")

Predicting the high-horizon iteration cost is too unreliable to hit a 12h target by tuning
iteration counts (per-iter time grows ~linearly with rollout horizon, but fixed python overhead
dominates at low T => no clean scaling law). So added a HARD wall-clock time-box plus compressed
the configs to front-load the useful horizon:
- graph_trainer: new `TrainingConfig.max_train_hours` (0=off). When >0, the iteration loop saves a
  final checkpoint (best_model_{epoch}_{N}.pt + best_model_{epoch}.pt + loss.pt) and returns as
  soon as elapsed > budget. Guarantees a bounded run regardless of horizon cost.
- compressed all 3 configs: data_augmentation_loop 5500->600 (Niter 396000->43200, ~9x smaller);
  schedules front-loaded so they REACH the cap horizon in the first ~few epochs and hold:
  cap200 [1,50,100,200x5] (8 ep), naive [1,100,300,500x5] (8 ep), zebra [1,100,300,500,800x4] (8 ep,
  lr decay trimmed to 8). All max_train_hours=12.
- GPU: cap200 (PRIMARY, the only scheme that preserved k) ALONE on cuda:0 for full throughput;
  naive+zebra share cuda:1 (secondary; naive already known to collapse).
Killed the previous optimized jobs (491408/485/586), relaunched 12h-boxed (pids 768291/364/465).
compile engaged, ~60 it/s at T=1, Niter=43200. Each run is now guaranteed <=12h; the time-box
stops mid-curriculum if needed but the front-loaded schedule ensures it trained at the cap first.
Eval at ~12h via toy_dashboard.py on the latest checkpoint. Code still NOT committed.

## 2026-06-11 — 12h-boxed recurrent+stimulus FINISHED + evaluated (NEGATIVE) + PDF fig

All 3 time-boxed runs hit the 12h budget cleanly (time-box worked) and stopped with checkpoints.
12h only bought training up to a T=100 horizon (none reached their caps: 200/500/800). Dashboard
eval (toy_dashboard.py, latest checkpoint):

| run | k raw R2 | trimmed | %out | rollout per-met | pooled | stable |
|---|---|---|---|---|---|---|
| cap200 (cap-100) | 0.44 | 0.92 | 14% | 0.31 | 0.77 | yes |
| naive | 0.53 | 0.89 | 32% | 0.39 | 0.81 | yes |
| zebra | 0.28 | 0.28 | 84% | 0.26 | 0.70 | yes |

BEST = cap200 (best k-recovery, closest to the <=10% outlier gate; the scheme the narrative
centers on). VERDICT (negative): the stimulus-anchored curriculum does NOT beat single-step+
stimulus. cap200 rollout per-met 0.31 only MATCHES single-step's 0.33 (no gain), while k-recovery
DEGRADES from 0.78/4%out to 0.44/14%out (now FAILS the <=10% gate). The earlier "k=0.80"
preliminary read was at the warmup/short horizon; once training reaches T=100 the curriculum
erodes k -- the stability/identifiability tension persists, only softened by the drive (not
removed). The stimulus DOES prevent the dead-collapse the stimulus-free curriculum suffered
(per-met <=0.23, k destroyed), but that's the only win.

=> Single-step + external stimulus remains the best toy configuration.

PDF: replaced the provisional "encouraging" sentence with the completed negative verdict; added
fig:toy_stim_rec (cap200 dashboard) after fig:toy_stim. Compiles, 18pp. Code still NOT committed
(trainer/model/config: compute_dxdt split, torch.compile rollout, max_train_hours time-box).

## 2026-06-11 (eve) — INTRINSIC-NOISE sweep DONE + companion figure in PDF

Tested the flyvis "intrinsic noise breaks identifiability degeneracy" lever on the toy.
Swept noise_model_level (SDE process noise in generation) on k_recovery_winner (single-step,
S given), sigma in {0,0.01,0.02,0.03,0.05,0.07}, full Fig-1 budget. Eval: k-recovery (authoritative
k_recovery.py) + rollout vs the noise-free twin (toy_noise_000, same seed=42).

| sigma | k raw R2 | trim | %out | rank99 | rollout per-met (vs clean) |
|---|---|---|---|---|---|
| 0.00 | 0.766 | 0.979 | 6% | 35 | 0.22 |
| 0.01 | 0.783 | 0.988 | 5% | 37 | 0.06 |
| 0.02 | 0.597 | 0.989 | 6% | 39 | 0.31 |
| 0.03 | 0.642 | 0.990 | 6% | 40 | 0.06 |
| 0.05 | 0.758 | 0.987 | 5% | 46 | 0.13 |
| 0.07 | 0.322 | 0.322 | 95% | 47 | 0.01 |

VERDICT (honest, skeptical): intrinsic noise is BENIGN here, not a fix.
(i) data diversity DOES rise (rank 35->47 monotonic). (ii) but k-recovery is merely ROBUST: trimmed
R2 0.98-0.99 / 5-6% out for sigma<=0.05 (raw R2 0.60-0.78 = within seed-draw noise, no clean gain);
sigma=0.07 COLLAPSES (95% out) as the noise corrupts the trajectory. (iii) deterministic model trained
on noisy data partially recovers the noise-free dynamics (panels c,d) but stays a poor long-horizon
simulator. WHY benign: with S given, k is ALREADY identifiable at sigma=0 (k_recovery_winner=0.77) so
there's NO k-degeneracy for noise to break -- unlike flyvis where noise broke a WEIGHT degeneracy.
A real test of the flyvis mechanism needs a k-degenerate regime (S unknown) -> future work.

GOTCHA caught: my first toy_noise_sweep.py k_metrics computed raw R2 differently from the
authoritative figures/k_recovery.py (gave 0.54 baseline vs true 0.77). Refactored to call
_plot_rate_constants_comparison directly -> numbers now match. (Always cross-check a reimplemented
metric against the authoritative one.)

PDF: added fig:toy_noise (companion to Fig 1) + paragraph "Intrinsic noise as an identifiability
lever" at end of toy section. Compiles, 19pp. Panels: (a) k-recovery vs sigma, (b) rollout per-met vs
sigma (clean vs noisy GT), (c) noisy training data, (d) rollout vs noise-free GT. Configs
toy_noise_{000..007}, script figures/toy_noise_sweep.py. sigma>=0.1 diverges (excluded). Code +
docs still uncommitted.

## 2026-06-15 — RUNG 2/3 launched: real topology + imposed MM kinetics (S given)

Cedric: continue the paper autonomously, cuda:0=Rung 2, cuda:1=Rung 3, with checkpoints.
Blocker found: yeast-GEM / iJO1366 / e_coli_core are FBA models with ZERO kinetic laws
(can't simulate dynamics, no GT k); SBML libs (cobra/libsbml/roadrunner) all unavailable.
Resolution (Cedric's pick): impose synthetic MM kinetics on the REAL topology -> GT-bearing
k-recovery (matches the PDF's "S-given on realistic topology" rung).
- NEW: library-free FBA-SBML topology loader (src/.../generators/sbml_topology.py): parse S
  from raw XML, drop biomass/exchange/lumped pseudo-reactions (|s|<=4), cytosolic central-carbon
  subgraph for genome-scale. Generator hook `topology_sbml`. MM model imposes random Vmax/Km = GT.
- ecoli_core_mm (cuda:1, "Rung 3"): e_coli_core 72 met x 71 rxn (orders 1-4).
- yeast_central_mm (cuda:0, "Rung 2"): yeast-GEM cytosolic subgraph 208 met x 120 rxn (orders 1-4).
- Both: S-given (freeze_stoichiometry), single-step, n_runs=4 (diverse ICs), **3h time-box +
  checkpoints every Niter/20** so a stuck/long run is never empty-handed.
- Generation STABLE (mass-conserving, finite, bounded) but **activity rank only ~2** (closed
  networks relax to steady state fast -> low dynamical information, same as glycolysis). So
  expect k-recovery to be challenged/degenerate; that itself is the finding for real closed
  topologies. Trainings running; evaluate Vmax-recovery + rollout on completion.

### 2026-06-15 (eval cycle) — Rung 2/3 crash fixed, relaunched
First launch crashed AT STARTUP (no training, no data lost): config files still had
n_reactions=30 (inherited from glyco_ar_base) while the GT models have 71 (e_coli) / 120
(yeast) reactions -> gt_model.load_state_dict size mismatch. The generator overrode the size
in-memory but didn't persist it to the config. Fixed: set n_metabolites/n_reactions to the real
topology sizes (ecoli 72/71, yeast 208/120) in the configs. Relaunched both; now genuinely
training (past GT-load), checkpoints landing (best_model_with_3_graphs_*), 3h time-box. Evaluate
Vmax-recovery + per-met rollout on completion (cron ae73821c).

### 2026-06-15 (eval cycle 2) — Rung 2/3 PRELIMINARY (mid-training, epoch ~30-35/40)
Both still training (under the 3h box); preliminary Vmax-recovery on the latest checkpoints:
- ecoli_core_mm (rank-2): Vmax raw R²=0.02, 71/71 (100%) outliers, slope -0.34 -> FAIL
- yeast_central_mm (rank-2): Vmax raw R²=0.00, 120/120 (100%) outliers, slope -0.10 -> FAIL
=> Complete degeneracy, even worse than glycolysis (73% out). Emerging monotonic story:
identifiability tracks DYNAMICAL RANK -- toy rank-50 oscillatory recovers k (0.74);
glycolysis rank-5 partially fails (73%); real CLOSED metabolic topologies relax to steady
state fast (rank-2) and carry far too little information to constrain 71/120 Vmax -> 100% fail.
NOT final (a few epochs left); next cycle does the final eval + PDF promotion. Held back from
the PDF until trainings complete.

### 2026-06-15 (eval cycle 3) — Rung 2/3 FINAL (both finished, 40 epochs)
| scheme | topology | Vmax raw/trim R² | %out | per-met Pearson | pooled | rank |
|---|---|---|---|---|---|---|
| ecoli_core_mm | e_coli_core 72x71 | 0.02 / 0.02 | 100% | 0.55 | 0.99 | 2 |
| yeast_central_mm | yeast-GEM subgraph 208x120 | 0.00 / 0.00 | 100% | 0.58 | 0.99 | 2 |
VERDICT: complete Vmax degeneracy on BOTH real topologies. Rollout looks plausible (pooled
0.99, per-met 0.55-0.58) but Vmax recovery is 100% outliers / R²~0 -- the glycolysis trap in
its most extreme form. Cause: rank-2 fast-settling closed-network dynamics carry too little
information to constrain 71/120 Vmax. Monotonic identifiability-vs-rank ladder now complete:
toy rank-50 -> k R²=0.74 (works); glycolysis rank-5 -> 73% fail; real closed topology rank-2
-> 100% fail. Promoting to PDF.

## 2026-06-15 — PHASE 1 (stimulus -> activity rank) DONE
Diagnosis confirmed (Cedric): Fig 10 traces settle to steady state (rank ~2) -> rollout 0.99
is trivial (predict flat) and the inverse problem is starved. Fix = inject external metabolite
SOURCE to hold the net off its fixed point. Added additive external drive (x[:,4]) to
PDE_MichaelisMenten (was unused). Linear-algebra: inverse problem linear in params (A theta=b);
never-varying metabolite -> null column -> unidentifiable. Linearised dc/dt=Jc+Bu: autonomous
collapses to slow eigenmodes (rank~2); driven trajectory spans controllability subspace
span{B,JB,...} -> rank grows with #driven (m) and freq richness.
RANK SWEEP (scripts/rank_sweep.py, fig:stim_rank): activity rank ~ #driven metabolites, on BOTH
e_coli (72) and yeast (208):
  m=0 -> rank 2-3 | m=5 -> 7 | m=10 -> 12 | m=20 -> 19-21 | m=40 -> 30-32 | all -> 40-47
=> GOAL (a) rank>20 achievable: drive m>=20-40 metabolites. PDF note + fig added.
NEXT (Phase 2): generate stimulus-driven data (m=40) for e_coli/yeast, train S-given, check
rollout + Vmax/k recovery improve with rank.

## 2026-06-15 — PHASE 2 launched: train on stimulus-driven (rank-38) data
Generated stimulus-driven datasets (external_input_type=modulation, additive, m=40 driven
metabolites, amp=1.0, per-run freq/phase): ecoli_core_stim (72x71) and yeast_central_stim
(208x120), BOTH activity rank99=38 (vs rank-2 closed -> goal (a) rank>20 cleared). Launched
S-given single-step training on both, time-boxed 4h + checkpoints (ecoli cuda:1, yeast cuda:0).
QUESTION: does the rank-38 stimulus-driven data lift Vmax-recovery out of the 100%-degenerate
regime (and rollout to >0.7-0.8)? If not, iterate Phase 1 (raise rank further / change drive).

### 2026-06-15 (Phase 2 eval 1) — rank-38 stimulus did NOT break MM degeneracy
ecoli_core_stim (rank 38, m=40 driven): Vmax raw R²=0.04, 100% outliers (FAIL); rollout per-met
0.59 / pooled 0.79. yeast_central_stim still training (epoch 38).
=> Activity rank>20 (goal a) is NECESSARY but NOT SUFFICIENT. Diagnostics rule out the obvious:
concentrations span Km (c[0,5.8] med 0.99 vs Km[0.32,3.14] med 1.07 -> MM curve exercised), and
the drive is moderate (|u|~0.64 vs fluxes 0.3-10 -> not drive-dominated).
LEADING HYPOTHESIS: representational, not informational. The GNN's substrate function
MLP_sub(c,|s|) is SHARED across reactions, but real MM has a different Km PER REACTION; a shared
saturation shape can't capture per-reaction Km, so per-reaction Vmax absorbs the mismatch and
stays degenerate -- regardless of data rank. The toy recovered k because mass-action has no
per-reaction shape heterogeneity. This also explains glycolysis (per-reaction Km, 73% out).
DIAGNOSTIC running: ecoli_core_stim_ma = same topology + MASS-ACTION (no per-reaction Km) +
stimulus. CAVEAT: this MA run generated at rank-3 (mass-action redistributes the drive into few
slow modes; the MM nonlinearity is what spread it to rank-38), so it's confounded -- a clean MA
test needs a higher-rank MA drive. Next cycle: evaluate MA + yeast, then decide the real
iteration (give the model PER-EDGE Km capacity, vs accept the shared-MLP_sub representational limit).

### 2026-06-15 (Phase 2 eval 2) — yeast confirms; launching the decisive Km test
yeast_central_stim (rank 38): Vmax raw R²=0.03, 120/120 (100%) out (FAIL); rollout per-met 0.58
/ pooled 0.81. So BOTH real-topology MM nets at rank-38 -> 100% Vmax-degenerate. Rank confirmed
necessary-not-sufficient on both. NOT blindly raising rank (already 38; diagnosed representational).
DECISIVE DIAGNOSTIC: ecoli_core_stim_constkm = same topology + stimulus (rank-38) but CONSTANT Km
(log_km_min=log_km_max=0, all reactions Km=1) -> a shared MLP_sub CAN represent a single saturation
shape. If Vmax now recovers, per-reaction Km heterogeneity is the confirmed bottleneck (not rank).
