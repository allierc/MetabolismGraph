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
