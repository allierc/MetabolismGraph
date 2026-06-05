# Weekend autonomous run — scientific playbook

Self-contained instructions for the every-4-hours wake-up (cron). Work like a
scientist: **state a hypothesis, run the experiment, record validate/falsify.**
A clean falsification is as valuable as a positive result — log both plainly.
The reader (Cedric) reads `docs/metabolism.pdf`; keep the hypothesis ledger there.

## Two goals
1. **Reproduce the toy-model result and quantify its robustness by CV** (multi-seed).
2. **Build intuition on whether the approach is amenable to real data** — feasibility
   probes on the three real datasets, not necessarily full recovery.

## Environment & station
- `PYTHON=/workspace/.conda_envs/neural-graph-linux/bin/python`; run from `/workspace/MetabolismGraph`.
- 2× RTX A6000 (`cuda:0`, `cuda:1`). Station is shared — **check load first**:
  `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`.
  The model is tiny + Python-bound (~20% util), so run **at most 2 trainings per GPU**.
- **GPU placement:** `device: auto` always picks the most-free GPU (usually cuda:0),
  so it stacks everything on cuda:0. To use cuda:1, set `device: cuda:1` in the config
  (verified: places correctly). Alternate seeds across cuda:0/cuda:1 to balance.
- Do NOT set `CUDA_VISIBLE_DEVICES` (breaks set_device → CPU fallback).

## GOAL 1 — toy-model reproduction + CV robustness
Validated working point: leak-free k_j recovery, S given, oscillatory rank_50.
- Seed-77 reproduction: config `k_recovery_winner` (running; log `/tmp/k_recovery_train.log`,
  out `log/k_recovery_winner/`). When done: `$PYTHON GNN_Main.py -o test k_recovery_winner best`;
  record final `k R²` (grep "k R²" last value) + outliers/slope from the rate-constants plot.
- **CV configs** `config/k_cv_s<seed>.yaml` (seeds 42,79,7,123) reuse the SAME data
  (`dataset: k_recovery_winner`) and vary only `training.seed` + `device`. Launch with
  `nohup $PYTHON GNN_Main.py -o train k_cv_s<seed> > /tmp/k_cv_s<seed>.log 2>&1 &`.
  No `generate` needed (data already exists).
- **H1**: the winner reproduces k R² ≈ 0.87 on a fresh seed-77 run. (testing)
- **H2 (robustness)**: across ≥5 seeds, k R² = mean ± std with std ≈ 0.2 (documented
  intrinsic variance). Record every seed's R²; report mean/std → validates or falsifies
  that the working point is reproducible vs seed-lucky.
- **H3 (optional)**: ensembling k across seeds (median log k) beats the single-run mean.

## GOAL 2 — real-data amenability (intuition, not full recovery)
For each dataset, the question is *is the inverse approach even amenable?* Prefer cheap
diagnostics over long training.

### Rung 1 — yeast glycolysis (real MM kinetics, known Vmax/Km) — the fittable test
Data ready: `graphs_data/glycolysis_yeast/`. Baseline single-step rollout FALSIFIED
(test R²=−390, Pearson 0.19 → diverges).
- **H4**: an autoregressive **n_steps curriculum** (cx recipe, memory note
  `project-cx-autoregressive-curriculum`) makes the rollout converge (test R² > 0,
  Pearson rising). Implement as an **opt-in** trainer path (active only when
  `training.n_steps_schedule` is set; never touch the k_recovery single-step path).
  If `config.py`/`graph_trainer.py` edits break import, revert and log it.
- Small sweep `config/glyco_ar_*.yaml` (one axis each): schedule shape, `coeff_tail_loss`
  (0.05/0.10), `grad_clip` (0.5/1.0), `batch_size` (8/16), `noise_recurrent_level` (0/0.01).
  Metric: `-o test <cfg> best` → "test R2 (rollout)" / "test Pearson".

### Rung 3 — real E. coli metabolomics (no ground truth) — amenability probe
Data: `papers/nmeth3584/41592_2015_BFnmeth3584_MOESM197_ESM.xlsx` (sheets Ecoli1/2/3 =
247 ions × 119 timepoints; "Annotation Ecoli" = KEGG IDs). Write probes to
`scripts/ecoli_amenability.py`, save figs to `graphs_data/ecoli_realtime/`.
- **H5 (low-rank)**: real E. coli dynamics are low-rank like synthetic (SVD: rank@99%).
- **H6 (predictability)**: dc/dt is smooth/autocorrelated, not noise (lag-1 autocorr of
  each ion trace; signal-to-noise). If dc/dt is pure noise → approach NOT amenable.
- **H7 (network coverage)**: a usable fraction of the 247 ions map to an E. coli network.
  Use `papers/e_coli_core.xml` if present (else note it's missing) — count KEGG→model hits.
  Low coverage → S is too incomplete → amenability limited.

### Rung 2 — yeast-GEM (real topology, no kinetics)
Lower priority. If time: extract central-carbon S subgraph from
`papers/yeast-GEM/model/yeast-GEM.xml` (ElementTree), report size; defer dynamics.

## Figures rule (important)
Every figure added to `docs/metabolism.pdf` MUST be produced by a committed,
re-runnable script under `figures/` — `figures/<name>.py` — that **recapitulates the
analysis** behind it: load the saved data/checkpoints/logs and recompute the result.
Do NOT copy-paste the temporary plots emitted during training (`log/*/tmp_training/*`,
`log/*/results/*`). The point is to *re-derive* the figure from first artifacts:
- inputs to read: `graphs_data/<ds>/x_list_*.npy`, `gt_model.pt`, `stoichiometry.pt`;
  trained checkpoints `log/<cfg>/models/best_model_*.pt`; metrics from the test pass.
- output: `figures/metabolism/<name>.png`, referenced by `\figdir/<name>.png` in the tex.
- the script must stand alone (take config/checkpoint paths, rebuild the model, run the
  analysis). This forces real inspection of the results and the code that made them.
Example figures to build this way: CV k_R² distribution across seeds (from each
`log/k_cv_s*/` final metric); learned-vs-true log k scatter (rebuild model from a
checkpoint + gt_model); E. coli SVD/amenability (from the xlsx). Read the trainer/tester
code (`graph_trainer.py` `_plot_rate_constants_comparison`, `data_test`) to reuse the
exact recovery computation rather than guessing.

## MLP_sub alternatives (active experiment)
Diagnosed: the plain MLP_sub collapses $c^2$ to ~linear (root of the
identifiability degeneracy). Testing opt-in substrate parameterisations
(`config.graph_model.substrate_func_type`): `logspace` (g=exp(MLP(log c,s))) and
`powerlaw` (g=c^{a(s)}). Configs: `k_logspace` (running), `k_powerlaw` (queued),
then `glyco_logspace` (build from glycolysis_yeast + substrate_func_type=logspace).
Evaluate each finished run with BOTH: `figures/mlp_functions.py <cfg>` (does
MLP_sub now bend like $c^2$? plain-MLP baseline grows 8.5x vs true 81x over the
range) and `figures/k_recovery.py <cfg>` (does raw R² / the 4.3% outlier tail
improve over the plain-MLP 0.74 / 4.3%?). Promote a figure to the PDF only if a
variant clearly beats the plain MLP.

**Shared station:** Cedric's zebrafish (connectome-cx) jobs may be running too —
keep at most ~2 metabolism GPU jobs at once and prefer the idle GPU.

## Per-cycle procedure
0. **Recall first.** Read `docs/lab_notebook.md` (the running log) and skim
   `docs/metabolism.tex` to remember what has been done, what the current
   hypotheses/verdicts are, and what was queued next — before doing anything.
1. `nvidia-smi` load + `pgrep -af GNN_Main.py`.
2. Evaluate finished runs; record k R² / rollout metrics.
3. If a GPU has < 2 jobs and experiments remain, launch the next (CV seed, or AR sweep
   config, or an amenability probe) — balance cuda:0/cuda:1.
4. **Two-tier docs.**
   (a) ALWAYS append a dated entry to `docs/lab_notebook.md` — the detailed,
       raw lab notebook (what was checked/launched/found/decided, incl. dead ends,
       configs, exact metrics, file paths). This is hourly and verbose.
   (b) Promote ONLY polished, semi-publishable results to `docs/metabolism.tex`
       (the PDF): finished-experiment verdicts as a ledger row, and proper
       figures/sections when a result is solid. Keep the PDF concise and
       publication-quality — NOT a copy of the notebook. Then recompile:
       `cd docs && pdflatex -interaction=nonstopmode metabolism.tex >/dev/null 2>&1` ×2;
       `rm -f metabolism.aux metabolism.log metabolism.out`.
   Every figure in the PDF must come from a re-runnable `figures/<name>.py`.
5. `git add -u docs/metabolism.tex metabolism.pdf WEEKEND_RUN.md && git add config/k_cv_*.yaml config/glyco_ar_*.yaml scripts/ 2>/dev/null; git commit -m "weekend <date>: <result>"; git push --no-verify`.
   Never stage `graphs_data/` or `log/`.

## Guardrails
- Opt-in only: never alter the working `k_recovery_winner` single-step path.
- Broken import on a code edit → revert it, log the falsification, continue.
- One cycle per wake; one concise ledger row + one commit per cycle. Honest reporting.
