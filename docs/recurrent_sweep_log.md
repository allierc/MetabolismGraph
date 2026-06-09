# Recurrent-training scheme sweep (D1 deep dive)

**Started 2026-06-08 ~16:55. Run autonomously for 24h. Cedric reviews at the 12h mark (~05:00).**

## Goal
Find a recurrent training scheme that gives a **TRUE good rollout** on the toy:
high **per-metabolite** rollout Pearson **AND** stable (no divergence) **AND**
preserved k-recovery. The naive curriculum failed: it stabilises by going smooth
(per-met 0.18) and collapses k (corr 0.07). Single-step baseline = per-met 0.47,
diverges at t≈124, k corr 0.87.

## Honest metric (NEVER use pooled alone)
`python figures/toy_dashboard.py <cfg>` prints, in one line:
`raw R2 / trim / %out  |  rollout Pearson per-met=<X> / pooled=<Y> | t_div=<T>`
- **per-met Pearson** = headline rollout quality (each metabolite vs its own trace).
- pooled = flattering (between-metabolite levels); report but do not trust.
- k-recovery (raw/trim/%out) = leak-resistant identifiability check.
- t_div near the full window (2001) = stable; small = diverges.

**Success bar:** per-met > 0.47 AND t_div=full AND raw k R² recovered (corr→0.87).

## Targets (baselines)
| config | scheme | per-met | pooled | k raw R² | k corr | t_div | verdict |
|---|---|---|---|---|---|---|---|
| k_recovery_winner | single-step | 0.47 | 0.74 | 0.74 | 0.87 | 1240 | recovers k, unstable |
| toy_recurrent | naive curriculum 1→1000 | 0.18 | 0.83 | 0.00 | 0.07 | 2001 | stable, dynamically dead |

## Wave 1 (launched 16:34, results pending)
| config | scheme | per-met | pooled | k raw R² | %out | t_div | verdict |
|---|---|---|---|---|---|---|---|
| toy_recurrent_anchor | S1 anchor-k (freeze k after warmup) | — | — | — | — | — | pending |
| toy_rec_cap200 | S2 horizon cap 200 | — | — | — | — | — | pending |
| toy_rec_cap500 | S3 horizon cap 500 | — | — | — | — | — | pending |
| toy_rec_lrdecay | S4 k/sub LR decay | — | — | — | — | — | pending |
| toy_rec_tail | S5 soft tail 0.5 | — | — | — | — | — | pending |
| toy_rec_warmup3 | S6 3 single-step warmups | — | — | — | — | — | pending |
| toy_rec_kitchen | S7 anchor+cap200+lrdecay | — | — | — | — | — | pending |
| toy_rec_zebra | S8 Cedric's zebrafish 800-plateau, 21 ep | — | — | — | — | — | pending |

## Backlog (launch as GPU frees; adapt toward what works)
Wave 2 — refine around wave-1 winners:
- vary plateau HEIGHT (cap 100 / 300 / 400 / 600) at fixed length — find the sweet spot
  between "too short → still diverges" and "too long → goes smooth".
- vary plateau LENGTH at the best height (few vs many epochs).
- anchor-k combined with each promising horizon/LR.
- LR floor variations (5e-5 vs 1e-5 vs 2e-4) on the zebra-style schedule.
Wave 3 — structurally different stabilisers (bigger trainer change, only if wave1/2 stall):
- scheduled sampling / pushforward: mix teacher-forced and free steps within a rollout.
- noise injection during rollout (denoising-style robustness).
- per-step loss weighting (emphasise early steps) vs trajectory-mean.
- two-phase: recover k single-step → FREEZE k → train ONLY stability params on long rollout.

## Cost model
per-epoch wall ≈ horizon T seconds (144 iters). total ≈ Σ(horizons). With 2 jobs/GPU
add ~1.5–2×. Capped schemes ~20–40 min; full-1000 ~2–3 h; zebra 800-plateau ~4–6 h.

## Results & decisions (appended each cycle)
<!-- cron appends dated rows + the running BEST per-met scheme here -->

## RESULTS — Wave 1 complete (2026-06-09 ~09:00, evaluated manually; cron was offline overnight)

Baselines: **single-step per-met=0.47, kR²=0.74, diverges@t1240** | **log-space single-step per-met=0.32, kR²=0.89, STABLE**

| scheme | per-met | pooled | k raw R² | %out | t_div | verdict |
|---|---|---|---|---|---|---|
| single-step (baseline) | **0.47** | 0.74 | **0.74** | 4 | 1240 | best per-met + k, but diverges |
| log-space single-step | 0.32 | 0.79 | **0.89** | 2 | stable | best STABLE option, best k |
| S-naive recurrent | 0.18 | 0.83 | 0.00 | 90 | stable | degenerate |
| S1 anchor-k | 0.09 | 0.03 | 0.00 | 99 | 1476 | FAILED |
| S2 cap-200 | 0.23 | 0.18 | 0.07 | 84 | 645 | best of recurrent, still ≪ baseline |
| S3 cap-500 | 0.21 | 0.62 | 0.04 | 84 | stable | degenerate |
| S4 lr-decay | 0.21 | 0.71 | 0.00 | 80 | stable | degenerate |
| S5 tail-loss | 0.22 | 0.78 | 0.00 | 76 | stable | degenerate |
| S6 warmup-3 | 0.18 | 0.45 | 0.02 | 73 | stable | degenerate |
| S7 kitchen-sink | 0.10 | 0.02 | 0.00 | 100 | stable | FAILED |
| S8 zebrafish 800-plateau | 0.11 | 0.05 | 0.00 | 100 | stable | FAILED |

**CONCLUSION: no recurrent scheme works on this toy.** All 8 (incl. the gentle zebrafish recipe)
collapse to the smooth-degenerate regime: per-met 0.09–0.23 (≪ single-step 0.47) AND k-recovery
destroyed (raw R² 0.00–0.07, 73–100% outliers). Trend: the schemes that stay bounded (t_div=2001)
have the WORST k — "stability = degeneracy" here. The least-curriculum scheme (cap-200, which still
diverges) is the least-bad, i.e. closer to single-step is better.

**KEY HYPOTHESIS (why it fails here but works for zebrafish/CX): the toy has NO external stimulus.**
The toy is an autonomous LINEAR system (all |s|=1 → dc/dt=Mc); a long autonomous rollout is dominated
by leading eigenvalues, so the multi-step loss finds a damped (stable) solution that kills the
oscillations and frees k to drift. In zebrafish/CX the curriculum works because a strong external
drive PINS the dynamics every step. => The recurrent curriculum likely needs the D3 stimulus to anchor
it. NEXT: combine D1+D3 (recurrent training WITH the external drive), not more stimulus-free variants.

**Best STABLE model so far = log-space single-step** (per-met 0.32, k 0.89, no divergence) — NOT a
recurrent scheme. The "true good rollout" (per-met>0.47 AND stable AND k preserved) was NOT achieved.

## PIVOT (2026-06-09): recurrent + external stimulus + 10 runs (D1+D3)
Levers that were inconclusive/negative (see lab_notebook 2026-06-09 summary): (1) recurrent
curriculum alone — 8 schemes all degenerate; (2) n_runs 1/3/10 — noisy + budget-confounded;
(3) log-exp for c^2 — no |s|=2 signal. Pivot: recurrent WITH stimulus (the missing anchor) +
10 runs/test. Tests: toy_stim10_{single,naive,zebra,cap200} on toy_stim10_data (held-out run 10).
