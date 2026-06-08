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
