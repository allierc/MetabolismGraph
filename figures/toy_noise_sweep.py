#!/usr/bin/env python
"""
Companion to the toy dashboard (Fig 1): INTRINSIC-NOISE sweep.

Tests the flyvis finding (Lappalainen/Allier) in the metabolism toy: intrinsic
process noise injected during data generation BREAKS the identifiability
degeneracy of the autonomous toy. We sweep simulation.noise_model_level and,
for each level, train single-step (leak-resistant k-recovery, S given) on the
NOISY trajectory, then evaluate against BOTH ground truths:
  - the noisy GT (what the model trained on) -- rollout can't match the noise,
  - the NOISE-FREE GT (toy_noise_000, same network/seed) -- the deterministic
    model rollout SHOULD reproduce it (it cannot fit the stochastic fluctuations
    and converges on the underlying deterministic dynamics).

Panels:
  (a) activity rank(99%) vs noise          -- degeneracy breaking, data side
  (b) k-recovery R^2 + %outliers vs noise   -- identifiability, the headline
  (c) rollout per-met Pearson vs noise, vs noise-free GT (green) and noisy GT (gray)
  (d) example rollout traces at one level: noisy GT (gray) / noise-free GT (green)
      / model rollout (black) -- the model recovers clean dynamics from noisy data

Output: figures/metabolism/toy_noise_sweep.png
Usage:  python figures/toy_noise_sweep.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data as pyg_Data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))
os.chdir(ROOT)
from k_recovery import load_model, OUTLIER_THRESHOLD, HARD_RULE_PCT
from MetabolismGraph.models.graph_trainer import _plot_rate_constants_comparison
from MetabolismGraph.utils import to_numpy

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10})
GT_C, PRED_C, NOISY_C = "#2ca02c", "k", "#999999"   # clean green, rollout black, noisy gray

# (tag, sigma). 000 is the noise-free twin used as the clean rollout reference.
LEVELS = [("000", 0.0), ("001", 0.01), ("002", 0.02),
          ("003", 0.03), ("005", 0.05), ("007", 0.07)]
CLEAN_DS = "toy_noise_000"
TRACE_TAG = "005"     # noise level whose example traces go in panel (d)


def panel(ax, L):
    ax.text(0.0, 1.02, L, transform=ax.transAxes, fontsize=18,
            fontweight="bold", va="bottom", ha="left")


def rank99(ds):
    """activity rank at 99% cumulative singular-value energy (same as the generator)."""
    x = np.load(f"graphs_data/{ds}/x_list_0.npy")[:, :, 3]   # frames x metabolites
    x = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(e, 0.99) + 1)


def k_metrics(model, gt_model, log_dir, N, device):
    # authoritative computation (identical to figures/k_recovery.py / training)
    raw, trimmed, n_out, slope = _plot_rate_constants_comparison(
        model, gt_model, log_dir, 0, N, device=device, outlier_threshold=OUTLIER_THRESHOLD)
    n_rxn = len(to_numpy(gt_model.log_k.detach().cpu()).ravel())
    return raw, trimmed, 100.0 * n_out / n_rxn


def rollout(model, config, ref_ds, device):
    """Deterministic free rollout from ref_ds frame 0; returns (time, ctru, cp)."""
    dt = config.simulation.delta_t
    xt = torch.tensor(np.load(f"graphs_data/{ref_ds}/x_list_0.npy"), dtype=torch.float32)
    T = min(2000, xt.shape[0] - 1)
    ctru = to_numpy(xt[:T + 1, :, 3])
    cp = np.zeros_like(ctru)
    with torch.no_grad():
        x = xt[0].clone(); cp[0] = to_numpy(x[:, 3])
        for t in range(T):
            x[:, 4] = xt[t, :, 4]
            pr = model(pyg_Data(x=x.clone(), pos=x[:, 1:3]), stimulus=None)
            x[:, 3:4] = x[:, 3:4] + dt * pr.reshape(-1, 1)
            cp[t + 1] = to_numpy(x[:, 3])
    return np.arange(T + 1) * dt, ctru, cp


def per_met_pearson(gt, pr):
    v = [np.corrcoef(gt[:, i], pr[:, i])[0, 1] for i in range(gt.shape[1])
         if np.std(gt[:, i]) > 1e-6 and np.std(pr[:, i]) > 1e-6 and np.isfinite(pr[:, i]).all()]
    return float(np.mean(v)) if v else float("nan")


def main():
    dev = "cpu"
    sig, rk, kraw, ktrim, kout, pm_clean, pm_noisy = [], [], [], [], [], [], []
    trace = None
    clean_traj = np.load(f"graphs_data/{CLEAN_DS}/x_list_0.npy")[:, :, 3]

    for tag, s in LEVELS:
        cfg = f"toy_noise_{tag}"
        try:
            config, model, gt_model, log_dir, _ = load_model(cfg, dev)
        except Exception as e:
            print(f"skip {cfg}: {e}"); continue
        raw, trim, pct = k_metrics(model, gt_model, log_dir, config.simulation.n_metabolites, dev)
        # one deterministic rollout from the shared frame-0 state (clean reference)
        tg, ctru_clean, cp = rollout(model, config, CLEAN_DS, dev)
        T = cp.shape[0]
        noisy_traj = np.load(f"graphs_data/{cfg}/x_list_0.npy")[:T, :, 3]
        sig.append(s); rk.append(rank99(cfg))
        kraw.append(raw); ktrim.append(trim); kout.append(pct)
        pm_clean.append(per_met_pearson(ctru_clean, cp))
        pm_noisy.append(per_met_pearson(noisy_traj, cp))
        if tag == TRACE_TAG:
            trace = (tg, ctru_clean, noisy_traj, cp)
        print(f"sigma={s}: rank={rk[-1]} kraw={raw:.3f} ktrim={trim:.3f} out={pct:.0f}% "
              f"pm_clean={pm_clean[-1]:.3f} pm_noisy={pm_noisy[-1]:.3f}")

    sig = np.array(sig)
    sig_trace = dict(LEVELS)[TRACE_TAG]
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (a) k-recovery vs noise (raw + trimmed R^2; outliers on a twin axis) -- headline
    ax[0, 0].plot(sig, kraw, "o-", color="k", lw=2, label=r"raw $R^2$")
    ax[0, 0].plot(sig, ktrim, "s--", color="#1f77b4", lw=1.6, label=r"trimmed $R^2$")
    ax[0, 0].set_xlabel(r"intrinsic noise $\sigma$"); ax[0, 0].set_ylabel(r"$k$-recovery $R^2$")
    ax[0, 0].legend(loc="lower right", frameon=False)
    axb = ax[0, 0].twinx()
    axb.plot(sig, kout, "^:", color="#d62728", lw=1.4)
    axb.axhline(HARD_RULE_PCT, color="#d62728", ls=":", lw=1, alpha=0.5)
    axb.set_ylabel("% outliers", color="#d62728"); axb.tick_params(axis="y", colors="#d62728")
    panel(ax[0, 0], "a")

    # (b) rollout per-met Pearson vs noise: vs noise-free GT (green) and noisy GT (gray)
    ax[0, 1].plot(sig, pm_clean, "o-", color=GT_C, lw=2, label="vs noise-free GT")
    ax[0, 1].plot(sig, pm_noisy, "o--", color=NOISY_C, lw=1.8, label="vs noisy GT")
    ax[0, 1].set_xlabel(r"intrinsic noise $\sigma$")
    ax[0, 1].set_ylabel("rollout per-met Pearson")
    ax[0, 1].legend(loc="lower left", frameon=False)
    panel(ax[0, 1], "b")

    # shared metabolite selection + offset for the two trace panels (top-variance, clean GT)
    SEP = 5
    if trace is not None:
        tg, ctru_clean, noisy_traj, cp = trace
        sel = np.argsort(-np.nanvar(ctru_clean, axis=0))[:6]
        Tn = noisy_traj.shape[0]

        # (c) the NOISY training data the model was fit on (jagged) -- gray
        for k, i in enumerate(sel):
            g, gn = ctru_clean[:, i], noisy_traj[:, i]
            mu, sd = np.nanmean(g), np.nanstd(g) + 1e-9; off = k * SEP
            ax[1, 0].plot(tg[:Tn], (gn - mu) / sd + off, color=NOISY_C, lw=0.7)
        ax[1, 0].plot([], [], color=NOISY_C, lw=1.0, label="noisy GT (training data)")
        ax[1, 0].legend(loc="lower right", frameon=False)
        ax[1, 0].set_xlabel("time"); ax[1, 0].set_yticks([])
        ax[1, 0].set_ylabel("$z$-scored conc. (offset)")
        ax[1, 0].text(0.97, 0.02, rf"$\sigma={sig_trace}$ training data",
                      transform=ax[1, 0].transAxes, va="bottom", ha="right", fontsize=12)

        # (d) model rollout (black) vs the NOISE-FREE GT (green) -- recovers clean dynamics
        for k, i in enumerate(sel):
            g, p = ctru_clean[:, i], cp[:, i]
            mu, sd = np.nanmean(g), np.nanstd(g) + 1e-9; off = k * SEP
            ax[1, 1].plot(tg, (g - mu) / sd + off, color=GT_C, lw=1.4)
            ax[1, 1].plot(tg, np.clip((p - mu) / sd, -0.45 * SEP, 0.45 * SEP) + off,
                          color=PRED_C, lw=1, ls="--")
        ax[1, 1].plot([], [], color=GT_C, lw=1.4, label="noise-free GT")
        ax[1, 1].plot([], [], color=PRED_C, lw=1, ls="--", label="model rollout")
        ax[1, 1].legend(loc="lower right", frameon=False)
        ax[1, 1].set_xlabel("time"); ax[1, 1].set_yticks([])
        ax[1, 1].set_ylabel("$z$-scored conc. (offset)")
        pm_c = pm_clean[[t for t, _ in LEVELS].index(TRACE_TAG)]
        ax[1, 1].text(0.97, 0.02, rf"$\sigma={sig_trace}$:  per-met $r={pm_c:.2f}$ vs noise-free GT",
                      transform=ax[1, 1].transAxes, va="bottom", ha="right", fontsize=11)
    panel(ax[1, 0], "c")
    panel(ax[1, 1], "d")

    fig.tight_layout()
    out = os.path.join(ROOT, "figures/metabolism/toy_noise_sweep.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
