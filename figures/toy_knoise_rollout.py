#!/usr/bin/env python
"""
Rate-noise sweep, ROLLOUT traces (the stoichiometrically-sound noise: k_j(1+sigma*xi),
mass-conserving). For each noise level we show the noisy simulated ground truth (green)
and the trained model's deterministic free rollout (black), z-scored and stacked, with
the leak-resistant k-recovery R^2 annotated. Question: does rate noise do anything to
the dynamics / recovery? (Answer: benign at low sigma, corrupts the trajectory at high.)

Output: figures/metabolism/toy_knoise_rollout.png
Usage:  python figures/toy_knoise_rollout.py
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
from k_recovery import load_model, OUTLIER_THRESHOLD
from MetabolismGraph.models.graph_trainer import _plot_rate_constants_comparison
from MetabolismGraph.utils import to_numpy

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10})
GT_C, PRED_C = "#2ca02c", "k"          # ground truth green, rollout black
COLS = [("k_recovery_winner", 0.0), ("toy_knoise_010", 0.1),
        ("toy_knoise_030", 0.3), ("toy_knoise_050", 0.5)]


def panel(ax, L):
    ax.text(0.0, 1.02, L, transform=ax.transAxes, fontsize=17, fontweight="bold",
            va="bottom", ha="left")


def rollout(model, config, ds):
    dt = config.simulation.delta_t
    xt = torch.tensor(np.load(f"graphs_data/{ds}/x_list_0.npy"), dtype=torch.float32)
    T = min(2000, xt.shape[0] - 1)
    ctru = to_numpy(xt[:T + 1, :, 3]); cp = np.zeros_like(ctru)
    with torch.no_grad():
        x = xt[0].clone(); cp[0] = to_numpy(x[:, 3])
        for t in range(T):
            x[:, 4] = xt[t, :, 4]
            pr = model(pyg_Data(x=x.clone(), pos=x[:, 1:3]), stimulus=None)
            x[:, 3:4] = x[:, 3:4] + dt * pr.reshape(-1, 1); cp[t + 1] = to_numpy(x[:, 3])
    return np.arange(T + 1) * dt, ctru, cp


def main():
    fig, ax = plt.subplots(1, 4, figsize=(17, 6), sharey=True)
    SEP = 4.5
    # fix the displayed metabolites from the clean run so columns are comparable
    clean = np.load("graphs_data/k_recovery_winner/x_list_0.npy")[:, :, 3]
    sel = np.argsort(-np.nanvar(clean, axis=0))[:9]
    for j, (cfg, sigma) in enumerate(COLS):
        config, model, gt, log_dir, _ = load_model(cfg, "cpu")
        raw, trim, nout, _ = _plot_rate_constants_comparison(
            model, gt, log_dir, 0, config.simulation.n_metabolites,
            device="cpu", outlier_threshold=OUTLIER_THRESHOLD)
        nrx = len(to_numpy(gt.log_k.detach().cpu()).ravel())
        tg, ctru, cp = rollout(model, config, config.dataset)
        Tn = min(len(tg), ctru.shape[0], cp.shape[0])
        pm = [np.corrcoef(ctru[:Tn, i], cp[:Tn, i])[0, 1] for i in range(ctru.shape[1])
              if np.std(ctru[:Tn, i]) > 1e-6 and np.std(cp[:Tn, i]) > 1e-6 and np.isfinite(cp[:Tn, i]).all()]
        pm = float(np.mean(pm)) if pm else float("nan")
        for k, i in enumerate(sel):
            mu = np.nanmean(clean[:, i]); sd = np.nanstd(clean[:, i]) + 1e-9; off = k * SEP
            ax[j].plot(tg[:Tn], (ctru[:Tn, i] - mu) / sd + off, color=GT_C, lw=1.0, alpha=0.9)
            ax[j].plot(tg[:Tn], np.clip((cp[:Tn, i] - mu) / sd, -0.48 * SEP, 0.48 * SEP) + off,
                       color=PRED_C, lw=0.7)
        ax[j].set_title(rf"$\sigma_k={sigma}$", fontsize=14, loc="center")
        ax[j].text(0.97, 0.015, f"$k$ $R^2$={raw:.2f}, {100*nout/nrx:.0f}\\% out\nper-met $r$={pm:.2f}",
                   transform=ax[j].transAxes, va="bottom", ha="right", fontsize=10)
        ax[j].set_xlabel("time (frames)"); ax[j].set_yticks([])
        panel(ax[j], "abcd"[j])
    ax[0].plot([], [], color=GT_C, lw=1.6, label="noisy simulation (GT)")
    ax[0].plot([], [], color=PRED_C, lw=1.0, label="model rollout")
    ax[0].legend(loc="upper left", frameon=False, fontsize=9)
    ax[0].set_ylabel("$z$-scored conc. (offset)")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures/metabolism/toy_knoise_rollout.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
