#!/usr/bin/env python
"""
Fig 12 (complement to fig:stim_rank): free-rollout traces, one panel per model, for
the three rungs under their external drive -- Rung 1 glycolysis, Rung 3 E. coli core,
Rung 2 yeast-GEM subgraph. Ground truth (green) vs trained-model rollout (black),
z-scored and stacked; per-metabolite Pearson annotated. x-axis only (the y is a
z-scored offset). Shows the dynamics the rank/identifiability story is computed on.

Output: figures/metabolism/model_traces.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as _mpl
_mpl.rcParams["axes.spines.top"] = False; _mpl.rcParams["axes.spines.right"] = False
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data as pyg_Data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))
os.chdir(ROOT)
from k_recovery import load_model
from MetabolismGraph.utils import to_numpy

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11,
                     "legend.fontsize": 10})
GT_C, PRED_C = "#2ca02c", "k"
MODELS = [("glyco_ar_base", "Rung 1: yeast glycolysis"),
          ("ecoli_core_stim", "Rung 3: E. coli core (driven)"),
          ("yeast_central_stim", "Rung 2: yeast-GEM subgraph (driven)")]


def panel(ax, L):
    ax.text(0.0, 1.02, L, transform=ax.transAxes, fontsize=16, fontweight="bold",
            va="bottom", ha="left")


def rollout(cfg):
    config, model, gt, log_dir, ckpt = load_model(cfg, "cpu")
    ds = config.dataset; dt = config.simulation.delta_t
    xt = torch.tensor(np.load(f"graphs_data/{ds}/x_list_0.npy"), dtype=torch.float32)
    sp = f"graphs_data/{ds}/stimulus.npy"; has_stim = os.path.exists(sp)
    stim = torch.tensor(np.load(sp), dtype=torch.float32) if has_stim else None
    T = min(2000, xt.shape[0] - 1)
    ctru = to_numpy(xt[:T + 1, :, 3]); cp = np.zeros_like(ctru)
    with torch.no_grad():
        x = xt[0].clone(); cp[0] = to_numpy(x[:, 3])
        for t in range(T):
            s_ = stim[t] if has_stim else None
            x[:, 4] = xt[t, :, 4]
            pr = model(pyg_Data(x=x.clone(), pos=x[:, 1:3]), stimulus=s_)
            x[:, 3:4] = x[:, 3:4] + dt * pr.reshape(-1, 1); cp[t + 1] = to_numpy(x[:, 3])
    pm = [np.corrcoef(ctru[:, i], cp[:, i])[0, 1] for i in range(ctru.shape[1])
          if np.std(ctru[:, i]) > 1e-6 and np.std(cp[:, i]) > 1e-6 and np.isfinite(cp[:, i]).all()]
    return np.arange(T + 1) * dt, ctru, cp, float(np.mean(pm)) if pm else float("nan")


def main():
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
    SEP = 4.5
    for j, (cfg, lab) in enumerate(MODELS):
        tg, ctru, cp, pm = rollout(cfg)
        sel = np.argsort(-np.nanvar(ctru, axis=0))[:8]
        for k, i in enumerate(sel):
            mu = np.nanmean(ctru[:, i]); sd = np.nanstd(ctru[:, i]) + 1e-9; off = k * SEP
            ax[j].plot(tg, (ctru[:, i] - mu) / sd + off, color=GT_C, lw=1.2)
            ax[j].plot(tg, np.clip((cp[:, i] - mu) / sd, -0.48 * SEP, 0.48 * SEP) + off,
                       color=PRED_C, lw=0.8)
        ax[j].set_title(lab, fontsize=13, loc="center")
        ax[j].text(0.97, 0.015, f"per-met $r$={pm:.2f}", transform=ax[j].transAxes,
                   va="bottom", ha="right", fontsize=11)
        ax[j].set_xlabel("time"); ax[j].set_yticks([])
        ax[j].spines["left"].set_visible(False)
        panel(ax[j], "abc"[j])
    ax[0].plot([], [], color=GT_C, lw=1.5, label="ground truth")
    ax[0].plot([], [], color=PRED_C, lw=1.0, label="model rollout")
    ax[0].legend(loc="upper left", frameon=False)
    ax[0].set_ylabel("$z$-scored conc. (offset)")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures/metabolism/model_traces.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
