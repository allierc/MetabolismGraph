#!/usr/bin/env python
"""
Rung-1 glycolysis: free-running rollout, ground truth vs learned.

Recapitulates figures/metabolism/glyco_rollout.png. Loads a trained AR-curriculum
checkpoint, autoregressively rolls the metabolite concentrations forward from
frame 0 under the GIVEN boundary stimulus (predicting c(t+1)=c(t)+dt*f(c(t))),
and compares the free-running trajectory to the ground-truth simulation.
Shows that the curriculum recovers trajectory SHAPE (high Pearson) while the
amplitude/scale drifts (rollout R^2 < 0).

Inputs:  graphs_data/glycolysis_yeast/{x_list_0.npy, stimulus.npy, stoich_graph.pt,
         metadata.pt} ; log/<config>/models/best_model_*.pt
Output:  figures/metabolism/glyco_rollout.png
Usage:   python figures/glyco_rollout.py [config_name]   (default glyco_ar_base)
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data as pyg_Data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
from MetabolismGraph.config import MetabolismGraphConfig
from MetabolismGraph.models.Metabolism_Propagation import Metabolism_Propagation
from MetabolismGraph.utils import to_numpy

plt.rcParams.update({"font.size": 14, "axes.labelsize": 16,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12})
GT_C, PRED_C = "#2ca02c", "k"          # color convention: GT=green, predicted=black
T_ROLL = 2000


def panel_label(ax, letter):
    ax.text(0.015, 0.985, letter, transform=ax.transAxes, fontsize=20,
            fontweight="bold", va="top", ha="left")


def main():
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "glyco_ar_base"
    device = "cpu"
    config = MetabolismGraphConfig.from_yaml(f"config/{cfg_name}.yaml")
    config.config_file = cfg_name
    ds = config.dataset
    dt = config.simulation.delta_t

    stoich_graph = torch.load(f"graphs_data/{ds}/stoich_graph.pt", map_location=device)
    meta_path = f"graphs_data/{ds}/metadata.pt"
    if os.path.exists(meta_path):
        names = torch.load(meta_path, map_location=device, weights_only=False).get(
            "species_names", [str(i) for i in range(config.simulation.n_metabolites)])
    else:
        names = [str(i) for i in range(config.simulation.n_metabolites)]

    x_true = torch.tensor(np.load(f"graphs_data/{ds}/x_list_0.npy"), dtype=torch.float32)
    stim_path = f"graphs_data/{ds}/stimulus.npy"
    has_stim = os.path.exists(stim_path)   # toy regimes have no external stimulus
    stim = torch.tensor(np.load(stim_path), dtype=torch.float32) if has_stim else None
    T = min(T_ROLL, x_true.shape[0] - 1)
    c_true = to_numpy(x_true[:T + 1, :, 3])                # (T+1, N)

    cks = sorted(glob.glob(f"log/{cfg_name}/models/best_model_with_*graphs_*.pt"),
                 key=os.path.getmtime)
    model = Metabolism_Propagation(config=config, device=device)
    model.load_stoich_graph(stoich_graph)
    model.load_state_dict(torch.load(cks[-1], map_location=device)["model_state_dict"])
    model.eval()

    def rollout(use_stim):
        """true free rollout from frame 0: only c=x[:,3] evolves; aux cols static."""
        cp = np.zeros_like(c_true)
        with torch.no_grad():
            x = x_true[0].clone()
            cp[0] = to_numpy(x[:, 3])
            for t in range(T):
                s = stim[t] if use_stim else None
                pred = model(pyg_Data(x=x.clone(), pos=x[:, 1:3]), stimulus=s)
                x[:, 3:4] = x[:, 3:4] + dt * pred.reshape(-1, 1)
                cp[t + 1] = to_numpy(x[:, 3])
        return cp

    def metrics(cp):
        """per-metabolite mean R2/Pearson (official style) + global."""
        r2s, prs = [], []
        for i in range(c_true.shape[1]):
            gt, pr = c_true[:, i], cp[:, i]
            if not np.all(np.isfinite(pr)) or np.std(gt) < 1e-12 or np.std(pr) < 1e-12:
                continue
            r2s.append(1 - np.sum((gt - pr) ** 2) / np.sum((gt - gt.mean()) ** 2))
            prs.append(np.corrcoef(gt, pr)[0, 1])
        yt, yp = c_true.ravel(), cp.ravel(); ok = np.isfinite(yp)
        g_r2 = 1 - np.sum((yt[ok] - yp[ok]) ** 2) / np.sum((yt[ok] - yt[ok].mean()) ** 2)
        return (np.mean(r2s), np.mean(prs), g_r2, np.corrcoef(yt[ok], yp[ok])[0, 1])

    c_pred = rollout(use_stim=has_stim)
    m_stim = metrics(c_pred)
    print(f"  rollout: per-met R2={m_stim[0]:.3f} Pearson={m_stim[1]:.3f} | "
          f"global R2={m_stim[2]:.3f} Pearson={m_stim[3]:.3f}")
    if has_stim:   # glyco: also show the no-stimulus control (= the buggy data_test)
        m_nostim = metrics(rollout(use_stim=False))
        print(f"  no-stimulus control: per-met R2={m_nostim[0]:.3f} Pearson={m_nostim[1]:.3f} "
              f"(matches buggy data_test)")
    r2, pear = m_stim[0], m_stim[1]   # per-metabolite metric (the honest one)
    g_r2, g_pear = m_stim[2], m_stim[3]
    tgrid = np.arange(T + 1) * dt

    # ---- figure: (a) example traces, (b) predicted-vs-true scatter ----
    sel = np.argsort(-np.nanvar(c_true, axis=0))[:4]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for i in sel:
        ax[0].plot(tgrid, c_true[:, i], color=GT_C, lw=2, alpha=.9)
        ax[0].plot(tgrid, c_pred[:, i], color=PRED_C, lw=1.3, ls="--")
        ax[0].annotate(str(names[i]), (tgrid[-1], c_true[-1, i]), fontsize=11,
                       color="0.3", va="center")
    ax[0].plot([], [], color=GT_C, lw=2, label="ground truth")
    ax[0].plot([], [], color=PRED_C, lw=1.3, ls="--", label="learned rollout")
    ax[0].set_xlabel("time"); ax[0].set_ylabel("concentration")
    ax[0].set_ylim(0, 1.5 * float(np.nanmax(c_true)))   # clip if the rollout diverges
    ax[0].legend(loc="lower right", frameon=False); panel_label(ax[0], "a")

    yt, yp = c_true.ravel(), c_pred.ravel(); ok = np.isfinite(yp)
    ax[1].scatter(yt[ok][::20], yp[ok][::20], s=4, c="#1f77b4", alpha=.2, edgecolors="none")
    lim_lo = float(np.nanpercentile(yt, 1)); lim_hi = float(np.nanpercentile(yt, 99))
    ax[1].plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", c="gray", lw=1)
    ax[1].set_xlim(lim_lo, lim_hi); ax[1].set_ylim(lim_lo, lim_hi)
    ax[1].set_xlabel("true concentration"); ax[1].set_ylabel("learned (rollout)")
    fmt = lambda r: r"$\ll 0$" if r < -100 else f"{r:.2f}"
    ax[1].text(0.97, 0.06,
               f"per-metabolite $R^2$ = {fmt(r2)}\nglobal $R^2$ = {fmt(g_r2)}\n"
               f"per-met Pearson = {pear:.2f}",
               transform=ax[1].transAxes, va="bottom", ha="right", fontsize=13)
    panel_label(ax[1], "b")

    fname = ("glyco_rollout.png" if "glyco" in cfg_name
             else "toy_rollout.png" if cfg_name == "k_recovery_winner"
             else f"{cfg_name}_rollout.png")
    out = os.path.join(ROOT, "figures/metabolism", fname)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}  ({cfg_name}, T={T})")
    print(f"  rollout Pearson={pear:.3f}  R2={r2:.3f}")
    return dict(config=cfg_name, pearson=float(pear), r2=float(r2), T=T)


if __name__ == "__main__":
    main()
