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

    # rollout PEARSON (bounded, shape-based) on a window, and divergence onset
    # (first frame where the rollout exceeds 5x the true dynamic range).
    def win_pearson(gt, cp):
        prs = [np.corrcoef(gt[:, i], cp[:, i])[0, 1] for i in range(gt.shape[1])
               if np.std(gt[:, i]) > 1e-9 and np.all(np.isfinite(cp[:, i])) and np.std(cp[:, i]) > 1e-12]
        yt, yp = gt.ravel(), cp.ravel(); ok = np.isfinite(yp)
        gP = np.corrcoef(yt[ok], yp[ok])[0, 1] if np.std(yp[ok]) > 1e-12 else float("nan")
        return (float(np.mean(prs)) if prs else float("nan"), float(gP))
    thr = 5.0 * float(np.nanmax(np.abs(c_true)))
    diverged = np.nanmax(np.abs(c_pred), axis=1) > thr
    t_div = int(np.argmax(diverged)) if diverged.any() else T + 1
    conv_pm, conv_g = win_pearson(c_true[:t_div], c_pred[:t_div])
    print(f"  rollout Pearson FULL: per-met={pear:.3f} global={g_pear:.3f}")
    print(f"  convergent window t<{t_div} (time<{t_div*dt:.0f}): "
          f"Pearson per-met={conv_pm:.3f} global={conv_g:.3f}")
    for tf in (250, 500, 1000, t_div):
        if tf <= T:
            wp, wg = win_pearson(c_true[:tf], c_pred[:tf])
            print(f"    t<{tf} (time<{tf*dt:.0f}): Pearson per-met={wp:.3f} global={wg:.3f}")
    pear, g_pear = conv_pm, conv_g   # report the convergent-window Pearson on the figure

    # ---- single panel: z-scored traces, stacked by a vertical offset ----
    # each metabolite z-scored by its GT mean/std; the rollout uses the SAME
    # transform, so a perfect rollout overlays GT and divergence departs (clipped
    # to the band so it cannot overrun neighbours). GT green, rollout black.
    sel = np.argsort(-np.nanvar(c_true, axis=0))[:8]
    SEP = 5.0
    STIM_C = "#ff7f0e"                      # stimulus = given INPUT (orange, distinct from GT/pred)
    n_met = len(sel)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    for k, i in enumerate(sel):
        gt = c_true[:, i]; pr = c_pred[:, i]
        mu, sd = np.nanmean(gt), np.nanstd(gt) + 1e-9
        off = k * SEP
        ax.plot(tgrid, (gt - mu) / sd + off, color=GT_C, lw=1.6)
        ax.plot(tgrid, np.clip((pr - mu) / sd, -0.45 * SEP, 0.45 * SEP) + off,
                color=PRED_C, lw=1.1, ls="--")
        ax.text(tgrid[0], off + 0.4 * SEP, f" {names[i]}", fontsize=10,
                color="0.4", va="center", ha="left")
    # overlay the GIVEN external stimulus (boundary drive) above the metabolites,
    # z-scored and stacked. This is a known INPUT to the inverse problem, not predicted.
    top = n_met * SEP
    if has_stim:
        stim_np = to_numpy(stim[:T + 1])                       # (T+1, n_stim)
        sidx = [j for j in range(stim_np.shape[1]) if np.nanstd(stim_np[:, j]) > 1e-9]
        for kk, j in enumerate(sidx):
            sg = stim_np[:, j]; mu, sd = np.nanmean(sg), np.nanstd(sg) + 1e-9
            off = (n_met + kk) * SEP
            ax.plot(tgrid, (sg - mu) / sd + off, color=STIM_C, lw=1.3)
            ax.text(tgrid[0], off + 0.4 * SEP, f" stim {j}", fontsize=9,
                    color=STIM_C, va="center", ha="left")
        if sidx:
            ax.plot([], [], color=STIM_C, lw=1.3, label="stimulus (given input)")
            top = (n_met + len(sidx)) * SEP
    ax.plot([], [], color=GT_C, lw=1.6, label="ground truth")
    ax.plot([], [], color=PRED_C, lw=1.1, ls="--", label="learned rollout")
    if t_div <= T:   # mark divergence onset
        ax.axvline(t_div * dt, color="#cc0000", ls=":", lw=1.2)
        ax.text(t_div * dt, top, " diverges", color="#cc0000",
                fontsize=10, va="top", ha="left")
    ax.text(0.985, 0.985, f"rollout Pearson (per-metabolite) = {pear:.2f}\n"
            f"per-metabolite $R^2$ = {r2:.2f}  (global Pearson {g_pear:.2f})",
            transform=ax.transAxes, va="top", ha="right", fontsize=11)
    ax.set_xlabel("time"); ax.set_yticks([])
    ax.set_ylabel("$z$-scored concentration  (offset per metabolite)")
    ax.set_ylim(-SEP, top + SEP)
    ax.legend(loc="lower right", frameon=False)

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
