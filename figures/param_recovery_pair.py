#!/usr/bin/env python
"""
Companion parameter-recovery scatters: per-reaction recovered vs true log10 Vmax,
on the three realistic-OU-stimulus rungs (glycolysis, E. coli core, yeast central).
Two figures with identical layout:
  param_recovery_lsq.png  -- least squares given the true substrate shape
  param_recovery_gnn.png  -- the trained GNN (leak-resistant recovery)
Each is a 1x3 scatter (one panel per rung) with the identity line and R^2.

Output: figures/metabolism/param_recovery_{lsq,gnn}.png
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
from MetabolismGraph.config import MetabolismGraphConfig
from MetabolismGraph.generators.PDE_MichaelisMenten import PDE_MichaelisMenten
from MetabolismGraph.models.graph_trainer import _compute_scalar_correction
from MetabolismGraph.utils import to_numpy

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11})
INK, OUT_C = "k", "#e74c3c"          # dots black; outliers red (as in the k-recovery figures)
OUTLIER_DEX = 0.3                     # |recovered - true| outlier gate (dex), as in Fig. 3
RUNGS = [("glyco_topo_ou", "Rung 1: glycolysis"),
         ("ecoli_core_ou", "Rung 2: $E.\\ coli$ core"),
         ("yeast_central_ou", "Rung 3: yeast central")]


def linfit_r2(true, pred):
    A = np.vstack([true, np.ones_like(true)]).T
    coef, *_ = np.linalg.lstsq(A, pred, rcond=None)
    res = pred - A @ coef
    ss_tot = np.sum((pred - pred.mean()) ** 2)
    return float(1 - np.sum(res ** 2) / ss_tot) if ss_tot > 0 else 0.0


def lsq_vmax(config, ds):
    sg = torch.load(f"graphs_data/{ds}/stoich_graph.pt", map_location="cpu")
    torch.manual_seed(0); m = PDE_MichaelisMenten(config=config, stoich_graph=sg, device="cpu")
    m.load_state_dict(torch.load(f"graphs_data/{ds}/gt_model.pt", map_location="cpu"), strict=False); m.eval()
    x = np.load(f"graphs_data/{ds}/x_list_0.npy"); dt = config.simulation.delta_t
    nrxn = config.simulation.n_reactions; nmet = config.simulation.n_metabolites
    lk0 = m.log_k.detach().clone(); Vgt = to_numpy(10 ** lk0); eps = 1e-3
    f = lambda xt: to_numpy(m(pyg_Data(x=torch.tensor(xt, dtype=torch.float32),
                            pos=torch.tensor(xt[:, 1:3], dtype=torch.float32)), dt=dt).reshape(-1))
    Ts = np.linspace(0, x.shape[0] - 2, 120, dtype=int); Ab, bb = [], []
    for t in Ts:
        xt = x[t].copy(); d0 = f(xt)
        with torch.no_grad(): m.log_k.copy_(torch.full_like(lk0, -30.)); g = f(xt); m.log_k.copy_(lk0)
        bb.append(d0 - g); At = np.zeros((nmet, nrxn))
        for j in range(nrxn):
            with torch.no_grad(): m.log_k[j] += eps; dj = f(xt); m.log_k.copy_(lk0)
            At[:, j] = (dj - d0) / (Vgt[j] * (10 ** eps - 1))
        Ab.append(At)
    Vhat, *_ = np.linalg.lstsq(np.vstack(Ab), np.concatenate(bb), rcond=None)
    return to_numpy(lk0), Vhat


def gnn_recovery(cfg):
    config, model, gt, log_dir, ckpt = load_model(cfg, "cpu")
    true = to_numpy(gt.log_k.detach().cpu()).ravel()
    learned = to_numpy(model.log_k.detach().cpu()).ravel()
    log_alpha, n_sub = _compute_scalar_correction(model, "cpu")
    return true, learned + n_sub * log_alpha


def scatter_lsq(ax, true, pred, title):
    """LSQ panel: black inliers, red outliers; report raw/trimmed R^2 + outliers (as Fig. 3)."""
    m = np.isfinite(pred) & np.isfinite(true)
    t, p = true[m], pred[m]
    out = np.abs(p - t) > OUTLIER_DEX            # identity-based outlier gate (0.3 dex)
    raw = linfit_r2(t, p)
    trimmed = linfit_r2(t[~out], p[~out]) if (~out).sum() > 2 else raw
    n, N = int(out.sum()), len(t)
    lo = min(t.min(), p.min()) - 0.25; hi = max(t.max(), p.max()) + 0.25
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 4)), color="#bbbbbb", lw=1.0, zorder=1)
    ax.scatter(t[~out], p[~out], s=26, c=INK, alpha=0.8, edgecolors="none", zorder=3)
    if n:
        ax.scatter(t[out], p[out], s=26, c=OUT_C, alpha=0.85, edgecolors="none", zorder=4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    txt = (f"raw $R^2$ = {raw:.2f}\ntrimmed $R^2$ = {trimmed:.2f}\n"
           f"outliers = {n}/{N} = {100*n/N:.0f}%")
    ax.text(0.05, 0.96, txt, transform=ax.transAxes, va="top", ha="left", fontsize=11)
    ax.set_title(title, fontsize=13)
    print(f"  lsq {title}: raw={raw:.3f} trimmed={trimmed:.3f} outliers={n}/{N}")


def scatter_gnn(ax, true, pred, title):
    """GNN panel: all black, report R^2 only (no outlier hunt -- R^2 is truly off)."""
    m = np.isfinite(pred) & np.isfinite(true)
    t, p = true[m], pred[m]
    r2 = linfit_r2(t, p)
    lo = min(t.min(), p.min()) - 0.25; hi = max(t.max(), p.max()) + 0.25
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 4)), color="#bbbbbb", lw=1.0, zorder=1)
    ax.scatter(t, p, s=26, c=INK, alpha=0.8, edgecolors="none", zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.text(0.05, 0.95, f"$R^2$ = {r2:.2f}", transform=ax.transAxes, va="top", ha="left",
            fontsize=12)
    ax.set_title(title, fontsize=13)
    print(f"  gnn {title}: R2={r2:.3f}")


def make_figure(method):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.8))
    for j, (cfg, lab) in enumerate(RUNGS):
        config = MetabolismGraphConfig.from_yaml(f"config/{cfg}.yaml")
        if method == "lsq":
            true, Vhat = lsq_vmax(config, config.dataset)
            keep = Vhat > 0
            pred = np.full_like(true, np.nan); pred[keep] = np.log10(Vhat[keep])
            scatter_lsq(ax[j], true, pred, lab)
        else:
            true, pred = gnn_recovery(cfg)
            scatter_gnn(ax[j], true, pred, lab)
        ax[j].set_xlabel("true $\\log_{10} V_{\\max}$")
        if j == 0:
            ax[j].set_ylabel("recovered $\\log_{10} V_{\\max}$")
    fig.tight_layout()
    out = os.path.join(ROOT, f"figures/metabolism/param_recovery_{method}.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


def main():
    make_figure("lsq")
    make_figure("gnn")


if __name__ == "__main__":
    main()
