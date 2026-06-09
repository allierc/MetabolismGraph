#!/usr/bin/env python
"""
Toy-model results DASHBOARD — one figure summarising a fit:
  (a) MLP_sub vs the true substrate law c^|s|
  (b) MLP_node vs the true homeostasis
  (c) rate-constant recovery: learned vs true log10 k (outliers red, raw/trimmed/%)
  (d) free rollout: z-scored stacked traces, ground truth (green) vs rollout (black),
      with the divergence marker and convergent-window rollout Pearson.
Config-parameterised so every new result is shown as the same dashboard.

Output: figures/metabolism/toy_dashboard[_<cfg>].png
Usage:  python figures/toy_dashboard.py [config_name]   (default k_recovery_winner)
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
from k_recovery import load_model
from MetabolismGraph.models.graph_trainer import (
    _plot_rate_constants_comparison, _compute_scalar_correction)
from MetabolismGraph.utils import to_numpy

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10})
GT_C, PRED_C = "#2ca02c", "k"     # ground truth green, predicted black


def panel(ax, L):
    # bold letter OUTSIDE the box, top-left, aligned across panels
    ax.text(0.0, 1.02, L, transform=ax.transAxes, fontsize=18,
            fontweight="bold", va="bottom", ha="left")


def main():
    cfg = sys.argv[1] if len(sys.argv) > 1 else "k_recovery_winner"
    dev = "cpu"
    config, model, gt, log_dir, ckpt = load_model(cfg, dev)
    N = config.simulation.n_metabolites
    cmax = float(config.simulation.concentration_max)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # ---- (a) MLP_sub vs c^s ----
    c = torch.linspace(0.05, cmax, 200); cn = to_numpy(c)
    with torch.no_grad():
        for s, col in [(1, "#1f77b4"), (2, "#d62728")]:
            sv = torch.full((200, 1), float(s))
            o = model.substrate_func(torch.cat([c.unsqueeze(-1), sv], -1))
            f = to_numpy(o.norm(dim=-1) if o.ndim > 1 else o.abs()); f = f / np.interp(1.0, cn, f)
            ax[0, 0].plot(cn, f, color=col, lw=2.2, label=f"learned $|s|={s}$")
            ax[0, 0].plot(cn, cn ** s, color=col, lw=1.4, ls="--", label=f"$c^{s}$ (true)")
    ax[0, 0].set_xlabel("concentration $c$")
    ax[0, 0].set_ylabel(r"$\mathrm{MLP_{sub}}$  (anchored $c{=}1$)")
    ax[0, 0].legend(loc="upper left", frameon=False, bbox_to_anchor=(0.06, 0.96))
    panel(ax[0, 0], "a")

    # ---- (b) MLP_node (homeostasis) ----
    with torch.no_grad():
        a = model.a if hasattr(model, "a") else None
        for i in range(0, N, max(1, N // 12)):
            ai = a[i:i + 1].repeat(len(c), 1) if a is not None else torch.zeros(len(c), 0)
            h = model.node_func(torch.cat([c.unsqueeze(-1), ai], -1)).squeeze(-1)
            ax[0, 1].plot(cn, to_numpy(h), color=PRED_C, lw=1, alpha=.5)
    ax[0, 1].axhline(0, color=GT_C, lw=1.6, ls="--", label="true homeostasis")
    ax[0, 1].set_xlabel("concentration $c$"); ax[0, 1].set_ylabel(r"$\mathrm{MLP_{node}}$")
    ax[0, 1].legend(loc="upper right", frameon=False); panel(ax[0, 1], "b")

    # ---- (c) rate-constant recovery ----
    raw, trim, nout, slope = _plot_rate_constants_comparison(
        model, gt, log_dir, 0, N, device=dev, outlier_threshold=0.3)
    gtk = to_numpy(gt.log_k.detach().cpu()).ravel()
    lk = to_numpy(model.log_k.detach().cpu()).ravel()
    la, nsub = _compute_scalar_correction(model, dev); cor = lk + nsub * la
    nrx = len(gtk); outm = np.abs(cor - gtk) > 0.3; pct = 100.0 * nout / nrx
    lo, hi = min(gtk.min(), cor.min()) - 0.1, max(gtk.max(), cor.max()) + 0.1
    ax[1, 0].plot([lo, hi], [lo, hi], "--", c="gray", lw=1)
    ax[1, 0].scatter(gtk[~outm], cor[~outm], s=18, c="k", alpha=.6, edgecolors="none")
    ax[1, 0].scatter(gtk[outm], cor[outm], s=18, c="#e74c3c", alpha=.7, edgecolors="none")
    ax[1, 0].set_xlabel(r"true $\log_{10} k$"); ax[1, 0].set_ylabel(r"learned $\log_{10} k$")
    ax[1, 0].set_xlim(lo, hi); ax[1, 0].set_ylim(lo, hi)   # fill the panel like the others
    ax[1, 0].text(0.30, 0.18, f"raw $R^2$ = {raw:.2f}\ntrimmed $R^2$ = {trim:.2f}\n"
                  f"outliers {nout}/{nrx} ({pct:.0f}\\%)",
                  transform=ax[1, 0].transAxes, va="bottom", fontsize=11)
    panel(ax[1, 0], "c")

    # ---- (d) free rollout ----
    ds = config.dataset; dt = config.simulation.delta_t
    xt = torch.tensor(np.load(f"graphs_data/{ds}/x_list_0.npy"), dtype=torch.float32)
    sp = f"graphs_data/{ds}/stimulus.npy"; has_stim = os.path.exists(sp)
    stim = torch.tensor(np.load(sp), dtype=torch.float32) if has_stim else None
    T = min(2000, xt.shape[0] - 1); ctru = to_numpy(xt[:T + 1, :, 3])
    cp = np.zeros_like(ctru)
    with torch.no_grad():
        x = xt[0].clone(); cp[0] = to_numpy(x[:, 3])
        for t in range(T):
            s_ = stim[t] if has_stim else None
            x[:, 4] = xt[t, :, 4]          # feed the GIVEN external drive (col 4) at step t
            pr = model(pyg_Data(x=x.clone(), pos=x[:, 1:3]), stimulus=s_)
            x[:, 3:4] = x[:, 3:4] + dt * pr.reshape(-1, 1); cp[t + 1] = to_numpy(x[:, 3])
    thr = 5 * np.nanmax(np.abs(ctru)); dv = np.nanmax(np.abs(cp), axis=1) > thr
    tdiv = int(np.argmax(dv)) if dv.any() else T + 1
    # PER-METABOLITE Pearson (honest: within-trace dynamics, each metabolite vs itself)
    # alongside POOLED (flattering: dominated by between-metabolite level differences).
    gw, pw = ctru[:tdiv], cp[:tdiv]
    pm = [np.corrcoef(gw[:, i], pw[:, i])[0, 1] for i in range(gw.shape[1])
          if np.std(gw[:, i]) > 1e-6 and np.std(pw[:, i]) > 1e-6 and np.isfinite(pw[:, i]).all()]
    pe_pm = float(np.mean(pm)) if pm else float("nan")
    yt, yp = gw.ravel(), pw.ravel(); ok = np.isfinite(yp)
    pe = np.corrcoef(yt[ok], yp[ok])[0, 1] if np.std(yp[ok]) > 1e-9 else float("nan")
    tg = np.arange(T + 1) * dt; sel = np.argsort(-np.nanvar(ctru, axis=0))[:8]; SEP = 5
    for k, i in enumerate(sel):
        g, p = ctru[:, i], cp[:, i]; mu, sd = np.nanmean(g), np.nanstd(g) + 1e-9; off = k * SEP
        ax[1, 1].plot(tg, (g - mu) / sd + off, color=GT_C, lw=1.3)
        ax[1, 1].plot(tg, np.clip((p - mu) / sd, -0.45 * SEP, 0.45 * SEP) + off,
                      color=PRED_C, lw=1, ls="--")
    if tdiv <= T:
        ax[1, 1].axvline(tdiv * dt, color="#cc0000", ls=":", lw=1.2)
    ax[1, 1].plot([], [], color=GT_C, lw=1.3, label="ground truth")
    ax[1, 1].plot([], [], color=PRED_C, lw=1, ls="--", label="rollout")
    ax[1, 1].set_xlabel("time"); ax[1, 1].set_yticks([])
    ax[1, 1].set_ylabel("$z$-scored conc. (offset)")
    ax[1, 1].set_ylim(-SEP, len(sel) * SEP)
    ax[1, 1].legend(loc="lower right", frameon=False)
    ax[1, 1].text(0.97, 0.97, f"Pearson: {pe_pm:.2f} per-met / {pe:.2f} pooled",
                  transform=ax[1, 1].transAxes, va="top", ha="right", fontsize=11)
    panel(ax[1, 1], "d")

    fig.tight_layout()
    fname = "toy_dashboard.png" if cfg == "k_recovery_winner" else f"toy_dashboard_{cfg}.png"
    out_path = os.path.join(ROOT, "figures/metabolism", fname)
    fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"saved {out_path}: raw R2={raw:.3f} trim={trim:.3f} {pct:.0f}% out | "
          f"rollout Pearson per-met={pe_pm:.3f} / pooled={pe:.3f} | t_div={tdiv}")


if __name__ == "__main__":
    main()
