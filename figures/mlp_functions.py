#!/usr/bin/env python
"""
Diagnostic: can MLP_sub even represent the kinetic law?

Plots the two learned MLPs against ground truth:
  MLP_node(c, a_i)  -- the homeostasis term, ~ -lambda(c - c_base): the EASY one
                       (roughly linear / flat).
  MLP_sub(c, |s|)   -- the substrate kinetic law: the HARD one. For mass-action
                       it should be the power law c^|s| (c^1 easy, c^2 needs real
                       curvature); for Michaelis-Menten a saturating c/(Km+c).
                       MLP_sub also spans a large dynamic range under the
                       multiplicative aggregation, which is hard for an MLP.

Anchored honestly at c=1 (where c^s = 1 and the alpha-normalised MLP_sub ~ 1),
so the comparison shows SHAPE/curvature, not an arbitrary rescale.

Output: figures/metabolism/mlp_<config>.png
Usage:  python figures/mlp_functions.py [config_name]   (default k_recovery_winner)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))
os.chdir(ROOT)
from k_recovery import load_model            # reuse the exact model/gt loader
from MetabolismGraph.utils import to_numpy

plt.rcParams.update({"font.size": 14, "axes.labelsize": 16,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12})
LEARNED, TRUE = "k", "#2ca02c"             # predicted=black, GT=green


def panel_label(ax, letter):
    ax.text(0.015, 0.985, letter, transform=ax.transAxes, fontsize=20,
            fontweight="bold", va="top", ha="left")


def main():
    cfg = sys.argv[1] if len(sys.argv) > 1 else "k_recovery_winner"
    device = "cpu"
    config, model, gt_model, log_dir, ckpt = load_model(cfg, device)
    is_mm = "MichaelisMenten" in config.graph_model.model_name
    cmax = float(config.simulation.concentration_max)
    c = torch.linspace(0.05, cmax, 200, device=device)
    c_np = to_numpy(c)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # ---- (a) MLP_sub vs true kinetic law, for |s| = 1, 2 ----
    with torch.no_grad():
        for s_val, col in [(1, "#1f77b4"), (2, "#d62728")]:
            s = torch.full((len(c), 1), float(s_val), device=device)
            out = model.substrate_func(torch.cat([c.unsqueeze(-1), s], dim=-1))
            f = to_numpy(out.norm(dim=-1) if out.ndim > 1 else out.abs())
            # anchor at c=1: divide by learned value at c=1
            f1 = np.interp(1.0, c_np, f)
            ax[0].plot(c_np, f / max(f1, 1e-8), color=col, lw=2.2,
                       label=f"learned $|s|={s_val}$")
            if is_mm:
                Km = float(np.median(10.0 ** to_numpy(gt_model.log_km))) if hasattr(gt_model, "log_km") else 1.0
                true = (c_np / (Km + c_np)); true = true / np.interp(1.0, c_np, true)
                lbl = f"$c/(K_m+c)$ (true), |s|={s_val}" if s_val == 1 else None
            else:
                true = c_np ** s_val
                lbl = f"$c^{{{s_val}}}$ (true)"
            ax[0].plot(c_np, true, color=col, lw=1.6, ls="--", alpha=.9, label=lbl)
    ax[0].axvline(1.0, color="0.8", lw=1, zorder=0)
    ax[0].set_xlabel("concentration $c$")
    ax[0].set_ylabel(r"$\mathrm{MLP_{sub}}(c,|s|)$  (anchored at $c{=}1$)")
    ax[0].legend(loc="lower right", frameon=False); panel_label(ax[0], "a")

    # ---- (b) MLP_node per metabolite (the easy term) ----
    with torch.no_grad():
        a = model.a if hasattr(model, "a") else None
        N = config.simulation.n_metabolites
        for i in range(0, N, max(1, N // 12)):
            ai = a[i:i + 1].repeat(len(c), 1) if a is not None else torch.zeros(len(c), 0)
            h = model.node_func(torch.cat([c.unsqueeze(-1), ai], dim=-1)).squeeze(-1)
            ax[1].plot(c_np, to_numpy(h), color=LEARNED, lw=1, alpha=.5)
    ax[1].axhline(0, color=TRUE, lw=1.6, ls="--", label="true (homeostasis)")
    ax[1].set_xlabel("concentration $c$"); ax[1].set_ylabel(r"$\mathrm{MLP_{node}}(c,a_i)$")
    ax[1].legend(loc="upper right", frameon=False); panel_label(ax[1], "b")

    out = os.path.join(ROOT, "figures/metabolism",
                       f"mlp_{'glyco' if 'glyco' in cfg else cfg}.png")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    # quantify the c^2 curvature failure: learned slope ratio f(cmax)/f at c=1 vs true
    print(f"saved {out}  ({cfg}, {'MM' if is_mm else 'mass-action'})")
    return out


if __name__ == "__main__":
    main()
