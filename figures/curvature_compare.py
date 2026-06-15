#!/usr/bin/env python
"""
Curvature recovery: plain MLP vs log-space MLP_sub, on a regime WITH |s|>=2.

Recapitulates figures/metabolism/curvature_compare.png. On the mixed_s2 regime
(31% of substrate edges have |s|>=2, so there IS quadratic+ curvature to learn),
plots the learned substrate function MLP_sub(c,|s|) for |s|=1,2,3 against the true
power law c^s. Both are anchored at c=1 (where c^s=1), so the curves show SHAPE.

Finding: with real |s|>=2 training signal both learn some curvature (unlike the
all-|s|=1 toy where |s|=2 was untrained extrapolation), and log-space learns MORE,
but both still undershoot the true power law badly for c^2 and especially c^3.

Output: figures/metabolism/curvature_compare.png
Usage:  python figures/curvature_compare.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as _mpl
_mpl.rcParams["axes.spines.top"] = False; _mpl.rcParams["axes.spines.right"] = False  # bare x/y axes
import matplotlib.pyplot as plt
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))
os.chdir(ROOT)
from k_recovery import load_model
from MetabolismGraph.utils import to_numpy

plt.rcParams.update({"font.size": 14, "axes.labelsize": 16,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12})
COLORS = {1: "#1f77b4", 2: "#d62728", 4: "#2ca02c"}   # glycolysis orders: 1, 2, 4 (no 3)


def panel_label(ax, letter):
    ax.text(0.015, 0.985, letter, transform=ax.transAxes, fontsize=20,
            fontweight="bold", va="top", ha="left")


def curves(cfg):
    _, m, _, _, _ = load_model(cfg, "cpu")
    c = torch.linspace(0.05, 9.0, 200); cn = to_numpy(c)
    out = {}
    with torch.no_grad():
        for s in (1, 2, 4):
            sv = torch.full((200, 1), float(s))
            o = m.substrate_func(torch.cat([c.unsqueeze(-1), sv], dim=-1))
            f = to_numpy(o.norm(dim=-1) if o.ndim > 1 else o.abs())
            out[s] = f / np.interp(1.0, cn, f)
    return cn, out


def main():
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for j, (cfg, title, tag) in enumerate([
            ("mixed_s2", "plain MLP", "a"),
            ("mixed_s2_logspace", "log-space MLP", "b")]):
        cn, out = curves(cfg)
        # growth ratio f(c=9)/f(c=1)=f(9) (anchored at 1); true = 9^s
        ratios = {s: float(np.interp(9.0, cn, out[s])) for s in (1, 2, 4)}
        print(f"{cfg:22s} ({title:13s}) growth f(9)/f(1):  "
              f"|s|=1 {ratios[1]:6.1f} (true 9) | |s|=2 {ratios[2]:7.1f} (true 81) | "
              f"|s|=4 {ratios[4]:9.1f} (true 6561)")
        for s in (1, 2, 4):
            ax[j].plot(cn, out[s], color=COLORS[s], lw=2.4, label=f"learned $|s|={s}$")
            ax[j].plot(cn, cn ** s, color=COLORS[s], lw=1.5, ls="--", alpha=.8,
                       label=f"$c^{{{s}}}$ (true)")
        ax[j].set_yscale("log"); ax[j].set_xlabel("concentration $c$")
        ax[j].legend(loc="lower right", frameon=False, ncol=1, fontsize=10)
        panel_label(ax[j], tag)
    ax[0].set_ylabel(r"$\mathrm{MLP_{sub}}(c,|s|)$  (anchored at $c{=}1$)")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures/metabolism/curvature_compare.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
