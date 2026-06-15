#!/usr/bin/env python
"""
Rung-1 glycolysis: does MLP_sub learn the substrate kinetics, and does the log-exp
form help? Glycolysis is Michaelis--Menten, so the true substrate law SATURATES,
[c/(K_m+c)]^|s| (not a power law). We plot the learned MLP_sub(c,|s|) for the
glycolysis substrate orders (|s|=1,2,4) against that MM truth, for the plain MLP
(glyco_ar_base) and the log-exp parameterisation (glyco_logspace), anchored at c=1.

Output: figures/metabolism/glyco_sub_compare.png
Usage:  python figures/glyco_sub_compare.py
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
from k_recovery import load_model
from MetabolismGraph.utils import to_numpy

plt.rcParams.update({"font.size": 14, "axes.labelsize": 16,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11})
COLORS = {1: "#1f77b4", 2: "#d62728", 4: "#2ca02c"}   # glycolysis orders 1, 2, 4


def panel_label(ax, letter):
    ax.text(0.015, 0.985, letter, transform=ax.transAxes, fontsize=20,
            fontweight="bold", va="top", ha="left")


def main():
    CMAX = 4.3                                  # glycolysis concentration range
    c = torch.linspace(0.05, CMAX, 200); cn = to_numpy(c)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for j, (cfg, title, tag, vmax_r2) in enumerate([
            ("glyco_ar_base", "plain MLP", "a", 0.00),
            ("glyco_logspace", "log-space MLP", "b", 0.04)]):
        _, m, gt, _, _ = load_model(cfg, "cpu")
        Km = float(np.median(10.0 ** to_numpy(gt.log_km))) if hasattr(gt, "log_km") else 1.0
        with torch.no_grad():
            for s in (1, 2, 4):
                sv = torch.full((200, 1), float(s))
                o = m.substrate_func(torch.cat([c.unsqueeze(-1), sv], dim=-1))
                f = to_numpy(o.norm(dim=-1) if o.ndim > 1 else o.abs())
                f = f / np.interp(1.0, cn, f)                      # learned, anchored at c=1
                tru = (cn / (Km + cn)) ** s; tru = tru / np.interp(1.0, cn, tru)  # MM truth, anchored
                ax[j].plot(cn, f, color=COLORS[s], lw=2.4, label=f"learned $|s|={s}$")
                ax[j].plot(cn, tru, color=COLORS[s], lw=1.5, ls="--", alpha=.8,
                           label=rf"$[c/(K_m+c)]^{{{s}}}$ (true)")
        ax[j].set_yscale("log"); ax[j].set_xlabel("concentration $c$")
        ax[j].text(0.97, 0.04, rf"$V_{{\max}}$ recovery $R^2 = {vmax_r2:.2f}$",
                   transform=ax[j].transAxes, va="bottom", ha="right", fontsize=12)
        ax[j].legend(loc="lower right", frameon=False, ncol=1, fontsize=9,
                     bbox_to_anchor=(1.0, 0.10))
        panel_label(ax[j], tag)
    ax[0].set_ylabel(r"$\mathrm{MLP_{sub}}(c,|s|)$  (anchored at $c{=}1$)")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures/metabolism/glyco_sub_compare.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
