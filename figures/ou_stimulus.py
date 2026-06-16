#!/usr/bin/env python
"""
Realistic (non-oscillatory) stimulus does NOT rescue recovery. The real E. coli
metabolome is smooth/aperiodic, so we replaced the multi-sinusoid drive with an
Ornstein-Uhlenbeck/AR(1) drive (phi=0.98) and retrained on all three rungs.
  (a) the OU drive + a driven metabolite trajectory: smooth, aperiodic, low-frequency
      (matches the real metabolome's character, unlike a sinusoid);
  (b) Vmax recovery R^2 (~0, all three rungs) vs free-rollout per-metabolite Pearson
      (0.83-0.90): the SAME decoupling as under the sinusoid -> the learnability
      barrier is stimulus-shape-invariant.

Output: figures/metabolism/ou_stimulus.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as _mpl
_mpl.rcParams["axes.spines.top"] = False; _mpl.rcParams["axes.spines.right"] = False
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 10})

STIM_C = "#ff7f0e"   # stimulus = orange (convention)
# evaluated 2026-06-16 (leak-resistant Vmax + per-met rollout Pearson on OU datasets)
RUNGS = [("Rung 1\nglycolysis", "glyco_topo_ou", 0.008, 0.896),
         ("Rung 2\n$E.\\ coli$ core", "ecoli_core_ou", 0.003, 0.895),
         ("Rung 3\nyeast central", "yeast_central_ou", 0.005, 0.833)]


def main():
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[1.3, 1]))

    # (a) OU drive + a driven metabolite trajectory (e_coli core)
    x = np.load("graphs_data/ecoli_core_ou/x_list_0.npy")
    T = min(2000, x.shape[0])
    drive = x[:T, 0, 4]                                   # OU drive into metabolite 0
    conc = x[:T, :, 3]
    sel = np.argsort(-np.nanvar(conc, axis=0))[0]         # most-active metabolite
    t = np.arange(T)
    axd = ax[0]
    axd.plot(t, (drive - drive.mean()) / (drive.std() + 1e-9), color=STIM_C, lw=1.0,
             label="OU/AR(1) drive ($\\phi{=}0.98$)")
    c0 = conc[:, sel]
    axd.plot(t, (c0 - c0.mean()) / (c0.std() + 1e-9) - 5.0, color="k", lw=1.0,
             label="driven metabolite")
    axd.set_xlabel("time step"); axd.set_yticks([])
    axd.spines["left"].set_visible(False)
    axd.legend(loc="upper right", frameon=False)
    axd.text(-0.02, 1.04, "a", transform=axd.transAxes, fontsize=16, fontweight="bold",
             va="bottom", ha="right")
    axd.set_title("realistic drive: smooth, aperiodic (not a sinusoid)", fontsize=12,
                  loc="left", pad=10)

    # (b) Vmax R^2 vs rollout per-met, per rung
    axb = ax[1]
    xpos = np.arange(len(RUNGS)); w = 0.38
    vmax = [r[2] for r in RUNGS]; roll = [r[3] for r in RUNGS]
    axb.bar(xpos - w / 2, vmax, w, color="#999999", label="$V_{\\max}$ recovery $R^2$")
    axb.bar(xpos + w / 2, roll, w, color="k", label="rollout per-met Pearson")
    for xi, v in zip(xpos - w / 2, vmax):
        axb.text(xi, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(xpos + w / 2, roll):
        axb.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axb.set_xticks(xpos); axb.set_xticklabels([r[0] for r in RUNGS])
    axb.set_ylim(0, 1.18); axb.set_ylabel("score")
    axb.legend(loc="upper center", frameon=False, ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 1.10), columnspacing=1.2, handletextpad=0.5)
    axb.text(-0.02, 1.04, "b", transform=axb.transAxes, fontsize=16, fontweight="bold",
             va="bottom", ha="right")

    fig.tight_layout()
    out = os.path.join(ROOT, "figures/metabolism/ou_stimulus.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
