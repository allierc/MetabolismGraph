#!/usr/bin/env python
"""
Real E. coli metabolome (Link et al. 2015, nmeth.3584): characterise the data the
inverse problem would run on. (a) kinograph of the measured trajectory (247
metabolites x 119 time points, z-scored, sorted by activity); (b) SVD cumulative-
energy spectrum -> activity rank (rank90, rank99). The trajectory is high-rank/rich
(rank99~100); the obstacle for recovery is not rank but partial observation
(only the measured metabolites; reactions also touch unmeasured species).

Output: figures/metabolism/ecoli_real_data.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as _mpl
_mpl.rcParams["axes.spines.top"] = False; _mpl.rcParams["axes.spines.right"] = False
import matplotlib.pyplot as plt
import openpyxl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))
os.chdir(ROOT)
from amenability import load_organism, zscore_rows

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 10})


def main():
    wb = openpyxl.load_workbook("papers/nmeth3584/41592_2015_BFnmeth3584_MOESM197_ESM.xlsx",
                                read_only=True, data_only=True)
    tvec, C, _ = load_organism(wb, "Ecoli1", "Annotation Ecoli")
    Z = zscore_rows(C)                                  # (n_met, T)
    s = np.linalg.svd(Z, compute_uv=False)
    cum = np.cumsum(s ** 2) / (s ** 2).sum()
    r90 = int(np.searchsorted(cum, .90) + 1); r99 = int(np.searchsorted(cum, .99) + 1)
    order = np.argsort(-np.nanvar(Z, axis=1))           # sort metabolites by activity

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw=dict(width_ratios=[1.5, 1]))
    im = ax[0].imshow(Z[order], aspect="auto", origin="lower", cmap="viridis",
                      vmin=-2, vmax=2, extent=[tvec[0], tvec[-1], 0, C.shape[0]],
                      interpolation="nearest")
    ax[0].set_xlabel("time"); ax[0].set_ylabel("metabolite (sorted by activity)")
    ax[0].set_title(f"real E. coli metabolome ({C.shape[0]} metabolites $\\times$ {C.shape[1]} time points)",
                    fontsize=12, loc="left")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04, label="$z$-score")
    ax[0].text(-0.07, 1.02, "a", transform=ax[0].transAxes, fontsize=17, fontweight="bold", va="bottom")

    ax[1].plot(np.arange(1, len(cum) + 1), cum, "-", color="k", lw=2)
    ax[1].axhline(0.99, color="#d62728", ls=":", lw=1); ax[1].axhline(0.90, color="#1f77b4", ls=":", lw=1)
    ax[1].axvline(r99, color="#d62728", ls=":", lw=1); ax[1].axvline(r90, color="#1f77b4", ls=":", lw=1)
    ax[1].plot(r99, 0.99, "o", color="#d62728"); ax[1].plot(r90, 0.90, "o", color="#1f77b4")
    ax[1].text(r99 + 3, 0.965, f"rank$_{{99}}$={r99}", color="#d62728", fontsize=11)
    ax[1].text(r90 + 3, 0.86, f"rank$_{{90}}$={r90}", color="#1f77b4", fontsize=11)
    ax[1].set_xlabel("number of components"); ax[1].set_ylabel("cumulative variance explained")
    ax[1].set_ylim(0, 1.02)
    ax[1].text(0.0, 1.02, "b", transform=ax[1].transAxes, fontsize=17, fontweight="bold", va="bottom")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures/metabolism/ecoli_real_data.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}: {C.shape[0]} metabolites x {C.shape[1]} t, rank90={r90}, rank99={r99}")


if __name__ == "__main__":
    main()
