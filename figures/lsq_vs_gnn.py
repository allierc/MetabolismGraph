#!/usr/bin/env python
"""
Least squares does the job; the GNN hardly does it yet. Across the three rungs
(realistic OU-stimulus, imposed-MM kinetics), Vmax recovery R^2 for:
  - least squares given the true substrate shape (the achievable upper bound),
    built from the simulator's exact Jacobian (scripts/design_matrix2.py);
  - the trained GNN (leak-resistant k_recovery).
Where the dynamics are genuinely linear in Vmax (rungs 1-2, reconstruction
residual ~1e-4) LSQ recovers Vmax PERFECTLY (R^2=1.00) while the GNN sits at ~0.
Rung 3 (yeast) is flux-limit-clamped, so even the linear design matrix breaks
(residual 0.30) and LSQ degrades too -- a simulator artefact, flagged.

Output: figures/metabolism/lsq_vs_gnn.png
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

LSQ_C, GNN_C = "#1f77b4", "k"      # LSQ = blue, GNN (the model) = black
# verified 2026-06-18: LSQ from design_matrix2.py (exact Jacobian), GNN from k_recovery.py
RUNGS = [  # label,                lsq_r2, gnn_r2, residual_ok
    ("Rung 1\nglycolysis",        1.000, 0.008, True),
    ("Rung 2\n$E.\\ coli$ core",  1.000, 0.003, True),
    ("Rung 3\nyeast central",     0.124, 0.005, False),
]


def main():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xpos = np.arange(len(RUNGS)); w = 0.38
    lsq = [r[1] for r in RUNGS]; gnn = [r[2] for r in RUNGS]
    ax.bar(xpos - w / 2, lsq, w, color=LSQ_C,
           label="least squares (true shape)")
    ax.bar(xpos + w / 2, gnn, w, color=GNN_C, label="GNN (learned)")
    for xi, v in zip(xpos - w / 2, lsq):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10, color=LSQ_C)
    for xi, v in zip(xpos + w / 2, gnn):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    # flag the rung whose linear design matrix is invalidated by flux clamping
    for k, r in enumerate(RUNGS):
        if not r[3]:
            ax.text(xpos[k] - w / 2, 0.30, "flux-limit\nbreaks linear $A$\n(residual 0.30)",
                    ha="center", va="bottom", fontsize=8.5, color=LSQ_C, style="italic")
    ax.set_xticks(xpos); ax.set_xticklabels([r[0] for r in RUNGS])
    ax.set_ylim(0, 1.12); ax.set_ylabel("$V_{\\max}$ recovery $R^2$")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend(loc="upper center", frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout()
    out = os.path.join(ROOT, "figures/metabolism/lsq_vs_gnn.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
