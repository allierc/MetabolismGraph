#!/usr/bin/env python
"""
The homeostasis confound — why the GNN can't recover Vmax, and the fix. All on the SAME
ecoli-core data with the EXACT MM shape (Km=GT). Vmax recovery R^2 under five conditions:

  1. SGD, no homeostasis            -> 1.00  (SGD recovers the scale when isolated)
  2. SGD, generic MLP_node homeo    -> fails (co-trained generic homeostasis swamps + scrambles)
  3. SGD, structured -lam(c-b) homeo -> 0.22 (structured fits, but joint SGD is ill-conditioned)
  4. LSQ, structured homeostasis     -> 1.00 (the system is linear in (Vmax,lam,mu): closed-form solve nails it)
  5. full GNN (oracle, exact Km)     -> 0.11 (the as-shipped end-to-end model)

Story: identifiability is fine (1 & 4); the barrier is (a) the generic homeostasis MLP_node and
(b) end-to-end SGD. The fix is to STRUCTURE homeostasis and SOLVE the parameters by least squares.

Output: figures/metabolism/homeostasis_confound.png
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
plt.rcParams.update({"font.size": 11.5, "axes.labelsize": 13, "legend.fontsize": 10})

# measured this campaign (ecoli_core, exact Km); generic-homeo R2 is strongly negative -> show as ~0 "fails"
BARS = [
    ("SGD\nno homeostasis", 1.00, "#2ca02c", False),
    ("SGD\ngeneric MLP homeo", 0.0, "#d62728", True),
    ("SGD\nstructured homeo", 0.22, "#ff7f0e", False),
    ("LSQ\nstructured homeo", 1.00, "#1f77b4", False),
    ("full GNN\n(oracle, exact Km)", 0.11, "#7f7f7f", False),
]


def main():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    xs = np.arange(len(BARS))
    for i, (lab, v, c, fails) in enumerate(BARS):
        ax.bar(i, max(v, 0.012), 0.66, color=c)
        ax.text(i, max(v, 0.012) + 0.02, ("fails ($R^2{<}0$)" if fails else f"{v:.2f}"),
                ha="center", va="bottom", fontsize=10, color=c, fontweight="bold")
    ax.axhline(1.0, color="0.7", ls="--", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in BARS])
    ax.set_ylabel("$V_{\\max}$ recovery $R^2$"); ax.set_ylim(0, 1.14)
    ax.set_title("Given the EXACT MM shape (Km=GT): the homeostasis confound, and the fix",
                 fontsize=12, loc="left")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures/metabolism/homeostasis_confound.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
