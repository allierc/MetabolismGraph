#!/usr/bin/env python
"""
Hybrid structured-MM matrix summary: final Vmax (log_k) and Km recovery R^2 across the
three regimes -- ORACLE (Km=GT frozen, upper bound), JOINT (Km learnable from epoch 0),
CURRICULUM (freeze Km then ramp) -- per rung. Reads log/<cfg>/hybrid_recovery.npy for the
last-epoch (Vmax_R2, Km_R2). Tells the story: how close does each strategy get to the
oracle, and does the freeze->ramp curriculum beat learning shape+scale jointly?

Output: figures/metabolism/hybrid_summary.png
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

# rung -> {regime: config}
RUNGS = [
    ("Rung 1\nglycolysis", {"oracle": "glyco_hybrid_oracle", "curriculum": "glyco_hybrid"}),
    ("Rung 2\n$E.\\ coli$ core", {"oracle": "ecoli_core_hybrid_oracle",
                                  "joint": "ecoli_core_hybrid_joint",
                                  "curriculum": "ecoli_core_hybrid"}),
    ("Rung 3\nyeast central", {"oracle": "yeast_hybrid_oracle", "curriculum": "yeast_hybrid"}),
]
REGIMES = ["oracle", "joint", "curriculum"]
RC = {"oracle": "#2ca02c", "joint": "#999999", "curriculum": "#1f77b4"}


def final_r2(cfg):
    p = f"log/{cfg}/hybrid_recovery.npy"
    if not os.path.exists(p):
        return None
    r = np.load(p)
    if len(r) == 0:
        return None
    last = r[r[:, 0] == np.unique(r[:, 0]).max()][-1]
    return float(last[2]), float(last[3])      # Vmax_R2, Km_R2


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for col, (ax, which, ylab) in enumerate(
            [(axes[0], 0, "$V_{\\max}$ recovery $R^2$"),
             (axes[1], 1, "$K_m$ recovery $R^2$")]):
        xticks, xlabels = [], []
        x = 0
        for rung_lab, cfgs in RUNGS:
            base = x
            for reg in REGIMES:
                if reg not in cfgs:
                    continue
                res = final_r2(cfgs[reg])
                val = (res[which] if res else np.nan)
                ax.bar(x, np.clip(val, -0.1, 1.05) if np.isfinite(val) else 0,
                       0.8, color=RC[reg], label=reg if (rung_lab.startswith("Rung 2")) else None)
                if np.isfinite(val):
                    ax.text(x, max(val, 0) + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
                x += 1
            xticks.append((base + x - 1) / 2.0); xlabels.append(rung_lab)
            x += 1   # gap between rungs
        ax.axhline(1.0, color="0.7", ls="--", lw=0.8)
        ax.set_xticks(xticks); ax.set_xticklabels(xlabels)
        ax.set_ylabel(ylab); ax.set_ylim(-0.12, 1.12)
        ax.text(0.0, 1.03, "ab"[col], transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
        if col == 0:
            ax.legend(loc="upper right", frameon=False, title="regime")
    fig.suptitle("Hybrid structured-MM: oracle (Km=GT) vs joint vs freeze→ramp curriculum",
                 fontsize=12, x=0.5, y=1.0)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures/metabolism/hybrid_summary.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    # also print a text table
    print("config                         Vmax_R2   Km_R2")
    for _, cfgs in RUNGS:
        for reg in REGIMES:
            if reg in cfgs:
                res = final_r2(cfgs[reg])
                if res:
                    print(f"{cfgs[reg]:30s} {res[0]:+.3f}   {res[1]:+.3f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
