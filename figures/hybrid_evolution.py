#!/usr/bin/env python
"""
Hybrid structured-MM curriculum: how the solution evolves as the kinetic SHAPE is
unfrozen. The substrate function is fixed to the exact MM form [c/(Km+c)]^|s| with a
learnable per-edge Km. Phase A (shaded): Km FROZEN at init (lr=0) while Vmax (log_k)
converges -> tests "given a fixed shape, can the GNN find the scale?". Phase B: the
Km learning rate is slowly ramped up -> watch Vmax-R2 and Km-R2 co-evolve.

Reads log/<cfg>/hybrid_recovery.npy (cols: epoch, N, Vmax_R2, Km_R2).
Output: figures/metabolism/hybrid_evolution.png
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
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11})

CFG = sys.argv[1] if len(sys.argv) > 1 else "ecoli_core_hybrid"
N_PHASE_A = int(sys.argv[2]) if len(sys.argv) > 2 else 15   # epochs with Km frozen
VMAX_C, KM_C = "#1f77b4", "#d62728"


def main():
    rec = np.load(f"log/{CFG}/hybrid_recovery.npy")          # (n_eval, 4)
    ep, _, vmax_r2, km_r2 = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]
    # last eval per epoch
    epochs = np.unique(ep).astype(int)
    v_last = np.array([vmax_r2[ep == e][-1] for e in epochs])
    k_last = np.array([km_r2[ep == e][-1] for e in epochs])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axvspan(epochs.min() - 0.5, N_PHASE_A - 0.5, color="0.9", zorder=0)
    ax.text(N_PHASE_A / 2.0, 1.02, "Phase A: $K_m$ frozen\n(learn $V_{\\max}$)",
            ha="center", va="bottom", fontsize=10, color="0.35")
    ax.text((N_PHASE_A + epochs.max()) / 2.0, 1.02, "Phase B: ramp $K_m$ lr",
            ha="center", va="bottom", fontsize=10, color="0.35")
    ax.axvline(N_PHASE_A - 0.5, color="0.6", ls=":", lw=1)

    ax.plot(epochs, np.clip(v_last, -0.1, 1.05), "-o", color=VMAX_C, lw=2, ms=4,
            label="$V_{\\max}$ recovery $R^2$")
    ax.plot(epochs, np.clip(k_last, -0.1, 1.05), "-s", color=KM_C, lw=2, ms=4,
            label="$K_m$ recovery $R^2$")
    ax.axhline(1.0, color="0.7", ls="--", lw=0.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("recovery $R^2$ (log space)")
    ax.set_ylim(-0.12, 1.12)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures/metabolism/hybrid_evolution.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}: final Vmax R2={v_last[-1]:.3f}, Km R2={k_last[-1]:.3f} "
          f"(phase-A end Vmax R2={v_last[min(N_PHASE_A-1, len(v_last)-1)]:.3f})")


if __name__ == "__main__":
    main()
