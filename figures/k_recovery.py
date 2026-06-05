#!/usr/bin/env python
"""
Toy-model rate-constant (k) recovery figure.

Recapitulates the analysis behind figures/metabolism/k_recovery.png: load a
trained leak-free checkpoint, recompute the learned-vs-true rate constants with
the exact recovery computation used in training (scalar/alpha correction,
outlier rule), and report:
  - raw R^2 (all reactions)
  - trimmed R^2 (excluding outliers)
  - number and PERCENT of outlier reactions (|corrected log10 k - true| > 0.3)
  - HARD RULE: outlier fraction must not exceed 10%  ->  PASS / FAIL badge.

Reuses graph_trainer._plot_rate_constants_comparison (authoritative R^2) and
_compute_scalar_correction (alpha correction) so the numbers match the ledger.
Runs on CPU by default to leave the GPUs for training.

Usage:  python figures/k_recovery.py [config_name]   (default: k_recovery_winner)
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
from MetabolismGraph.config import MetabolismGraphConfig
from MetabolismGraph.models.Metabolism_Propagation import Metabolism_Propagation
from MetabolismGraph.models.graph_trainer import (
    _plot_rate_constants_comparison, _compute_scalar_correction)
from MetabolismGraph.generators.PDE_M1 import PDE_M1
from MetabolismGraph.utils import to_numpy

plt.rcParams.update({"font.size": 13, "axes.labelsize": 15,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})
OUTLIER_THRESHOLD = 0.3      # |corrected log10 k - true| in dex
HARD_RULE_PCT = 10.0         # outlier fraction must not exceed this


def panel_label(ax, letter):
    ax.text(0.015, 0.985, letter, transform=ax.transAxes, fontsize=18,
            fontweight="bold", va="top", ha="left")


def load_model(config_name, device):
    config = MetabolismGraphConfig.from_yaml(f"config/{config_name}.yaml")
    config.config_file = config_name
    ds = config.dataset
    log_dir = os.path.join("./log", config_name)
    stoich_graph = torch.load(f"graphs_data/{ds}/stoich_graph.pt", map_location=device)

    # ground-truth model (PDE_M1 for the synthetic mass-action regime)
    gt_state = torch.load(f"graphs_data/{ds}/gt_model.pt", map_location=device)
    gt_model = PDE_M1(config=config, stoich_graph=stoich_graph, device=device)
    gt_model.load_state_dict(gt_state, strict=False); gt_model.to(device).eval()

    # trained model: pick the latest 'best_model_*' checkpoint
    cks = sorted(glob.glob(f"{log_dir}/models/best_model_with_*graphs_*.pt"),
                 key=os.path.getmtime)
    if not cks:
        raise FileNotFoundError(f"no checkpoint in {log_dir}/models")
    model = Metabolism_Propagation(config=config, device=device)
    model.load_stoich_graph(stoich_graph)
    model.load_state_dict(torch.load(cks[-1], map_location=device)["model_state_dict"])
    model.to(device).eval()
    return config, model, gt_model, log_dir, os.path.basename(cks[-1])


def main():
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "k_recovery_winner"
    device = "cpu"
    config, model, gt_model, log_dir, ckpt = load_model(cfg_name, device)
    N = config.simulation.n_metabolites

    # authoritative R^2 numbers (same computation as training)
    raw_r2, trimmed_r2, n_out, slope = _plot_rate_constants_comparison(
        model, gt_model, log_dir, 0, N, device=device,
        outlier_threshold=OUTLIER_THRESHOLD)

    # arrays for the scatter (corrected log k vs true)
    gt_log_k = to_numpy(gt_model.log_k.detach().cpu()).ravel()
    learned = to_numpy(model.log_k.detach().cpu()).ravel()
    log_alpha, n_sub = _compute_scalar_correction(model, device)
    corrected = learned + n_sub * log_alpha
    n_rxn = len(gt_log_k)
    err = np.abs(corrected - gt_log_k)
    outlier = err > OUTLIER_THRESHOLD
    pct = 100.0 * n_out / n_rxn
    passed = pct <= HARD_RULE_PCT

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    lo = min(gt_log_k.min(), corrected.min()) - 0.1
    hi = max(gt_log_k.max(), corrected.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "--", c="gray", lw=1, zorder=1)
    ax.scatter(gt_log_k[~outlier], corrected[~outlier], s=26, c="k", alpha=.65,
               edgecolors="none", label="inlier", zorder=2)
    ax.scatter(gt_log_k[outlier], corrected[outlier], s=26, c="#e74c3c", alpha=.75,
               edgecolors="none", label=f"outlier (>{OUTLIER_THRESHOLD} dex)", zorder=3)
    ax.set_xlabel(r"true $\log_{10} k_j$"); ax.set_ylabel(r"learned $\log_{10} k_j$")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False)
    panel_label(ax, "a")

    txt = (f"raw $R^2$ = {raw_r2:.3f}\n"
           f"trimmed $R^2$ = {trimmed_r2:.3f}\n"
           f"outliers = {n_out}/{n_rxn} = {pct:.1f}%\n"
           f"slope = {slope:.2f}")
    ax.text(0.035, 0.86, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=13, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7"))

    out = os.path.join(ROOT, "figures/metabolism/k_recovery.png")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}  ({cfg_name}, ckpt {ckpt})")
    print(f"  raw R2={raw_r2:.3f}  trimmed R2={trimmed_r2:.3f}  "
          f"outliers={n_out}/{n_rxn} ({pct:.1f}%)  slope={slope:.2f}  "
          f"-> {HARD_RULE_PCT:.0f}% rule {badge}")
    return dict(config=cfg_name, raw_r2=raw_r2, trimmed_r2=trimmed_r2,
                n_out=n_out, n_rxn=n_rxn, pct=pct, slope=slope, passed=passed)


if __name__ == "__main__":
    main()
