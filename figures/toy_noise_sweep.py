#!/usr/bin/env python
"""
Companion to the toy dashboard (Fig 1): INTRINSIC-NOISE sweep.

Tests the flyvis finding (Lappalainen/Allier) in the metabolism toy: intrinsic
process noise injected during data generation BREAKS the identifiability
degeneracy of the autonomous toy. We sweep simulation.noise_model_level and,
for each level, train single-step (leak-resistant k-recovery, S given) on the
NOISY trajectory, then evaluate against BOTH ground truths:
  - the noisy GT (what the model trained on) -- rollout can't match the noise,
  - the NOISE-FREE GT (toy_noise_000, same network/seed) -- the deterministic
    model rollout SHOULD reproduce it (it cannot fit the stochastic fluctuations
    and converges on the underlying deterministic dynamics).

Panels:
  (a) activity rank(99%) vs noise          -- degeneracy breaking, data side
  (b) k-recovery R^2 + %outliers vs noise   -- identifiability, the headline
  (c) rollout per-met Pearson vs noise, vs noise-free GT (green) and noisy GT (gray)
  (d) example rollout traces at one level: noisy GT (gray) / noise-free GT (green)
      / model rollout (black) -- the model recovers clean dynamics from noisy data

Output: figures/metabolism/toy_noise_sweep.png
Usage:  python figures/toy_noise_sweep.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data as pyg_Data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))
os.chdir(ROOT)
from k_recovery import load_model, OUTLIER_THRESHOLD, HARD_RULE_PCT
from MetabolismGraph.models.graph_trainer import _plot_rate_constants_comparison
from MetabolismGraph.utils import to_numpy

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10})
GT_C, PRED_C, NOISY_C = "#2ca02c", "k", "#999999"   # clean green, rollout black, noisy gray

# (tag, sigma). 000 is the noise-free twin used as the clean rollout reference.
LEVELS = [("000", 0.0), ("001", 0.01), ("002", 0.02),
          ("003", 0.03), ("005", 0.05), ("007", 0.07)]
CLEAN_DS = "toy_noise_000"
# noise levels shown as side-by-side rollout columns (noise-free / low / high)
DISPLAY_TAGS = ["000", "005", "007"]
DISPLAY_TITLES = {"000": r"noise-free  $\sigma=0$",
                  "005": r"low noise  $\sigma=0.05$",
                  "007": r"high noise  $\sigma=0.07$"}


def panel(ax, L):
    ax.text(0.0, 1.02, L, transform=ax.transAxes, fontsize=18,
            fontweight="bold", va="bottom", ha="left")


def rank99(ds):
    """activity rank at 99% cumulative singular-value energy (same as the generator)."""
    x = np.load(f"graphs_data/{ds}/x_list_0.npy")[:, :, 3]   # frames x metabolites
    x = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(e, 0.99) + 1)


def k_metrics(model, gt_model, log_dir, N, device):
    # authoritative computation (identical to figures/k_recovery.py / training)
    raw, trimmed, n_out, slope = _plot_rate_constants_comparison(
        model, gt_model, log_dir, 0, N, device=device, outlier_threshold=OUTLIER_THRESHOLD)
    n_rxn = len(to_numpy(gt_model.log_k.detach().cpu()).ravel())
    return raw, trimmed, 100.0 * n_out / n_rxn


def rollout(model, config, ref_ds, device):
    """Deterministic free rollout from ref_ds frame 0; returns (time, ctru, cp)."""
    dt = config.simulation.delta_t
    xt = torch.tensor(np.load(f"graphs_data/{ref_ds}/x_list_0.npy"), dtype=torch.float32)
    T = min(2000, xt.shape[0] - 1)
    ctru = to_numpy(xt[:T + 1, :, 3])
    cp = np.zeros_like(ctru)
    with torch.no_grad():
        x = xt[0].clone(); cp[0] = to_numpy(x[:, 3])
        for t in range(T):
            x[:, 4] = xt[t, :, 4]
            pr = model(pyg_Data(x=x.clone(), pos=x[:, 1:3]), stimulus=None)
            x[:, 3:4] = x[:, 3:4] + dt * pr.reshape(-1, 1)
            cp[t + 1] = to_numpy(x[:, 3])
    return np.arange(T + 1) * dt, ctru, cp


def per_met_pearson(gt, pr):
    v = [np.corrcoef(gt[:, i], pr[:, i])[0, 1] for i in range(gt.shape[1])
         if np.std(gt[:, i]) > 1e-6 and np.std(pr[:, i]) > 1e-6 and np.isfinite(pr[:, i]).all()]
    return float(np.mean(v)) if v else float("nan")


def main():
    dev = "cpu"
    sig, rk, kraw, ktrim, kout, pm_clean, pm_noisy = [], [], [], [], [], [], []
    traces = {}
    clean_traj = np.load(f"graphs_data/{CLEAN_DS}/x_list_0.npy")[:, :, 3]

    for tag, s in LEVELS:
        cfg = f"toy_noise_{tag}"
        try:
            config, model, gt_model, log_dir, _ = load_model(cfg, dev)
        except Exception as e:
            print(f"skip {cfg}: {e}"); continue
        raw, trim, pct = k_metrics(model, gt_model, log_dir, config.simulation.n_metabolites, dev)
        # one deterministic rollout from the shared frame-0 state (clean reference)
        tg, ctru_clean, cp = rollout(model, config, CLEAN_DS, dev)
        T = cp.shape[0]
        noisy_traj = np.load(f"graphs_data/{cfg}/x_list_0.npy")[:T, :, 3]
        sig.append(s); rk.append(rank99(cfg))
        kraw.append(raw); ktrim.append(trim); kout.append(pct)
        pm_clean.append(per_met_pearson(ctru_clean, cp))
        pm_noisy.append(per_met_pearson(noisy_traj, cp))
        if tag in DISPLAY_TAGS:
            traces[tag] = (tg, noisy_traj, cp)
        print(f"sigma={s}: rank={rk[-1]} kraw={raw:.3f} ktrim={trim:.3f} out={pct:.0f}% "
              f"pm_clean={pm_clean[-1]:.3f} pm_noisy={pm_noisy[-1]:.3f}")

    sig = np.array(sig)
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(15, 9.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.35, 1.0], hspace=0.32, wspace=0.22)

    # ---- TOP ROW: per-noise-level rollout columns ----
    # noisy GT (green, jaggier as sigma grows) overlaid with the model rollout (black).
    # The deterministic prediction stays smooth while the GT it is scored against is noisy.
    sel = np.argsort(-np.nanvar(clean_traj, axis=0))[:9]          # same metabolites in every column
    SEP = 4.5
    pm_by_tag = {t: pm_clean[i] for i, (t, _) in enumerate(LEVELS) if i < len(pm_clean)}
    for col, tag in enumerate(DISPLAY_TAGS):
        axt = fig.add_subplot(gs[0, col])
        if tag in traces:
            tg, gt_noisy, cp = traces[tag]
            Tn = min(len(tg), gt_noisy.shape[0], cp.shape[0])
            for k, i in enumerate(sel):
                mu = np.nanmean(clean_traj[:, i]); sd = np.nanstd(clean_traj[:, i]) + 1e-9
                off = k * SEP
                axt.plot(tg[:Tn], (gt_noisy[:Tn, i] - mu) / sd + off, color=GT_C, lw=1.0, alpha=0.9)
                axt.plot(tg[:Tn], np.clip((cp[:Tn, i] - mu) / sd, -0.48 * SEP, 0.48 * SEP) + off,
                         color=PRED_C, lw=0.6)
            axt.text(0.97, 0.015, rf"per-met $r={pm_by_tag.get(tag, float('nan')):.2f}$",
                     transform=axt.transAxes, va="bottom", ha="right", fontsize=11)
        axt.set_title(DISPLAY_TITLES[tag], fontsize=15, fontweight="bold", loc="left")
        axt.set_xlabel("time (frames)"); axt.set_yticks([])
        if col == 0:
            axt.set_ylabel("$z$-scored conc. (offset)")
    # shared legend (top center, like the reference)
    fig.legend(handles=[plt.Line2D([], [], color=GT_C, lw=2.4, label="ground truth (noisy)"),
                        plt.Line2D([], [], color=PRED_C, lw=1.2, label="model rollout (prediction)")],
               loc="upper center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.0))

    # ---- BOTTOM ROW: the quantitative sweep ----
    # (a) k-recovery vs noise
    axa = fig.add_subplot(gs[1, 0])
    axa.plot(sig, kraw, "o-", color="k", lw=2, label=r"raw $R^2$")
    axa.plot(sig, ktrim, "s--", color="#1f77b4", lw=1.6, label=r"trimmed $R^2$")
    axa.set_xlabel(r"intrinsic noise $\sigma$"); axa.set_ylabel(r"$k$-recovery $R^2$")
    axa.legend(loc="lower left", frameon=False)
    axb = axa.twinx()
    axb.plot(sig, kout, "^:", color="#d62728", lw=1.4)
    axb.axhline(HARD_RULE_PCT, color="#d62728", ls=":", lw=1, alpha=0.5)
    axb.set_ylabel("% outliers", color="#d62728"); axb.tick_params(axis="y", colors="#d62728")
    panel(axa, "a")

    # (b) rollout per-met Pearson vs noise (vs noise-free GT green, vs noisy GT gray)
    axc = fig.add_subplot(gs[1, 1])
    axc.plot(sig, pm_clean, "o-", color=GT_C, lw=2, label="vs noise-free GT")
    axc.plot(sig, pm_noisy, "o--", color=NOISY_C, lw=1.8, label="vs noisy GT")
    axc.set_xlabel(r"intrinsic noise $\sigma$"); axc.set_ylabel("rollout per-met Pearson")
    axc.legend(loc="upper right", frameon=False)
    panel(axc, "b")

    # (c) activity rank vs noise (data diversity rises -- the degeneracy-breaking signature)
    axd = fig.add_subplot(gs[1, 2])
    axd.plot(sig, rk, "o-", color="k", lw=2)
    axd.set_xlabel(r"intrinsic noise $\sigma$"); axd.set_ylabel("activity rank (99%)")
    panel(axd, "c")

    out = os.path.join(ROOT, "figures/metabolism/toy_noise_sweep.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
