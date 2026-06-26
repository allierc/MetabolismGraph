#!/usr/bin/env python
"""
Companion to fig:model_traces, for the LEAST-SQUARES estimator. Same three rungs and
the same ground-truth drive as model_traces.py, but the black trace is the free rollout
of the model whose per-reaction Vmax is set by a least-squares solve given the true
substrate shape (no GNN). Ground truth (green) vs least-squares rollout (black),
z-scored and stacked; per-metabolite Pearson annotated.

Output: figures/metabolism/lsq_rollout_traces.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as _mpl
_mpl.rcParams["axes.spines.top"] = False; _mpl.rcParams["axes.spines.right"] = False
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data as pyg_Data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))
os.chdir(ROOT)
from MetabolismGraph.config import MetabolismGraphConfig
from MetabolismGraph.generators.PDE_MichaelisMenten import PDE_MichaelisMenten
from MetabolismGraph.utils import to_numpy


def lsq_vmax_model(config, ds, stim=None):
    """PDE model with per-reaction Vmax set by a least-squares solve given the true
    substrate shape (the simulator's exact Jacobian). Boundary-species stimulus, when
    present, is fed through both the solve and the rollout."""
    sg = torch.load(f"graphs_data/{ds}/stoich_graph.pt", map_location="cpu")
    torch.manual_seed(0)
    m = PDE_MichaelisMenten(config=config, stoich_graph=sg, device="cpu")
    m.load_state_dict(torch.load(f"graphs_data/{ds}/gt_model.pt", map_location="cpu"), strict=False)
    m.eval()
    x = np.load(f"graphs_data/{ds}/x_list_0.npy"); dt = config.simulation.delta_t
    nrxn = config.simulation.n_reactions; nmet = config.simulation.n_metabolites
    lk0 = m.log_k.detach().clone(); Vgt = to_numpy(10 ** lk0); eps = 1e-3

    def f(xt, t):
        s_ = stim[t] if stim is not None else None
        return to_numpy(m(pyg_Data(x=torch.tensor(xt, dtype=torch.float32),
                          pos=torch.tensor(xt[:, 1:3], dtype=torch.float32)),
                          dt=dt, stimulus=s_).reshape(-1))

    Ts = np.linspace(0, x.shape[0] - 2, 120, dtype=int); Ab, bb = [], []
    for t in Ts:
        xt = x[t].copy(); d0 = f(xt, t)
        with torch.no_grad():
            m.log_k.copy_(torch.full_like(lk0, -30.)); g = f(xt, t); m.log_k.copy_(lk0)
        bb.append(d0 - g); At = np.zeros((nmet, nrxn))
        for j in range(nrxn):
            with torch.no_grad():
                m.log_k[j] += eps; dj = f(xt, t); m.log_k.copy_(lk0)
            At[:, j] = (dj - d0) / (Vgt[j] * (10 ** eps - 1))
        Ab.append(At)
    Vhat, *_ = np.linalg.lstsq(np.vstack(Ab), np.concatenate(bb), rcond=None)
    with torch.no_grad():
        m.log_k.copy_(torch.tensor(np.log10(np.clip(Vhat, 1e-9, None)), dtype=torch.float32))
    return m

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11,
                     "legend.fontsize": 10})
GT_C, PRED_C = "#2ca02c", "k"
# two dataset groups: the sinusoidal-stimulus rungs, and the realistic OU-stimulus rungs
GROUPS = {
    "stim": ([("glyco_ar_base", "Rung 1: yeast glycolysis"),
              ("ecoli_core_stim", "Rung 3: E. coli core (driven)"),
              ("yeast_central_stim", "Rung 2: yeast-GEM subgraph (driven)")],
             "lsq_rollout_traces.png"),
    "ou":   ([("glyco_topo_ou", "Rung 1: glycolysis (OU drive)"),
              ("ecoli_core_ou", "Rung 2: E. coli core (OU drive)"),
              ("yeast_central_ou", "Rung 3: yeast central (OU drive)")],
             "lsq_rollout_traces_ou.png"),
}
MODELS, OUTNAME = GROUPS[sys.argv[1] if len(sys.argv) > 1 else "stim"]


def rollout(model, x, dt, stimulus=None):
    T = min(2000, x.shape[0] - 1)
    ctru = x[:T + 1, :, 3]; cp = np.zeros_like(ctru); cp[0] = x[0, :, 3]
    xt = torch.tensor(x[0], dtype=torch.float32)
    with torch.no_grad():
        for t in range(T):
            xt[:, 4] = torch.tensor(x[t, :, 4], dtype=torch.float32)
            s_ = stimulus[t] if stimulus is not None else None
            pr = model(pyg_Data(x=xt.clone(), pos=xt[:, 1:3]), stimulus=s_, dt=dt)
            xt[:, 3] = xt[:, 3] + dt * pr.reshape(-1); cp[t + 1] = to_numpy(xt[:, 3])
    return np.arange(T + 1) * dt, ctru, cp


def per_met(ctru, cp):
    v = [np.corrcoef(ctru[:, i], cp[:, i])[0, 1] for i in range(ctru.shape[1])
         if np.std(ctru[:, i]) > 1e-6 and np.std(cp[:, i]) > 1e-6 and np.isfinite(cp[:, i]).all()]
    return float(np.mean(v)) if v else float("nan")


def main():
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
    SEP = 4.5
    for j, (cfg, lab) in enumerate(MODELS):
        config = MetabolismGraphConfig.from_yaml(f"config/{cfg}.yaml")
        ds = config.dataset; dt = config.simulation.delta_t
        x = np.load(f"graphs_data/{ds}/x_list_0.npy")
        sp = f"graphs_data/{ds}/stimulus.npy"
        stim = torch.tensor(np.load(sp), dtype=torch.float32) if os.path.exists(sp) else None
        m = lsq_vmax_model(config, ds, stim=stim)
        tg, ctru, cp = rollout(m, x, dt, stimulus=stim)
        pm = per_met(ctru, cp)
        sel = np.argsort(-np.nanvar(ctru, axis=0))[:8]
        for k, i in enumerate(sel):
            mu = np.nanmean(ctru[:, i]); sd = np.nanstd(ctru[:, i]) + 1e-9; off = k * SEP
            ax[j].plot(tg, (ctru[:, i] - mu) / sd + off, color=GT_C, lw=1.2)
            ax[j].plot(tg, np.clip((cp[:, i] - mu) / sd, -0.48 * SEP, 0.48 * SEP) + off,
                       color=PRED_C, lw=0.8)
        ax[j].set_title(lab, fontsize=13, loc="center")
        ax[j].text(0.97, 0.015, f"per-met $r$={pm:.2f}", transform=ax[j].transAxes,
                   va="bottom", ha="right", fontsize=11)
        ax[j].set_xlabel("time"); ax[j].set_yticks([])
        ax[j].spines["left"].set_visible(False)
        ax[j].text(0.0, 1.02, "abc"[j], transform=ax[j].transAxes, fontsize=16,
                   fontweight="bold", va="bottom", ha="left")
        print(f"  {cfg}: per-met r = {pm:.3f}")
    ax[0].plot([], [], color=GT_C, lw=1.5, label="ground truth")
    ax[0].plot([], [], color=PRED_C, lw=1.0, label="least-squares rollout")
    ax[0].legend(loc="upper left", frameon=False)
    ax[0].set_ylabel("$z$-scored conc. (offset)")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures/metabolism", OUTNAME)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
