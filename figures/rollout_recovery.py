#!/usr/bin/env python
"""
Rollout vs recovery (the learnability point). Free-running rollout of the driven
E. coli core network under two parameterisations, both vs ground truth (green):
  (a) the GNN-learned kinetics  -> wrong Vmax, mediocre rollout (per-met ~0.6);
  (b) least-squares Vmax given the true shape -> exact Vmax, PERFECT rollout (1.0).
A single correct solution nails parameters AND rollout; the GNN finds neither ->
the barrier is shape x scale learnability, not identifiability.

Output: figures/metabolism/rollout_recovery.png
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
from k_recovery import load_model
from MetabolismGraph.config import MetabolismGraphConfig
from MetabolismGraph.generators.PDE_MichaelisMenten import PDE_MichaelisMenten
from MetabolismGraph.utils import to_numpy

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11,
                     "legend.fontsize": 10})
GT_C, PRED_C = "#2ca02c", "k"
CFG = "ecoli_core_stim"


def rollout(model, x, dt, stimulus=None):
    T = min(2000, x.shape[0] - 1)
    ctru = x[:T + 1, :, 3]; cp = np.zeros_like(ctru); cp[0] = x[0, :, 3]
    xt = torch.tensor(x[0], dtype=torch.float32)
    with torch.no_grad():
        for t in range(T):
            xt[:, 4] = torch.tensor(x[t, :, 4], dtype=torch.float32)
            s_ = stimulus[t] if stimulus is not None else None
            pr = model(pyg_Data(x=xt.clone(), pos=xt[:, 1:3]), stimulus=s_, dt=dt) \
                if "dt" in model.forward.__code__.co_varnames else \
                model(pyg_Data(x=xt.clone(), pos=xt[:, 1:3]), stimulus=s_)
            xt[:, 3] = xt[:, 3] + dt * pr.reshape(-1); cp[t + 1] = to_numpy(xt[:, 3])
    return np.arange(T + 1) * dt, ctru, cp


def per_met(ctru, cp):
    v = [np.corrcoef(ctru[:, i], cp[:, i])[0, 1] for i in range(ctru.shape[1])
         if np.std(ctru[:, i]) > 1e-6 and np.std(cp[:, i]) > 1e-6 and np.isfinite(cp[:, i]).all()]
    return float(np.mean(v)) if v else float("nan")


def lsq_vmax_model(config, ds):
    sg = torch.load(f"graphs_data/{ds}/stoich_graph.pt", map_location="cpu")
    torch.manual_seed(0); m = PDE_MichaelisMenten(config=config, stoich_graph=sg, device="cpu")
    m.load_state_dict(torch.load(f"graphs_data/{ds}/gt_model.pt", map_location="cpu"), strict=False); m.eval()
    x = np.load(f"graphs_data/{ds}/x_list_0.npy"); dt = config.simulation.delta_t
    nrxn = config.simulation.n_reactions; nmet = config.simulation.n_metabolites
    lk0 = m.log_k.detach().clone(); Vgt = to_numpy(10 ** lk0); eps = 1e-3
    f = lambda xt: to_numpy(m(pyg_Data(x=torch.tensor(xt, dtype=torch.float32),
                              pos=torch.tensor(xt[:, 1:3], dtype=torch.float32)), dt=dt).reshape(-1))
    Ts = np.linspace(0, x.shape[0] - 2, 120, dtype=int); Ab, bb = [], []
    for t in Ts:
        xt = x[t].copy(); d0 = f(xt)
        with torch.no_grad(): m.log_k.copy_(torch.full_like(lk0, -30.)); g = f(xt); m.log_k.copy_(lk0)
        bb.append(d0 - g); At = np.zeros((nmet, nrxn))
        for j in range(nrxn):
            with torch.no_grad(): m.log_k[j] += eps; dj = f(xt); m.log_k.copy_(lk0)
            At[:, j] = (dj - d0) / (Vgt[j] * (10 ** eps - 1))
        Ab.append(At)
    Vhat, *_ = np.linalg.lstsq(np.vstack(Ab), np.concatenate(bb), rcond=None)
    with torch.no_grad(): m.log_k.copy_(torch.tensor(np.log10(np.clip(Vhat, 1e-9, None)), dtype=torch.float32))
    return m


def main():
    config = MetabolismGraphConfig.from_yaml(f"config/{CFG}.yaml"); ds = config.dataset
    dt = config.simulation.delta_t
    x = np.load(f"graphs_data/{ds}/x_list_0.npy")
    # (a) GNN-learned model
    _, gnn, _, _, _ = load_model(CFG, "cpu")
    tg, ctru, cp_gnn = rollout(gnn, x, dt)
    pm_gnn = per_met(ctru, cp_gnn)
    # (b) least-squares Vmax model (true shape)
    lsq = lsq_vmax_model(config, ds)
    _, _, cp_lsq = rollout(lsq, x, dt)
    pm_lsq = per_met(ctru, cp_lsq)

    sel = np.argsort(-np.nanvar(ctru, axis=0))[:8]; SEP = 4.5
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for j, (cp, lab, pm) in enumerate([(cp_gnn, "GNN-learned kinetics", pm_gnn),
                                       (cp_lsq, "least-squares $V_{\\max}$ (true shape)", pm_lsq)]):
        for k, i in enumerate(sel):
            mu = np.nanmean(ctru[:, i]); sd = np.nanstd(ctru[:, i]) + 1e-9; off = k * SEP
            ax[j].plot(tg, (ctru[:, i] - mu) / sd + off, color=GT_C, lw=1.3)
            ax[j].plot(tg, np.clip((cp[:, i] - mu) / sd, -0.48 * SEP, 0.48 * SEP) + off,
                       color=PRED_C, lw=0.9)
        ax[j].set_title(lab, fontsize=13, loc="center")
        ax[j].text(0.97, 0.015, f"per-met $r$={pm:.2f}", transform=ax[j].transAxes,
                   va="bottom", ha="right", fontsize=12)
        ax[j].set_xlabel("time"); ax[j].set_yticks([]); ax[j].spines["left"].set_visible(False)
        ax[j].text(0.0, 1.02, "ab"[j], transform=ax[j].transAxes, fontsize=16,
                   fontweight="bold", va="bottom", ha="left")
    ax[0].plot([], [], color=GT_C, lw=1.3, label="ground truth")
    ax[0].plot([], [], color=PRED_C, lw=0.9, label="rollout")
    ax[0].legend(loc="upper left", frameon=False); ax[0].set_ylabel("$z$-scored conc. (offset)")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures/metabolism/rollout_recovery.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}: GNN per-met={pm_gnn:.3f}, LSQ per-met={pm_lsq:.3f}")


if __name__ == "__main__":
    main()
