#!/usr/bin/env python
"""
Decisive Phase-2 test: with the EXACT MM shape (GT Km) AND a structured homeostasis
-lambda_i(c_i - b_i), the full dc/dt is LINEAR in (Vmax_j, lambda_i, mu_i:=lambda_i b_i):
  dc/dt_i = sum_j S_ij Vmax_j phi_j(c)  - lambda_i c_i + mu_i
So a single least-squares solve recovers everything IFF the joint design matrix is full rank.
This separates "reaction vs homeostasis is genuinely degenerate" (LSQ Vmax R2 < 1) from
"SGD is just slow / ill-conditioned" (LSQ Vmax R2 = 1).

Columns: reaction (finite-diff d/dVmax_j on GT, n_rxn) | lambda_i (-c_i(t), n_met) | mu_i (+1, n_met).
Target: full GT dc/dt (reaction + homeostasis, flux-limit off).

Usage: python scripts/joint_lsq_homeo.py <config>
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
from torch_geometric.data import Data
from MetabolismGraph.config import MetabolismGraphConfig
from MetabolismGraph.generators.PDE_MichaelisMenten import PDE_MichaelisMenten
from MetabolismGraph.utils import to_numpy


def r2(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(1 - np.sum((a - b) ** 2) / (np.sum((b - b.mean()) ** 2) + 1e-12))


def main():
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "ecoli_core_hybrid_oracle"
    config = MetabolismGraphConfig.from_yaml(f"config/{cfg_name}.yaml")
    ds = config.dataset
    sg = torch.load(f"graphs_data/{ds}/stoich_graph.pt", map_location="cpu")
    m = PDE_MichaelisMenten(config=config, stoich_graph=sg, device="cpu")
    m.load_state_dict(torch.load(f"graphs_data/{ds}/gt_model.pt", map_location="cpu"), strict=False)
    m.eval(); m.flux_limit_enabled = False; m.external_input_mode = "none"
    nrxn = config.simulation.n_reactions; nmet = config.simulation.n_metabolites
    lam_gt = to_numpy(m.p[:, 0]).copy()                    # GT homeostatic lambda (per type, here global)
    lk0 = m.log_k.detach().clone(); Vgt = to_numpy(10.0 ** lk0)

    x = np.load(f"graphs_data/{ds}/x_list_0.npy")
    Ts = np.linspace(0, x.shape[0] - 2, 150, dtype=int)
    eps = 1e-3

    def full_dcdt(xt):    # reaction + homeostasis (GT), flux off
        return to_numpy(m(Data(x=torch.tensor(xt, dtype=torch.float32),
                               pos=torch.tensor(xt[:, 1:3], dtype=torch.float32)), dt=None).reshape(-1))

    def react_only(xt):
        sav = m.homeostatic_strength; psav = m.p.detach().clone()
        with torch.no_grad(): m.p.zero_(); m.homeostatic_strength = 0.0
        d = to_numpy(m(Data(x=torch.tensor(xt, dtype=torch.float32),
                            pos=torch.tensor(xt[:, 1:3], dtype=torch.float32)), dt=None).reshape(-1))
        with torch.no_grad(): m.p.copy_(psav); m.homeostatic_strength = sav
        return d

    A_blocks, b_blocks = [], []
    for t in Ts:
        xt = x[t].copy(); ci = xt[:, 3]
        b_blocks.append(full_dcdt(xt))
        At = np.zeros((nmet, nrxn + 2 * nmet))
        # reaction columns: d(dc/dt)/dVmax_j  (finite diff on log_k, reaction-only)
        d0 = react_only(xt)
        for j in range(nrxn):
            with torch.no_grad(): m.log_k[j] += eps; dj = react_only(xt); m.log_k.copy_(lk0)
            At[:, j] = (dj - d0) / (Vgt[j] * (10.0 ** eps - 1.0))
        # homeostasis columns: lambda_i -> -c_i(t) on row i ; mu_i -> +1 on row i
        for i in range(nmet):
            At[i, nrxn + i] = -ci[i]
            At[i, nrxn + nmet + i] = 1.0
        A_blocks.append(At)
    A = np.vstack(A_blocks); b = np.concatenate(b_blocks)

    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    Vhat = theta[:nrxn]; lam_hat = theta[nrxn:nrxn + nmet]
    resid = np.linalg.norm(A @ theta - b) / (np.linalg.norm(b) + 1e-12)
    keep = Vhat > 0
    r2_V = r2(np.log10(np.clip(Vhat[keep], 1e-12, None)), np.log10(Vgt[keep]))
    r2_lam = r2(lam_hat, np.full(nmet, lam_gt[0]) if lam_gt.size == 1 else lam_gt[:nmet])
    sv = np.linalg.svd(A / np.clip(np.linalg.norm(A, axis=0), 1e-12, None), compute_uv=False)
    cond = sv[0] / max(sv[-1], 1e-30)
    print(f"=== {cfg_name} ({ds}): joint LSQ (Vmax, homeostasis) given exact shape ===")
    print(f"  params {nrxn} Vmax + {nmet} lambda + {nmet} mu = {nrxn+2*nmet}; eqs {A.shape[0]}")
    print(f"  reconstruction residual = {resid:.4e}  | design cond = {cond:.2e}")
    print(f"  >>> LSQ Vmax R2 = {r2_V:.4f}   lambda R2 = {r2_lam:.4f}")
    msg = ("reaction<->homeostasis SEPARABLE by LSQ -> SGD was just ill-conditioned/slow"
           if r2_V > 0.9 else
           "reaction<->homeostasis DEGENERATE even for LSQ -> need stimulus/regularization")
    print(f"  >>> {msg}")
    with open("docs/hybrid_overnight_results.md", "a") as f:
        f.write(f"- **joint-LSQ+homeo {cfg_name}** | Vmax R2={r2_V:.3f}, lambda R2={r2_lam:.3f}, "
                f"cond {cond:.1e}, resid {resid:.1e} -> {msg}\n")


if __name__ == "__main__":
    main()
