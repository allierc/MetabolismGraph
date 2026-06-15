#!/usr/bin/env python
"""
Design-matrix identifiability analysis (neurips-style). With S given and the
substrate shape known, recovering the per-reaction scale is LINEAR:
    dc_i/dt = sum_j S_ij * theta_j * phi_j(c(t))      (theta_j = Vmax_j or k_j)
so  A theta = b,  A[(i,t), j] = S_ij * phi_j(c(t)),  b[(i,t)] = dc_i/dt (minus drive).
theta is identifiable iff A is full column rank. We compute the Gram matrix G = A^T A
(n_rxn x n_rxn), its eigenvalue (=singular-value^2) spectrum, the EFFECTIVE rank
(sv/sv_max > 1e-3) and the number of null/sloppy directions. MM uses the GT saturation
phi=prod[c/(Km+c)]^s; mass-action uses phi=prod c^s. This tells us whether MM Vmax is
INTRINSICALLY unidentifiable (rank-deficient A) or whether failure is a learning issue.

Usage: python scripts/design_matrix.py <config>   (e.g. ecoli_core_stim, k_recovery_winner)
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
from MetabolismGraph.config import MetabolismGraphConfig
from MetabolismGraph.utils import to_numpy

EPS = 1e-8


def main():
    cfg_name = sys.argv[1]
    config = MetabolismGraphConfig.from_yaml(f"config/{cfg_name}.yaml")
    ds = config.dataset
    sg = torch.load(f"graphs_data/{ds}/stoich_graph.pt", map_location="cpu")
    gt = torch.load(f"graphs_data/{ds}/gt_model.pt", map_location="cpu")
    x = np.load(f"graphs_data/{ds}/x_list_0.npy")          # (T, n_met, 8)
    T = min(2000, x.shape[0]); c = x[:T, :, 3]             # (T, n_met)
    (met_sub, rxn_sub, sto_sub) = [to_numpy(t) for t in sg["sub"]]
    n_rxn = config.simulation.n_reactions
    n_met = config.simulation.n_metabolites
    is_mm = "MichaelisMenten" in config.graph_model.model_name
    km = (10.0 ** to_numpy(gt["log_km"])) if (is_mm and "log_km" in gt) else None

    # phi_j(c(t)) = prod over substrate edges of saturation^s (MM) or c^s (mass-action)
    sub_idx = met_sub.astype(int); sub_rxn = rxn_sub.astype(int); sub_s = sto_sub
    cc = np.clip(c[:, sub_idx], EPS, None)                 # (T, n_sub_edges)
    if is_mm:
        kk = km if km is not None else 1.0
        edge_term = (cc / (kk + cc)) ** sub_s              # saturation^s per edge
    else:
        edge_term = cc ** sub_s
    log_edge = np.log(np.clip(edge_term, EPS, None))
    log_phi = np.zeros((T, n_rxn))
    for e in range(len(sub_rxn)):
        log_phi[:, sub_rxn[e]] += log_edge[:, e]
    phi = np.exp(log_phi)                                  # (T, n_rxn)

    # A[(i,t), j] = S_ij * phi_j(t). Build Gram G = A^T A (n_rxn x n_rxn) cheaply:
    # G_jk = sum_i S_ij S_ik * sum_t phi_j(t) phi_k(t).  StS = sum_i S_ij S_ik.
    S = np.zeros((n_met, n_rxn))
    (met_all, rxn_all, sto_all) = [to_numpy(t) for t in sg["all"]]
    for e in range(len(rxn_all)):
        S[int(met_all[e]), int(rxn_all[e])] = sto_all[e]
    StS = S.T @ S                                          # (n_rxn, n_rxn) topology overlap
    PtP = phi.T @ phi                                      # (n_rxn, n_rxn) temporal overlap
    G = StS * PtP                                          # Hadamard = A^T A
    # normalise columns (so the rank reflects directions, not scale)
    d = np.sqrt(np.clip(np.diag(G), EPS, None))
    Gn = G / np.outer(d, d)
    ev = np.clip(np.linalg.eigvalsh(Gn), 0, None)[::-1]
    sv = np.sqrt(ev)                                       # singular values of normalised A
    smax = sv[0]
    eff_rank = int(np.sum(sv / smax > 1e-3))
    null = int(np.sum(sv / smax < 1e-6))
    sloppy = int(np.sum(sv / smax < 1e-3))
    print(f"=== {cfg_name} ({'MM' if is_mm else 'mass-action'}): {n_rxn} reactions ===")
    print(f"  design-matrix A^T A (normalised): effective rank (sv/smax>1e-3) = {eff_rank}/{n_rxn}")
    print(f"  null directions (sv/smax<1e-6) = {null}; sloppy (<1e-3) = {sloppy}")
    print(f"  condition number sv_max/sv_min = {smax/max(sv[-1],1e-30):.2e}")
    print(f"  smallest 5 sv/smax: {np.round(sv[-5:]/smax, 6)}")

    # ---- DECISIVE: least-squares solve theta from A theta = b (b = dc/dt minus drive) ----
    # Build the full A (n_met*T x n_rxn) and b, solve, compare log10(theta_hat) to GT.
    ypath = f"graphs_data/{ds}/y_list_0.npy"
    if os.path.exists(ypath):
        y = np.load(ypath)[:T, :, 0] if np.load(ypath).ndim == 3 else np.load(ypath)[:T]
    else:
        dt = config.simulation.delta_t
        y = (c[1:] - c[:-1]) / dt; phi = phi[:-1]; c = c[:-1]; T = T - 1
    drive = x[:T, :, 4]                                   # additive source (0 where none)
    b = (y[:T] - drive).reshape(-1)                       # (n_met*T,)  reaction-only dc/dt
    A = np.zeros((n_met * T, n_rxn))
    for e in range(len(rxn_all)):
        j = int(rxn_all[e]); i = int(met_all[e])
        A[i::n_met, j] += sto_all[e] * phi[:T, j]         # S_ij * phi_j(t) stacked over (i,t)
    theta_hat, *_ = np.linalg.lstsq(A, b, rcond=None)
    theta_gt = 10.0 ** to_numpy(gt["log_k"])
    keep = theta_hat > 0
    lr = np.log10(np.clip(theta_hat[keep], 1e-12, None)); lg = np.log10(theta_gt[keep])
    if lr.size > 2 and np.std(lr) > 0:
        r2 = 1 - np.sum((lr - lg) ** 2) / np.sum((lg - lg.mean()) ** 2)
        rr = np.corrcoef(lr, lg)[0, 1]
    else:
        r2 = rr = float("nan")
    print(f"  >>> LEAST-SQUARES Vmax recovery (GT shape given): R^2={r2:.3f} corr={rr:.3f} "
          f"({keep.sum()}/{n_rxn} positive)")
    return eff_rank, n_rxn, sv / smax


if __name__ == "__main__":
    main()
