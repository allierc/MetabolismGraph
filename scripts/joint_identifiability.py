#!/usr/bin/env python
"""
Phase 1 — joint (Vmax, Km) identifiability of the structured-MM inverse problem.

The hybrid model fits the EXACT family v = Vmax * prod [c/(Km+c)]^|s|, with per-reaction
log_k (=log Vmax) and per-edge log_km. Question: is the joint problem (recover BOTH) locally
well-posed, or is there a Vmax<->Km degeneracy that no optimizer can resolve from data?

Built from the GT model's ACTUAL forward (finite-difference Jacobian, like design_matrix2):
  A = [ d(dc/dt)/d(log_k) | d(dc/dt)/d(log_km) ]   stacked over T frames
  - reconstruction residual ||A dtheta - db|| for a small known perturbation (must be ~0 to trust)
  - SVD(A): joint rank, condition number, null directions (the unidentifiable param combos)
  - linearized recovery R^2 for log_k, log_km, and jointly (the achievable ceiling for ANY method)

Usage: python scripts/joint_identifiability.py <config>   (e.g. ecoli_core_hybrid)
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


def main():
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "ecoli_core_hybrid"
    config = MetabolismGraphConfig.from_yaml(f"config/{cfg_name}.yaml")
    ds = config.dataset
    sg = torch.load(f"graphs_data/{ds}/stoich_graph.pt", map_location="cpu")
    m = PDE_MichaelisMenten(config=config, stoich_graph=sg, device="cpu")
    m.load_state_dict(torch.load(f"graphs_data/{ds}/gt_model.pt", map_location="cpu"), strict=False)
    m.eval()
    # isolate the reaction term (the part the hybrid learns): no flux clamp, no homeostasis
    m.flux_limit_enabled = False
    with torch.no_grad():
        m.p.zero_(); m.homeostatic_strength = 0.0

    nk = m.log_k.shape[0]; nm = m.log_km.shape[0]
    n_met = config.simulation.n_metabolites
    dt = config.simulation.delta_t
    x = np.load(f"graphs_data/{ds}/x_list_0.npy")
    Ts = np.linspace(0, x.shape[0] - 2, 120, dtype=int)
    lk0 = m.log_k.detach().clone(); lm0 = m.log_km.detach().clone()
    eps = 1e-3

    def dcdt(xt):
        return to_numpy(m(Data(x=torch.tensor(xt, dtype=torch.float32),
                               pos=torch.tensor(xt[:, 1:3], dtype=torch.float32)),
                          dt=None, stimulus=None).reshape(-1))

    A_blocks = []
    for t in Ts:
        xt = x[t].copy(); d0 = dcdt(xt)
        cols = np.zeros((n_met, nk + nm))
        for j in range(nk):
            with torch.no_grad(): m.log_k[j] += eps; dj = dcdt(xt); m.log_k.copy_(lk0)
            cols[:, j] = (dj - d0) / eps                          # d(dc/dt)/d(log_k_j)
        for e in range(nm):
            with torch.no_grad(): m.log_km[e] += eps; de = dcdt(xt); m.log_km.copy_(lm0)
            cols[:, nk + e] = (de - d0) / eps                     # d(dc/dt)/d(log_km_e)
        A_blocks.append(cols)
    A = np.vstack(A_blocks)                                       # (n_met*T, nk+nm)

    # --- trust check: small known perturbation, does A dtheta reconstruct db? ---
    rng = np.random.default_rng(0)
    dtheta = rng.standard_normal(nk + nm) * 0.05
    db_lin = A @ dtheta
    db_true = []
    with torch.no_grad():
        m.log_k.copy_(lk0 + torch.tensor(dtheta[:nk], dtype=torch.float32))
        m.log_km.copy_(lm0 + torch.tensor(dtheta[nk:], dtype=torch.float32))
    for t in Ts:
        db_true.append(dcdt(x[t].copy()))
    with torch.no_grad():
        m.log_k.copy_(lk0); m.log_km.copy_(lm0)
    db_true = np.concatenate(db_true) - np.concatenate([dcdt(x[t].copy()) for t in Ts])
    resid = np.linalg.norm(db_lin - db_true) / (np.linalg.norm(db_true) + 1e-12)

    # --- SVD / rank / null ---
    dcol = np.linalg.norm(A, axis=0); An = A / np.clip(dcol, 1e-12, None)
    U, sv, Vt = np.linalg.svd(An, full_matrices=False)
    tol = sv[0] * 1e-6
    rank = int((sv > tol).sum()); null = (nk + nm) - rank
    cond = sv[0] / max(sv[-1], 1e-30)

    # --- linearized recovery: solve A dtheta_hat = db_true, compare to dtheta ---
    dtheta_hat, *_ = np.linalg.lstsq(A, db_lin, rcond=None)

    def r2(a, b):
        a = np.asarray(a); b = np.asarray(b)
        return float(1 - np.sum((a - b) ** 2) / (np.sum((b - b.mean()) ** 2) + 1e-12))
    r2_k = r2(dtheta_hat[:nk], dtheta[:nk])
    r2_m = r2(dtheta_hat[nk:], dtheta[nk:])
    r2_all = r2(dtheta_hat, dtheta)

    print(f"=== {cfg_name} ({ds}): joint (Vmax,Km) identifiability ===")
    print(f"  params: {nk} log_k (Vmax) + {nm} log_km (Km) = {nk+nm}; frames={len(Ts)}, eqs={A.shape[0]}")
    print(f"  reconstruction residual ||A dtheta - db||/||db|| = {resid:.4f}  (trust iff ~0)")
    print(f"  joint design-matrix rank: {rank}/{nk+nm}, null dirs {null}, cond {cond:.2e}")
    print(f"  linearized recovery R2:  log_k(Vmax)={r2_k:.3f}  log_km(Km)={r2_m:.3f}  joint={r2_all:.3f}")
    if null > 0:
        # report the dominant null direction's split between Vmax and Km blocks
        v = Vt[-1]; ek = np.linalg.norm(v[:nk]); em = np.linalg.norm(v[nk:])
        print(f"  dominant null direction energy: Vmax-block={ek:.2f}, Km-block={em:.2f} "
              f"(>0 on both => Vmax<->Km degeneracy)")
    # persist a one-line summary
    with open("docs/hybrid_overnight_results.md", "a") as f:
        f.write(f"- **identifiability {cfg_name}** | joint rank {rank}/{nk+nm}, null {null}, "
                f"cond {cond:.1e} | lin recovery R2 Vmax={r2_k:.3f} Km={r2_m:.3f} joint={r2_all:.3f} "
                f"(resid {resid:.3f})\n")


if __name__ == "__main__":
    main()
