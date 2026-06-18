#!/usr/bin/env python
"""
Phase 1b — Levenberg-Marquardt joint solve of (Vmax, Km) on the exact MM family. Target is the
GT model's reaction-only dc/dt (homeostasis + flux-limit off) over a subset of frames.

FINDING (Phase 1): the joint problem is rank-identifiable but the Km block is practically
ill-conditioned (cond 3.8e3-6.9e4) -- Km of saturated edges (c>>Km) is weakly constrained, and
this UNREGULARIZED LM diverges on those directions (Km R2 -> large negative) even though Vmax is
recoverable. TODO (Phase 2): add a trust-region step bound + Tikhonov/SVD-truncation so the solve
recovers the identifiable subspace (full Vmax + well-excited Km) -- the honest achievable ceiling.

Usage: python scripts/joint_gn_solve.py <config>
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
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "ecoli_core_hybrid"
    config = MetabolismGraphConfig.from_yaml(f"config/{cfg_name}.yaml")
    ds = config.dataset
    sg = torch.load(f"graphs_data/{ds}/stoich_graph.pt", map_location="cpu")
    m = PDE_MichaelisMenten(config=config, stoich_graph=sg, device="cpu")
    m.load_state_dict(torch.load(f"graphs_data/{ds}/gt_model.pt", map_location="cpu"), strict=False)
    m.eval(); m.flux_limit_enabled = False
    with torch.no_grad():
        m.p.zero_(); m.homeostatic_strength = 0.0

    nk = m.log_k.shape[0]; nm = m.log_km.shape[0]; npar = nk + nm
    x = np.load(f"graphs_data/{ds}/x_list_0.npy")
    Ts = np.linspace(0, x.shape[0] - 2, 80, dtype=int)
    lk_gt = m.log_k.detach().clone(); lm_gt = m.log_km.detach().clone()
    eps = 1e-3

    def dcdt_all(xt):
        return to_numpy(m(Data(x=torch.tensor(xt, dtype=torch.float32),
                               pos=torch.tensor(xt[:, 1:3], dtype=torch.float32)),
                          dt=None).reshape(-1))

    # target = GT reaction dc/dt
    with torch.no_grad():
        m.log_k.copy_(lk_gt); m.log_km.copy_(lm_gt)
    target = np.concatenate([dcdt_all(x[t].copy()) for t in Ts])

    # init: arg2 = perturbation scale around GT (0 -> near-GT, validates solver;
    # large/'generic' -> Km=1 + random Vmax, tests the global basin)
    init_mode = sys.argv[2] if len(sys.argv) > 2 else "generic"
    rng = np.random.default_rng(0)
    with torch.no_grad():
        if init_mode == "generic":
            m.log_k.copy_(torch.tensor(rng.uniform(float(lk_gt.min()), float(lk_gt.max()), nk),
                                       dtype=torch.float32))
            m.log_km.copy_(torch.zeros(nm))
        else:
            s = float(init_mode)
            m.log_k.copy_(lk_gt + torch.tensor(rng.standard_normal(nk) * s, dtype=torch.float32))
            m.log_km.copy_(lm_gt + torch.tensor(rng.standard_normal(nm) * s, dtype=torch.float32))

    def residual_and_jac():
        pred = np.concatenate([dcdt_all(x[t].copy()) for t in Ts])
        r = pred - target
        J = np.zeros((len(target), npar))
        lk = m.log_k.detach().clone(); lm = m.log_km.detach().clone()
        for j in range(nk):
            with torch.no_grad(): m.log_k[j] += eps
            pj = np.concatenate([dcdt_all(x[t].copy()) for t in Ts])
            with torch.no_grad(): m.log_k.copy_(lk)
            J[:, j] = (pj - pred) / eps
        for e in range(nm):
            with torch.no_grad(): m.log_km[e] += eps
            pe = np.concatenate([dcdt_all(x[t].copy()) for t in Ts])
            with torch.no_grad(): m.log_km.copy_(lm)
            J[:, nk + e] = (pe - pred) / eps
        return r, J

    def cost_now():
        pred = np.concatenate([dcdt_all(x[t].copy()) for t in Ts])
        return float(np.mean((pred - target) ** 2))

    lam = 1e-2
    print(f"=== {cfg_name} ({ds}): Levenberg-Marquardt joint (Vmax,Km) solve, init='{init_mode}' ===")
    for it in range(40):
        r, J = residual_and_jac()
        cost = float(np.mean(r ** 2))
        vmax_r2 = r2(to_numpy(m.log_k), to_numpy(lk_gt))
        km_r2 = r2(to_numpy(m.log_km), to_numpy(lm_gt))
        if it % 4 == 0 or it < 3:
            print(f"  iter {it:2d}: cost={cost:.3e}  lam={lam:.1e}  Vmax R2={vmax_r2:+.3f}  Km R2={km_r2:+.3f}")
        if cost < 1e-12:
            break
        JtJ = J.T @ J; g = J.T @ r
        lk_b = m.log_k.detach().clone(); lm_b = m.log_km.detach().clone()
        # adaptive LM: try a step, accept iff it reduces cost, else increase damping
        for _try in range(8):
            step = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-9), -g)
            with torch.no_grad():
                m.log_k.copy_(lk_b + torch.tensor(step[:nk], dtype=torch.float32))
                m.log_km.copy_(lm_b + torch.tensor(step[nk:], dtype=torch.float32))
            if cost_now() < cost:
                lam = max(lam * 0.5, 1e-9); break
            lam = min(lam * 4.0, 1e6)
            with torch.no_grad():
                m.log_k.copy_(lk_b); m.log_km.copy_(lm_b)
        else:
            print(f"  (no decrease at iter {it}; stuck in local basin)"); break
    vmax_r2 = r2(to_numpy(m.log_k), to_numpy(lk_gt)); km_r2 = r2(to_numpy(m.log_km), to_numpy(lm_gt))
    print(f"  FINAL: Vmax R2={vmax_r2:+.4f}  Km R2={km_r2:+.4f}")
    with open("docs/hybrid_overnight_results.md", "a") as f:
        f.write(f"- **GN-solve {cfg_name}** | Gauss-Newton from generic init -> "
                f"Vmax R2={vmax_r2:.3f}, Km R2={km_r2:.3f} (achievable ceiling on the exact family)\n")


if __name__ == "__main__":
    main()
