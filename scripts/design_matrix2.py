#!/usr/bin/env python
"""
CORRECT design-matrix identifiability analysis, built from the GT model's ACTUAL
forward (finite-difference Jacobian wrt Vmax), so flux-limiting / homeostasis / drive
are all captured exactly (the hand-reimplementation in design_matrix.py was buggy).

dc/dt = A(c) @ Vmax + g(c),  A_ij = d(dc/dt_i)/d(Vmax_j)  [exact via FD on the model],
g = dc/dt at Vmax=0 (homeostasis + drive). b = dc/dt - g (reaction part). Then:
  - residual ||A Vmax_GT - b||  (must be ~0 to trust the analysis)
  - SVD(A) -> identifiability rank + null directions
  - LSQ solve theta -> Vmax recovery R^2 (the achievable upper bound for ANY method)

Usage: python scripts/design_matrix2.py <config>
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
from torch_geometric.data import Data
from MetabolismGraph.config import MetabolismGraphConfig
from MetabolismGraph.generators.PDE_MichaelisMenten import PDE_MichaelisMenten
from MetabolismGraph.generators.PDE_M1 import PDE_M1
from MetabolismGraph.utils import to_numpy


def main():
    cfg = sys.argv[1]
    config = MetabolismGraphConfig.from_yaml(f"config/{cfg}.yaml")
    ds = config.dataset
    sg = torch.load(f"graphs_data/{ds}/stoich_graph.pt", map_location="cpu")
    is_mm = "MichaelisMenten" in config.graph_model.model_name
    Model = PDE_MichaelisMenten if is_mm else PDE_M1
    torch.manual_seed(0)
    m = Model(config=config, stoich_graph=sg, device="cpu")
    m.load_state_dict(torch.load(f"graphs_data/{ds}/gt_model.pt", map_location="cpu"), strict=False)
    m.eval()
    n_rxn = config.simulation.n_reactions; n_met = config.simulation.n_metabolites
    dt = config.simulation.delta_t
    x = np.load(f"graphs_data/{ds}/x_list_0.npy")
    Ts = np.linspace(0, x.shape[0] - 2, 120, dtype=int)        # subsample 120 frames
    logk0 = m.log_k.detach().clone()
    Vmax_gt = to_numpy(10.0 ** logk0)
    eps = 1e-3

    def dcdt(xt):
        return to_numpy(m(Data(x=torch.tensor(xt, dtype=torch.float32),
                               pos=torch.tensor(xt[:, 1:3], dtype=torch.float32)), dt=dt).reshape(-1))

    A_blocks, b_blocks = [], []
    for t in Ts:
        xt = x[t].copy()
        d0 = dcdt(xt)
        with torch.no_grad():
            m.log_k.copy_(torch.full_like(logk0, -30.0))       # Vmax -> 0
            g = dcdt(xt)                                        # homeostasis + drive
            m.log_k.copy_(logk0)
        b_blocks.append(d0 - g)
        A_t = np.zeros((n_met, n_rxn))
        for j in range(n_rxn):
            with torch.no_grad():
                m.log_k[j] += eps                              # Vmax_j *= 10^eps
                dj = dcdt(xt)
                m.log_k.copy_(logk0)
            A_t[:, j] = (dj - d0) / (Vmax_gt[j] * (10.0 ** eps - 1.0))
        A_blocks.append(A_t)
    A = np.vstack(A_blocks); b = np.concatenate(b_blocks)
    resid = np.linalg.norm(A @ Vmax_gt - b) / (np.linalg.norm(b) + 1e-12)

    # SVD / rank (column-normalised) + LSQ recovery
    dcol = np.linalg.norm(A, axis=0); An = A / np.clip(dcol, 1e-12, None)
    sv = np.linalg.svd(An, compute_uv=False)
    eff = int(np.sum(sv / sv[0] > 1e-3)); nullc = int(np.sum(sv / sv[0] < 1e-6))
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    keep = theta > 0
    lr = np.log10(np.clip(theta[keep], 1e-12, None)); lg = np.log10(Vmax_gt[keep])
    r2 = 1 - np.sum((lr - lg) ** 2) / np.sum((lg - lg.mean()) ** 2) if keep.sum() > 2 else float("nan")
    corr = np.corrcoef(lr, lg)[0, 1] if keep.sum() > 2 else float("nan")
    print(f"=== {cfg} ({'MM' if is_mm else 'mass-action'}): {n_rxn} reactions, {len(Ts)} frames ===")
    print(f"  reconstruction residual ||A Vmax_GT - b||/||b|| = {resid:.4f}  (trust analysis iff ~0)")
    print(f"  design-matrix rank: effective {eff}/{n_rxn}, null dirs {nullc}, cond {sv[0]/max(sv[-1],1e-30):.2e}")
    print(f"  >>> LSQ Vmax recovery (exact A, GT shape): R2={r2:.3f} corr={corr:.3f} ({keep.sum()}/{n_rxn} positive)")


if __name__ == "__main__":
    main()
