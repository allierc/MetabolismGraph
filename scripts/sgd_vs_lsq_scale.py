#!/usr/bin/env python
"""
Decisive diagnostic: given the EXACT MM shape (Km=GT, S=GT frozen, NO homeostasis confound),
can plain SGD/Adam recover the scale (Vmax = log_k)? Compares:
  - LSQ solve (closed form, the design_matrix2 result): Vmax R^2 ~ 1.0
  - Adam on log_k ONLY, minimizing ||reaction dc/dt - GT reaction dc/dt||^2

If Adam also -> ~1, the oracle GNN's failure (R^2~0.1) is the co-trained homeostasis/regularization
confound. If Adam stays ~0.1, SGD on the prediction loss is fundamentally gradient-starved
(low-flux reactions get ~no gradient) -> the fix is a linear solve, not end-to-end learning.

Usage: python scripts/sgd_vs_lsq_scale.py <config>   (e.g. ecoli_core_hybrid_oracle)
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
from torch_geometric.data import Data
from MetabolismGraph.config import MetabolismGraphConfig
from MetabolismGraph.models.Metabolism_Propagation import Metabolism_Propagation
from MetabolismGraph.generators.PDE_MichaelisMenten import PDE_MichaelisMenten
from MetabolismGraph.utils import to_numpy


def r2(a, b):
    a = to_numpy(a); b = to_numpy(b)
    return float(1 - np.sum((a - b) ** 2) / (np.sum((b - b.mean()) ** 2) + 1e-12))


def main():
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "ecoli_core_hybrid_oracle"
    dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    # 'none'  : reaction only, learn log_k (isolates the scale)
    # 'homeo' : reaction + GT homeostasis target, co-train generic MLP_node (the full-GNN confound)
    # 'homeo_struct': same target, but homeostasis is STRUCTURED -lambda_i(c_i-b_i), co-trained
    mode = sys.argv[3] if len(sys.argv) > 3 else "none"
    config = MetabolismGraphConfig.from_yaml(f"config/{cfg_name}.yaml")
    config.graph_model.substrate_func_type = "mm"
    ds = config.dataset
    sg = torch.load(f"graphs_data/{ds}/stoich_graph.pt", map_location=dev)

    gt = PDE_MichaelisMenten(config=config, stoich_graph=sg, device=dev)
    gt.load_state_dict(torch.load(f"graphs_data/{ds}/gt_model.pt", map_location=dev), strict=False)
    gt = gt.to(dev); gt.eval(); gt.flux_limit_enabled = False
    if mode == "none":                          # isolate the reaction term
        with torch.no_grad(): gt.p.zero_(); gt.homeostatic_strength = 0.0
    gt.external_input_mode = "none"
    lk_gt = gt.log_k.detach().clone()

    m = Metabolism_Propagation(config=config, device=dev); m.load_stoich_graph(sg); m = m.to(dev)
    with torch.no_grad():
        m.sto_all.data.copy_(sg['all'][2])                  # S = GT, frozen
        m.log_km.data.copy_(gt.log_km.detach())             # Km = GT, frozen
        for mod in m.node_func.modules():                   # homeostasis OFF (re-zeroed below)
            if isinstance(mod, torch.nn.Linear): mod.weight.zero_(); mod.bias.zero_()
    m.sto_all.requires_grad = False; m.log_km.requires_grad = False
    m.external_input_mode = "none"
    # generic homeostasis (MLP_node) co-trains only in 'homeo'; frozen-off otherwise
    train_homeo = (mode == "homeo")
    for p in m.node_func.parameters(): p.requires_grad = train_homeo
    m.a.requires_grad = train_homeo
    struct = (mode == "homeo_struct")   # structured homeostasis params created after X is built
    # random init for log_k (the thing we recover)
    rng = np.random.default_rng(0)
    with torch.no_grad():
        m.log_k.data.copy_(torch.tensor(rng.uniform(-2, 1, m.log_k.shape[0]), dtype=torch.float32))

    x = np.load(f"graphs_data/{ds}/x_list_0.npy")
    Ts = np.linspace(0, x.shape[0] - 2, 60, dtype=int)
    X = [torch.tensor(x[t], dtype=torch.float32, device=dev) for t in Ts]
    with torch.no_grad():
        target = torch.stack([gt(Data(x=xt, pos=xt[:, 1:3]), dt=None).reshape(-1) for xt in X])

    if struct:
        log_lam = torch.zeros(config.simulation.n_metabolites, device=dev, requires_grad=True)
        base = X[0][:, 3].clone().detach().requires_grad_(True)   # init baseline = c(t=0)
    params = [m.log_k]
    if train_homeo: params += [*m.node_func.parameters(), m.a]
    if struct: params += [log_lam, base]
    opt = torch.optim.Adam(params, lr=0.03)
    htag = {"none": "OFF", "homeo": "generic MLP_node (co-trained)",
            "homeo_struct": "STRUCTURED -lam(c-b) (co-trained)"}[mode]
    print(f"=== {cfg_name} ({ds}): Adam on log_k, exact Km+S, homeostasis={htag} ===")
    print(f"  LSQ ceiling (design_matrix2): Vmax R2 ~ 1.000")
    for it in range(4001):
        opt.zero_grad()
        pred = torch.stack([m(Data(x=xt, pos=xt[:, 1:3])).reshape(-1) for xt in X])
        if struct:
            homeo = torch.stack([-torch.nn.functional.softplus(log_lam) * (xt[:, 3] - base) for xt in X])
            pred = pred + homeo
        loss = ((pred - target) ** 2).mean()
        loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"  iter {it:4d}: loss={loss.item():.3e}  Vmax R2={r2(m.log_k, lk_gt):+.4f}")
    final = r2(m.log_k, lk_gt)
    print(f"  FINAL Adam Vmax R2={final:+.4f}")
    with open("docs/hybrid_overnight_results.md", "a") as f:
        f.write(f"- **SGD-vs-LSQ scale {cfg_name}** | Adam on log_k only (exact Km+S, no homeostasis) "
                f"-> Vmax R2={final:.3f}  vs  LSQ ceiling ~1.000. "
                f"{'SGD gradient-starved' if final < 0.7 else 'SGD recovers when isolated'}.\n")


if __name__ == "__main__":
    main()
