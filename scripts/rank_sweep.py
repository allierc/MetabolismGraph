#!/usr/bin/env python
"""
PHASE 1 (stimulus -> activity rank). Closed metabolic networks relax to a steady
state (activity rank ~2), starving the inverse problem. An external metabolite
SOURCE (additive open-system drive in x[:,4]) holds the network off its fixed point.
Linear-algebra view: for the locally linear system dc/dt = J c + B u(t), the
trajectory spans the controllability subspace span{B, JB, J^2 B, ...}; driving more
metabolites with multi-frequency u raises that subspace's rank. Identifiability is
linear in the parameters (A theta = b) and a metabolite that never varies leaves a
null column in A -> unidentifiable. So MORE driven metabolites + richer u => higher
activity rank => more identifiable kinetics.

Sweeps n_input (#driven metabolites) x amplitude, measures activity rank99.
Usage: python scripts/rank_sweep.py <config>   (config has topology_sbml)
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
from torch_geometric.data import Data
from MetabolismGraph.config import MetabolismGraphConfig
from MetabolismGraph.generators.PDE_MichaelisMenten import PDE_MichaelisMenten
from MetabolismGraph.generators.sbml_topology import parse_fba_sbml, central_carbon_subgraph
from MetabolismGraph.generators.utils import init_concentration


def rank99(c):
    cc = c - c.mean(0)
    sv = np.linalg.svd(cc, compute_uv=False)
    if sv.sum() == 0:
        return 0
    return int(np.sum(np.cumsum(sv ** 2) / np.sum(sv ** 2) < 0.99) + 1)


def simulate(model, n_met, T, dt, c0, n_input, amp, seed):
    rng = np.random.RandomState(seed)
    freq = rng.uniform(0.05, 0.5, max(n_input, 1))
    phase = rng.uniform(0, 2 * np.pi, max(n_input, 1))
    x = torch.zeros(n_met, 8, dtype=torch.float32); x[:, 3] = c0.clone()
    model.c_baseline = None
    traj = np.zeros((T + 1, n_met), dtype=np.float32); traj[0] = c0.numpy()
    for t in range(T):
        if n_input > 0:
            drive = amp * np.sin(2 * np.pi * freq[:n_input] * (t * dt) + phase[:n_input])
            x[:n_input, 4] = torch.tensor(drive.astype(np.float32))
        with torch.no_grad():
            d = model(Data(x=x.clone(), pos=x[:, 1:3]), dt=dt)
        x[:, 3] = torch.clamp(x[:, 3] + dt * d.squeeze(), min=0.0)
        traj[t + 1] = x[:, 3].numpy()
    return traj


def main():
    cfg_name = sys.argv[1]
    config = MetabolismGraphConfig.from_yaml(f"config/{cfg_name}.yaml")
    config.simulation.external_input_mode = "additive"
    topo = config.simulation.topology_sbml
    nsub = getattr(config.simulation, "topology_subgraph_reactions", 0)
    if nsub and nsub > 0:
        sg, S, *_ = central_carbon_subgraph(topo, max_reactions=nsub, device="cpu")
    else:
        sg, S, *_ = parse_fba_sbml(topo, device="cpu")
    n_met, n_rxn = S.shape
    config.simulation.n_metabolites = n_met
    config.simulation.n_reactions = n_rxn
    torch.manual_seed(0)
    model = PDE_MichaelisMenten(config=config, stoich_graph=sg, device="cpu")
    c0 = init_concentration(n_met, "cpu", mode="random", seed=0, c_min=0.2, c_max=3.0)
    dt = config.simulation.delta_t
    T = 1500
    print(f"=== {cfg_name}: {n_met} met x {n_rxn} rxn, dt={dt} ===")
    print(f"{'n_input':>8} {'amp':>5} {'rank99':>7} {'finite':>7}")
    for n_input in [0, 5, 10, 20, 40, n_met]:
        ni = min(n_input, n_met)
        for amp in (0.5, 1.0):
            traj = simulate(model, n_met, T, dt, c0, ni, amp, 0)
            fin = bool(np.isfinite(traj).all())
            r = rank99(traj) if fin else -1
            print(f"{n_input:>8} {amp:>5} {r:>7} {str(fin):>7}")


if __name__ == "__main__":
    main()
