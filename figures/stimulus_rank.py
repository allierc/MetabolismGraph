#!/usr/bin/env python
"""
Phase-1 figure: activity rank vs number of driven metabolites, on the real
topologies (e_coli core, yeast-GEM subgraph) with imposed MM kinetics. Confirms the
controllability-subspace prediction: an external metabolite source on m metabolites
raises the trajectory's activity rank ~linearly (rank ~ m), lifting a closed network
from its rank-~2 steady state past the rank-20 identifiability target.

Output: figures/metabolism/stimulus_rank.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
from rank_sweep import simulate, rank99
from MetabolismGraph.config import MetabolismGraphConfig
from MetabolismGraph.generators.PDE_MichaelisMenten import PDE_MichaelisMenten
from MetabolismGraph.generators.sbml_topology import parse_fba_sbml, central_carbon_subgraph
from MetabolismGraph.generators.utils import init_concentration

plt.rcParams.update({"font.size": 13, "axes.labelsize": 15, "legend.fontsize": 11})


def sweep(cfg_name, n_inputs):
    config = MetabolismGraphConfig.from_yaml(f"config/{cfg_name}.yaml")
    config.simulation.external_input_mode = "additive"
    topo = config.simulation.topology_sbml
    nsub = getattr(config.simulation, "topology_subgraph_reactions", 0)
    if nsub and nsub > 0:
        sg, S, *_ = central_carbon_subgraph(topo, max_reactions=nsub, device="cpu")
    else:
        sg, S, *_ = parse_fba_sbml(topo, device="cpu")
    n_met = S.shape[0]
    config.simulation.n_metabolites, config.simulation.n_reactions = S.shape
    torch.manual_seed(0)
    model = PDE_MichaelisMenten(config=config, stoich_graph=sg, device="cpu")
    c0 = init_concentration(n_met, "cpu", mode="random", seed=0, c_min=0.2, c_max=3.0)
    dt = config.simulation.delta_t
    ranks = []
    for ni in n_inputs:
        traj = simulate(model, n_met, 1500, dt, c0, min(ni, n_met), 1.0, 0)
        ranks.append(rank99(traj) if np.isfinite(traj).all() else np.nan)
    return n_met, ranks


def main():
    n_inputs = [0, 5, 10, 20, 40, 80]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for cfg, lab, col in [("ecoli_core_mm", "E. coli core (72 met)", "#d62728"),
                          ("yeast_central_mm", "yeast-GEM subgraph (208 met)", "#1f77b4")]:
        n_met, ranks = sweep(cfg, n_inputs)
        xs = [min(ni, n_met) for ni in n_inputs]
        ax.plot(xs, ranks, "o-", color=col, lw=2, label=lab)
    lim = max(n_inputs)
    ax.plot([0, lim], [0, lim], "--", color="0.6", lw=1, label="rank = #driven (identity)")
    ax.axhline(20, color="0.3", ls=":", lw=1.2)
    ax.text(1, 21, "rank-20 identifiability target", fontsize=10, color="0.3")
    ax.set_xlabel("number of driven metabolites (external source)")
    ax.set_ylabel("activity rank (99%)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    out = os.path.join("figures/metabolism", "stimulus_rank.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
