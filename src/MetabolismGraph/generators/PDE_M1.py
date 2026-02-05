
import torch
import torch.nn as nn


def mlp(sizes, activation=nn.Tanh, final_activation=None):
    """build a simple feedforward MLP."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif final_activation is not None:
            layers.append(final_activation())
    return nn.Sequential(*layers)


class PDE_M1(nn.Module):
    """Stoichiometric kinetics ODE for metabolic networks.

    Given a fixed stoichiometric matrix S (n_met x n_rxn), learn the reaction
    rate functions from metabolite concentrations.

    The stoichiometric graph is bipartite (metabolites <-> reactions):
      - substrate edges: metabolite -> reaction with |s_ij| (consumed)
      - all edges: metabolite -> reaction with signed s_ij

    Forward pass:
      1. For each substrate edge, build message from [concentration, |stoich|]
      2. Aggregate messages per reaction via sum
      3. Compute rate v_j = softplus(rate_func(h_j))  (non-negative rates)
      4. dx/dt = sum over edges: s_ij * v_j

    X tensor layout:
      x[:, 0]   = index (metabolite ID)
      x[:, 1:3] = positions (x, y) for visualisation
      x[:, 3]   = concentration
      x[:, 4]   = external input (unused in M1)
      x[:, 5]   = 0 (unused)
      x[:, 6]   = metabolite_type
      x[:, 7]   = 0 (unused)

    Parameters
    ----------
    config : MetabolismGraphConfig
    stoich_graph : dict  with keys
        'sub': (met_sub, rxn_sub, sto_sub)   substrate edges
        'all': (met_all, rxn_all, sto_all)   all stoichiometric edges
    device : torch.device
    """

    def __init__(self, config=None, stoich_graph=None, device=None):
        super().__init__()

        n_met = config.simulation.n_metabolites
        n_rxn = config.simulation.n_reactions
        hidden = config.graph_model.hidden_dim
        msg_dim = config.graph_model.output_size

        self.n_met = n_met
        self.n_rxn = n_rxn
        self.device = device

        # learnable functions (larger init for diverse outputs)
        self.substrate_func = mlp([2, hidden, msg_dim], activation=nn.Tanh)
        self.rate_func = mlp([msg_dim, hidden, 1], activation=nn.Tanh)
        self.softplus = nn.Softplus(beta=1.0)

        # per-reaction rate constants: log-uniform in [0.001, 0.1]
        # 2 orders of magnitude variation, slow enough to fill 2880-frame window
        log_k = torch.empty(n_rxn)
        log_k.uniform_(-3.0, -1.0)  # log10 in [-3, -1]  =>  k in [0.001, 0.1]
        self.log_k = nn.Parameter(log_k)

        # store stoichiometric graph (fixed, not learned)
        (met_sub, rxn_sub, sto_sub) = stoich_graph['sub']
        (met_all, rxn_all, sto_all) = stoich_graph['all']

        self.register_buffer('met_sub', met_sub)
        self.register_buffer('rxn_sub', rxn_sub)
        self.register_buffer('sto_sub', sto_sub)
        self.register_buffer('met_all', met_all)
        self.register_buffer('rxn_all', rxn_all)
        self.register_buffer('sto_all', sto_all)

        self.p = torch.zeros(1)

    def forward(self, data=None, has_field=False, frame=None, dt=None):
        """Compute dx/dt for all metabolites."""
        x = data.x
        concentrations = x[:, 3]

        # 1. gather substrate concentrations and stoichiometric coefficients
        x_src = concentrations[self.met_sub].unsqueeze(-1)
        s_abs = self.sto_sub.unsqueeze(-1)
        msg_in = torch.cat([x_src, s_abs], dim=-1)

        # 2. compute messages
        msg = self.substrate_func(msg_in)

        # 3. aggregate messages per reaction
        h_rxn = torch.zeros(self.n_rxn, msg.shape[1], dtype=msg.dtype, device=msg.device)
        h_rxn.index_add_(0, self.rxn_sub, msg)

        # 4. compute non-negative reaction rates, scaled by per-reaction k_j
        k = torch.pow(10.0, self.log_k)
        v = k * self.softplus(self.rate_func(h_rxn).squeeze(-1))

        # 5. flux limiting: scale rates so no substrate goes negative
        if dt is not None and dt > 0:
            v = self._flux_limit(v, concentrations, dt)

        # 6. compute dx/dt via stoichiometric matrix: dx_i/dt = sum_j S_ij * v_j
        contrib = self.sto_all * v[self.rxn_all]
        dxdt = torch.zeros(self.n_met, dtype=contrib.dtype, device=contrib.device)
        dxdt.index_add_(0, self.met_all, contrib)

        return dxdt.unsqueeze(-1)

    def _flux_limit(self, v, concentrations, dt):
        """Scale reaction rates so no substrate is over-consumed in one step."""
        consumption = self.sto_sub * v[self.rxn_sub] * dt

        total_consumption = torch.zeros(
            self.n_met, dtype=v.dtype, device=v.device
        )
        total_consumption.index_add_(0, self.met_sub, consumption)

        met_scale = torch.ones(self.n_met, dtype=v.dtype, device=v.device)
        active = total_consumption > 1e-12
        met_scale[active] = torch.clamp(
            concentrations[active] / total_consumption[active], max=1.0
        )

        edge_scale = met_scale[self.met_sub]
        rxn_scale = torch.ones(self.n_rxn, dtype=v.dtype, device=v.device)
        rxn_scale.scatter_reduce_(
            0, self.rxn_sub, edge_scale, reduce='amin', include_self=True
        )

        return v * rxn_scale

    def get_rates(self, data):
        """Return reaction rates for diagnostics."""
        x = data.x
        concentrations = x[:, 3]

        x_src = concentrations[self.met_sub].unsqueeze(-1)
        s_abs = self.sto_sub.unsqueeze(-1)
        msg_in = torch.cat([x_src, s_abs], dim=-1)
        msg = self.substrate_func(msg_in)

        h_rxn = torch.zeros(self.n_rxn, msg.shape[1], dtype=msg.dtype, device=msg.device)
        h_rxn.index_add_(0, self.rxn_sub, msg)

        k = torch.pow(10.0, self.log_k)
        v = k * self.softplus(self.rate_func(h_rxn).squeeze(-1))
        return v
