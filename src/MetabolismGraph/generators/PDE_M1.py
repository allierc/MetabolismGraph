
import math
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
      3. Compute rate v_j = k_j * rate_func(h_j)
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
        msg_dim = config.graph_model.output_size

        # per-MLP architecture from config (with fallbacks)
        hidden_sub = getattr(config.graph_model, 'hidden_dim_sub', config.graph_model.hidden_dim)
        n_layers_sub = getattr(config.graph_model, 'n_layers_sub', 2)
        hidden_node = getattr(config.graph_model, 'hidden_dim_node', config.graph_model.hidden_dim)
        n_layers_node = getattr(config.graph_model, 'n_layers_node', 2)

        self.n_met = n_met
        self.n_rxn = n_rxn
        self.device = device

        # mass-action kinetics: v = k * Π(c^s) (multiplicative rates for oscillations)
        self.use_mass_action = getattr(config.simulation, 'use_mass_action', False)

        # flux limiting: prevent negative concentrations (can dampen oscillations)
        self.flux_limit_enabled = getattr(config.simulation, 'flux_limit', True)

        # substrate_func (MLP_sub): input=2 (c, |s|), output=msg_dim
        sub_sizes = [2] + [hidden_sub] * (n_layers_sub - 1) + [msg_dim]
        self.substrate_func = mlp(sub_sizes, activation=nn.Tanh)

        # rate_func (MLP_node): input=msg_dim, output=1
        node_sizes = [msg_dim] + [hidden_node] * (n_layers_node - 1) + [1]
        self.rate_func = mlp(node_sizes, activation=nn.Tanh)

        # per-reaction rate constants: log-uniform in configurable range
        log_k_min = getattr(config.simulation, 'log_k_min', -3.0)
        log_k_max = getattr(config.simulation, 'log_k_max', -1.0)
        log_k = torch.empty(n_rxn)
        log_k.uniform_(log_k_min, log_k_max)  # k in [10^min, 10^max]
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

        # homeostatic dynamics parameters
        sim_cfg = config.simulation
        self.homeostatic_strength = sim_cfg.homeostatic_strength
        self.baseline_mode = sim_cfg.baseline_mode
        self.baseline_concentration = sim_cfg.baseline_concentration
        self.circadian_amplitude = sim_cfg.circadian_amplitude
        self.circadian_period = sim_cfg.circadian_period

        # metabolite types for per-type homeostasis
        self.n_metabolite_types = getattr(sim_cfg, 'n_metabolite_types', 1)

        # per-type parameters: p[type, :] = [lambda, c_baseline]
        p = torch.zeros(self.n_metabolite_types, 2)

        # support per-type lambda values via list config
        lambda_per_type = getattr(sim_cfg, 'homeostatic_lambda_per_type', None)
        if lambda_per_type is not None and len(lambda_per_type) == self.n_metabolite_types:
            for t, lam in enumerate(lambda_per_type):
                p[t, 0] = lam
        else:
            p[:, 0] = self.homeostatic_strength  # same lambda for all types

        # support per-type baseline values via list config
        baseline_per_type = getattr(sim_cfg, 'homeostatic_baseline_per_type', None)
        if baseline_per_type is not None and len(baseline_per_type) == self.n_metabolite_types:
            for t, base in enumerate(baseline_per_type):
                p[t, 1] = base
        else:
            p[:, 1] = self.baseline_concentration  # same baseline for all types

        self.p = nn.Parameter(p, requires_grad=False)  # fixed parameters

        # baseline will be set on first forward pass if baseline_mode="initial"
        self.register_buffer('c_baseline', None)

    def forward(self, data=None, has_field=False, frame=None, dt=None):
        """Compute dx/dt for all metabolites."""
        x = data.x
        concentrations = x[:, 3]

        # initialize baseline on first call if using "initial" mode
        if self.c_baseline is None:
            if self.baseline_mode == "initial":
                self.c_baseline = concentrations.clone().detach()
            else:  # "fixed"
                self.c_baseline = torch.full_like(concentrations, self.baseline_concentration)

        # compute reaction rates
        k = torch.pow(10.0, self.log_k)

        if self.use_mass_action:
            # True mass-action kinetics: v_j = k_j * Π_{substrates} c_i^|s_ij|
            # Use log-space for numerical stability: log(Π c^s) = Σ s*log(c)
            eps = 1e-8
            log_c = torch.log(concentrations[self.met_sub].clamp(min=eps))
            log_contrib = log_c * self.sto_sub  # s * log(c)
            log_prod = torch.zeros(self.n_rxn, dtype=log_c.dtype, device=log_c.device)
            log_prod.index_add_(0, self.rxn_sub, log_contrib)
            v = k * torch.exp(log_prod)
        else:
            # GNN message-passing rate computation (additive aggregation)
            # 1. gather substrate concentrations and stoichiometric coefficients
            x_src = concentrations[self.met_sub].unsqueeze(-1)
            s_abs = self.sto_sub.unsqueeze(-1)
            msg_in = torch.cat([x_src, s_abs], dim=-1)

            # 2. compute messages
            msg = self.substrate_func(msg_in)

            # 3. aggregate messages per reaction
            h_rxn = torch.zeros(self.n_rxn, msg.shape[1], dtype=msg.dtype, device=msg.device)
            h_rxn.index_add_(0, self.rxn_sub, msg)

            # 4. compute reaction rates
            v = k * self.rate_func(h_rxn).squeeze(-1)

        # 5. flux limiting: scale rates so no substrate goes negative
        if self.flux_limit_enabled and dt is not None and dt > 0:
            v = self._flux_limit(v, concentrations, dt)

        # 6. compute dx/dt via stoichiometric matrix: dx_i/dt = sum_j S_ij * v_j
        contrib = self.sto_all * v[self.rxn_all]
        dxdt = torch.zeros(self.n_met, dtype=contrib.dtype, device=contrib.device)
        dxdt.index_add_(0, self.met_all, contrib)

        # 7. homeostatic term: -λ_i * (c_i - c_baseline_i(t))
        # use per-type parameters from self.p
        if self.homeostatic_strength > 0 or self.n_metabolite_types > 1:
            metabolite_type = x[:, 6].long()
            lambda_i = self.p[metabolite_type, 0]  # per-metabolite homeostatic strength
            c_baseline_i = self.p[metabolite_type, 1]  # per-metabolite baseline

            # compute time-dependent baseline with circadian modulation
            if self.circadian_amplitude > 0 and frame is not None:
                phase = 2 * math.pi * frame / self.circadian_period
                modulation = 1 + self.circadian_amplitude * math.sin(phase)
                c_target = c_baseline_i * modulation
            else:
                c_target = c_baseline_i
            dxdt = dxdt - lambda_i * (concentrations - c_target)

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
        v = k * self.rate_func(h_rxn).squeeze(-1)
        return v
