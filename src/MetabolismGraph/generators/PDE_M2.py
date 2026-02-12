
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


class PDE_M2(nn.Module):
    """Stoichiometric kinetics ODE with external rate modulation.

    Extends PDE_M1 by allowing an external signal (e.g. a movie mapped onto
    metabolite positions) to modulate the per-reaction rate constants.

    The external input lives in x[:, 4] (one value per metabolite).  For each
    reaction j the modulation factor is the mean external input across its
    substrate metabolites.

    Modulation modes (set via config.simulation.external_input_mode):
      - "multiplicative_substrate":  modulate rate via mean external input
            at substrate (reactant) metabolites of each reaction
      - "multiplicative_product":    modulate rate via mean external input
            at product metabolites of each reaction
      - "additive":        dx_i/dt += external_input_i   (direct flux)
      - "none":            identical to PDE_M1

    X tensor layout:
      x[:, 0]   = index (metabolite ID)
      x[:, 1:3] = positions (x, y)
      x[:, 3]   = concentration
      x[:, 4]   = external_input (modulation signal)
      x[:, 5]   = 0 (unused)
      x[:, 6]   = metabolite_type
      x[:, 7]   = 0 (unused)
    """

    def __init__(self, config=None, stoich_graph=None, device=None):
        super().__init__()

        n_met = config.simulation.n_metabolites
        n_rxn = config.simulation.n_reactions
        hidden = getattr(config.graph_model, 'hidden_dim_sub', getattr(config.graph_model, 'hidden_dim', 64))
        msg_dim = getattr(config.graph_model, 'output_size_sub', getattr(config.graph_model, 'output_size', 1))

        self.n_met = n_met
        self.n_rxn = n_rxn
        self.device = device
        self.external_input_mode = getattr(
            config.simulation, 'external_input_mode', 'none'
        )

        # learnable functions
        self.substrate_func = mlp([2, hidden, msg_dim], activation=nn.Tanh)
        self.rate_func = mlp([msg_dim, hidden, 1], activation=nn.Tanh)

        # per-reaction rate constants: log-uniform in [0.001, 0.1]
        log_k = torch.empty(n_rxn)
        log_k.uniform_(-3.0, -1.0)
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

        # pre-compute number of substrates per reaction for averaging
        n_sub_per_rxn = torch.zeros(n_rxn, dtype=torch.float32, device=rxn_sub.device)
        n_sub_per_rxn.index_add_(0, rxn_sub, torch.ones_like(rxn_sub, dtype=torch.float32))
        n_sub_per_rxn.clamp_(min=1.0)
        self.register_buffer('n_sub_per_rxn', n_sub_per_rxn)

        # extract product edges from the 'all' graph (positive stoichiometry)
        prod_mask = sto_all > 0
        met_prod = met_all[prod_mask]
        rxn_prod = rxn_all[prod_mask]
        self.register_buffer('met_prod', met_prod)
        self.register_buffer('rxn_prod', rxn_prod)

        # pre-compute number of products per reaction for averaging
        n_prod_per_rxn = torch.zeros(n_rxn, dtype=torch.float32, device=rxn_prod.device)
        n_prod_per_rxn.index_add_(0, rxn_prod, torch.ones_like(rxn_prod, dtype=torch.float32))
        n_prod_per_rxn.clamp_(min=1.0)
        self.register_buffer('n_prod_per_rxn', n_prod_per_rxn)

        # homeostatic dynamics parameters
        sim_cfg = config.simulation
        self.homeostatic_strength = sim_cfg.homeostatic_strength
        self.baseline_mode = sim_cfg.baseline_mode
        self.baseline_concentration = sim_cfg.baseline_concentration
        self.circadian_amplitude = sim_cfg.circadian_amplitude
        self.circadian_period = sim_cfg.circadian_period

        # baseline will be set on first forward pass if baseline_mode="initial"
        self.register_buffer('c_baseline', None)

        self.p = torch.zeros(1)

    def forward(self, data=None, has_field=False, frame=None, dt=None):
        """Compute dx/dt for all metabolites."""
        x = data.x
        concentrations = x[:, 3]
        external_input = x[:, 4] * 2

        # initialize baseline on first call if using "initial" mode
        if self.c_baseline is None:
            if self.baseline_mode == "initial":
                self.c_baseline = concentrations.clone().detach()
            else:  # "fixed"
                self.c_baseline = torch.full_like(concentrations, self.baseline_concentration)

        # 1. gather substrate concentrations and stoichiometric coefficients
        x_src = concentrations[self.met_sub].unsqueeze(-1)
        s_abs = self.sto_sub.unsqueeze(-1)
        msg_in = torch.cat([x_src, s_abs], dim=-1)

        # 2. compute messages
        msg = self.substrate_func(msg_in)

        # 3. aggregate messages per reaction
        h_rxn = torch.zeros(self.n_rxn, msg.shape[1], dtype=msg.dtype, device=msg.device)
        h_rxn.index_add_(0, self.rxn_sub, msg)

        # 4. compute base reaction rates
        k = torch.pow(10.0, self.log_k)
        base_v = self.rate_func(h_rxn).squeeze(-1)

        # 5. apply external modulation
        if self.external_input_mode == "multiplicative_substrate":
            ext_src = external_input[self.met_sub]
            ext_agg = torch.zeros(self.n_rxn, dtype=ext_src.dtype, device=ext_src.device)
            ext_agg.index_add_(0, self.rxn_sub, ext_src)
            ext_mean = ext_agg / self.n_sub_per_rxn
            v = k * ext_mean * base_v
        elif self.external_input_mode == "multiplicative_product":
            ext_src = external_input[self.met_prod]
            ext_agg = torch.zeros(self.n_rxn, dtype=ext_src.dtype, device=ext_src.device)
            ext_agg.index_add_(0, self.rxn_prod, ext_src)
            ext_mean = ext_agg / self.n_prod_per_rxn
            v = k * ext_mean * base_v
        else:
            v = k * base_v

        # 6. flux limiting: scale rates so no substrate goes negative
        if dt is not None and dt > 0:
            v = self._flux_limit(v, concentrations, dt)

        # 7. compute dx/dt via stoichiometric matrix
        contrib = self.sto_all * v[self.rxn_all]
        dxdt = torch.zeros(self.n_met, dtype=contrib.dtype, device=contrib.device)
        dxdt.index_add_(0, self.met_all, contrib)

        # 8. additive external input (direct flux injection)
        if self.external_input_mode == "additive":
            dxdt = dxdt + external_input

        # 9. homeostatic term: -λ * (c - c_baseline(t))
        if self.homeostatic_strength > 0:
            # compute time-dependent baseline with circadian modulation
            if self.circadian_amplitude > 0 and frame is not None:
                phase = 2 * math.pi * frame / self.circadian_period
                modulation = 1 + self.circadian_amplitude * math.sin(phase)
                c_target = self.c_baseline * modulation
            else:
                c_target = self.c_baseline
            dxdt = dxdt - self.homeostatic_strength * (concentrations - c_target)

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
        external_input = x[:, 4]

        x_src = concentrations[self.met_sub].unsqueeze(-1)
        s_abs = self.sto_sub.unsqueeze(-1)
        msg_in = torch.cat([x_src, s_abs], dim=-1)
        msg = self.substrate_func(msg_in)

        h_rxn = torch.zeros(self.n_rxn, msg.shape[1], dtype=msg.dtype, device=msg.device)
        h_rxn.index_add_(0, self.rxn_sub, msg)

        k = torch.pow(10.0, self.log_k)
        base_v = self.rate_func(h_rxn).squeeze(-1)

        if self.external_input_mode == "multiplicative_substrate":
            ext_src = external_input[self.met_sub]
            ext_agg = torch.zeros(self.n_rxn, dtype=ext_src.dtype, device=ext_src.device)
            ext_agg.index_add_(0, self.rxn_sub, ext_src)
            ext_mean = ext_agg / self.n_sub_per_rxn
            v = k * ext_mean * base_v
        elif self.external_input_mode == "multiplicative_product":
            ext_src = external_input[self.met_prod]
            ext_agg = torch.zeros(self.n_rxn, dtype=ext_src.dtype, device=ext_src.device)
            ext_agg.index_add_(0, self.rxn_prod, ext_src)
            ext_mean = ext_agg / self.n_prod_per_rxn
            v = k * ext_mean * base_v
        else:
            v = k * base_v

        return v
