
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


class PDE_MichaelisMenten(nn.Module):
    """Michaelis-Menten kinetics ODE for metabolic networks.

    Each reaction follows saturating kinetics:

        v_j = Vmax_j * Π_{substrates k} [ c_k / (Km_{kj} + c_k) ]^|s_kj|

    Boundary species act as **time-varying external sources** — their
    concentrations are set at each time step via `stimulus`,
    analogous to stimuli in neural network models. Without time-varying
    boundary input, the system equilibrates to a steady state.

    For the inverse problem, the GNN must discover:
      - MLP_sub learns c/(Km+c) instead of c^s
      - Rate constants Vmax_j (per-reaction)

    X tensor layout (same as PDE_M1):
      x[:, 0]   = index (metabolite ID)
      x[:, 1:3] = positions (x, y) for visualisation
      x[:, 3]   = concentration
      x[:, 4]   = external input (unused)
      x[:, 5]   = 0 (unused)
      x[:, 6]   = metabolite_type
      x[:, 7]   = 0 (unused)

    Parameters
    ----------
    config : MetabolismGraphConfig
    stoich_graph : dict  with keys
        'sub': (met_sub, rxn_sub, sto_sub)   substrate edges
        'all': (met_all, rxn_all, sto_all)   all stoichiometric edges
        'stimulus_sub': (stim_idx, stim_rxn, stim_sto)  boundary substrate edges
    device : torch.device
    """

    def __init__(self, config=None, stoich_graph=None, device=None):
        super().__init__()

        n_met = config.simulation.n_metabolites
        n_rxn = config.simulation.n_reactions
        msg_dim = getattr(config.graph_model, 'output_size_sub', 1)

        hidden_sub = getattr(config.graph_model, 'hidden_dim_sub', 64)
        n_layers_sub = getattr(config.graph_model, 'n_layers_sub', 2)
        hidden_node = getattr(config.graph_model, 'hidden_dim_node', 64)
        n_layers_node = getattr(config.graph_model, 'n_layers_node', 2)

        self.n_met = n_met
        self.n_rxn = n_rxn
        self.device = device

        self.flux_limit_enabled = getattr(config.simulation, 'flux_limit', True)

        # substrate_func (MLP_sub): input=2 (c, |s|), output=msg_dim
        sub_sizes = [2] + [hidden_sub] * (n_layers_sub - 1) + [msg_dim]
        self.substrate_func = mlp(sub_sizes, activation=nn.Tanh)

        # rate_func (MLP_node): input=msg_dim, output=1
        node_sizes = [msg_dim] + [hidden_node] * (n_layers_node - 1) + [1]
        self.rate_func = mlp(node_sizes, activation=nn.Tanh)

        # --- Michaelis-Menten parameters ---
        sim_cfg = config.simulation

        # Vmax per reaction (log-space)
        log_k_min = getattr(sim_cfg, 'log_k_min', -2.0)
        log_k_max = getattr(sim_cfg, 'log_k_max', 0.0)
        log_vmax = torch.empty(n_rxn)
        log_vmax.uniform_(log_k_min, log_k_max)
        self.log_k = nn.Parameter(log_vmax)  # named log_k for pipeline compatibility

        # Km per substrate edge (log-space)
        n_sub_edges = stoich_graph['sub'][0].shape[0]
        log_km_min = getattr(sim_cfg, 'log_km_min', -1.0)
        log_km_max = getattr(sim_cfg, 'log_km_max', 1.0)
        log_km = torch.empty(n_sub_edges)
        log_km.uniform_(log_km_min, log_km_max)
        self.log_km = nn.Parameter(log_km, requires_grad=False)

        # stoichiometric graph
        (met_sub, rxn_sub, sto_sub) = stoich_graph['sub']
        (met_all, rxn_all, sto_all) = stoich_graph['all']

        self.register_buffer('met_sub', met_sub)
        self.register_buffer('rxn_sub', rxn_sub)
        self.register_buffer('sto_sub', sto_sub)
        self.register_buffer('met_all', met_all)
        self.register_buffer('rxn_all', rxn_all)
        self.register_buffer('sto_all', sto_all)

        # --- Boundary species (external sources) ---
        self.n_stimulus = 0
        boundary_conc = getattr(sim_cfg, 'stimulus_concentrations', None)
        if boundary_conc is not None and len(boundary_conc) > 0:
            self.n_stimulus = len(boundary_conc)
            # baseline boundary concentrations (used when no stimulus provided)
            self.register_buffer('stimulus_conc_baseline',
                                 torch.tensor(boundary_conc, dtype=torch.float32, device=device))

        stimulus_sub = stoich_graph.get('stimulus_sub', None)
        if stimulus_sub is not None:
            (stim_idx, stim_rxn, stim_sto) = stimulus_sub
            self.register_buffer('stim_idx', stim_idx)
            self.register_buffer('stim_rxn', stim_rxn)
            self.register_buffer('stim_sto', stim_sto)
            n_bnd_edges = stim_idx.shape[0]
            log_km_stim = torch.empty(n_bnd_edges)
            log_km_stim.uniform_(log_km_min, log_km_max)
            self.log_km_stim = nn.Parameter(log_km_stim, requires_grad=False)
        else:
            self.register_buffer('stim_idx', None)

        # homeostatic dynamics
        self.homeostatic_strength = sim_cfg.homeostatic_strength
        self.baseline_mode = sim_cfg.baseline_mode
        self.baseline_concentration = sim_cfg.baseline_concentration
        self.circadian_amplitude = sim_cfg.circadian_amplitude
        self.circadian_period = sim_cfg.circadian_period
        self.n_metabolite_types = getattr(sim_cfg, 'n_metabolite_types', 1)

        p = torch.zeros(self.n_metabolite_types, 2)
        lambda_per_type = getattr(sim_cfg, 'homeostatic_lambda_per_type', None)
        if lambda_per_type is not None and len(lambda_per_type) == self.n_metabolite_types:
            for t, lam in enumerate(lambda_per_type):
                p[t, 0] = lam
        else:
            p[:, 0] = self.homeostatic_strength
        baseline_per_type = getattr(sim_cfg, 'homeostatic_baseline_per_type', None)
        if baseline_per_type is not None and len(baseline_per_type) == self.n_metabolite_types:
            for t, base in enumerate(baseline_per_type):
                p[t, 1] = base
        else:
            p[:, 1] = self.baseline_concentration

        self.p = nn.Parameter(p, requires_grad=False)
        self.register_buffer('c_baseline', None)

    def forward(self, data=None, has_field=False, frame=None, dt=None,
                stimulus=None):
        """Compute dx/dt for all metabolites using Michaelis-Menten kinetics.

        Parameters
        ----------
        data : torch_geometric.data.Data
        frame : int  current time step (for circadian)
        dt : float  time step size
        stimulus : Tensor (n_stimulus,) or None
            Time-varying boundary species concentrations for this frame.
            If None, uses the fixed baseline concentrations.
        """
        x = data.x
        concentrations = x[:, 3]

        if self.c_baseline is None:
            if self.baseline_mode == "initial":
                self.c_baseline = concentrations.clone().detach()
            else:
                self.c_baseline = torch.full_like(concentrations, self.baseline_concentration)

        vmax = torch.pow(10.0, self.log_k)
        km = torch.pow(10.0, self.log_km)

        # --- Michaelis-Menten rate for floating substrates ---
        eps = 1e-8
        c_sub = concentrations[self.met_sub].clamp(min=eps)
        saturation = c_sub / (km + c_sub)
        sat_powered = torch.pow(saturation, self.sto_sub)

        log_sat = torch.log(sat_powered.clamp(min=eps))
        log_prod = torch.zeros(self.n_rxn, dtype=log_sat.dtype, device=log_sat.device)
        log_prod.index_add_(0, self.rxn_sub, log_sat)

        # --- Boundary species contribution (time-varying stimulus) ---
        if self.stim_idx is not None:
            km_stim = torch.pow(10.0, self.log_km_stim)
            # Use stimulus if provided, otherwise baseline
            if stimulus is not None:
                c_stim = stimulus[self.stim_idx].clamp(min=eps)
            else:
                c_stim = self.stimulus_conc_baseline[self.stim_idx].clamp(min=eps)
            sat_stim = c_stim / (km_stim + c_stim)
            sat_stim_powered = torch.pow(sat_stim, self.stim_sto)
            log_sat_stim = torch.log(sat_stim_powered.clamp(min=eps))
            log_prod.index_add_(0, self.stim_rxn, log_sat_stim)

        v = vmax * torch.exp(log_prod)

        # flux limiting
        if self.flux_limit_enabled and dt is not None and dt > 0:
            v = self._flux_limit(v, concentrations, dt)

        # dx/dt = Σ_j S_ij * v_j
        contrib = self.sto_all * v[self.rxn_all]
        dxdt = torch.zeros(self.n_met, dtype=contrib.dtype, device=contrib.device)
        dxdt.index_add_(0, self.met_all, contrib)

        # homeostatic term
        if self.homeostatic_strength > 0 or self.n_metabolite_types > 1:
            metabolite_type = x[:, 6].long()
            lambda_i = self.p[metabolite_type, 0]
            c_baseline_i = self.p[metabolite_type, 1]
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
        total_consumption = torch.zeros(self.n_met, dtype=v.dtype, device=v.device)
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
