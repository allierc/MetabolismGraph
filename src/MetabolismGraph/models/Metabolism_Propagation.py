
import torch
import torch.nn as nn
import numpy as np
from MetabolismGraph.models.Siren_Network import Siren


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


class Metabolism_Propagation(nn.Module):
    """training model for metabolic networks with learnable stoichiometry.

    mirrors PDE_M2 architecture but makes stoichiometric coefficients
    learnable nn.Parameters (instead of fixed buffers), so that training
    can recover the ground-truth stoichiometric matrix from time-series
    data (dx/dt predictions).

    learnable parameters
    --------------------
    sto_all : nn.Parameter (n_all_edges,)
        signed stoichiometric coefficients for all edges.
        substrate |stoich| for messages is derived as |sto_all[sub_to_all]|.
    substrate_func : nn.Sequential
        substrate function: [concentration, |stoich|] -> contribution vector.
    rate_func : nn.Sequential
        rate function: aggregated substrate info -> scalar rate.
    log_k : nn.Parameter (n_rxn,)
        log10 per-reaction rate constants.

    buffers (fixed graph structure)
    ------
    met_sub, rxn_sub : LongTensor
        metabolite and reaction indices for substrate edges.
    met_all, rxn_all : LongTensor
        metabolite and reaction indices for all edges.
    sub_to_all : LongTensor (n_sub_edges,)
        maps each substrate edge to its index in the all-edges array,
        so that |sto_all[sub_to_all]| gives substrate absolute coefficients.

    optional: SIREN for visual field reconstruction (external input).
    """

    def __init__(self, config=None, device=None):
        super().__init__()

        simulation_config = config.simulation
        model_config = config.graph_model

        n_met = simulation_config.n_metabolites
        n_rxn = simulation_config.n_reactions
        hidden = model_config.hidden_dim
        msg_dim = model_config.output_size

        self.n_met = n_met
        self.n_rxn = n_rxn
        self.device = device
        self.external_input_mode = getattr(
            simulation_config, 'external_input_mode', 'none'
        )
        self.n_input_metabolites = simulation_config.n_input_metabolites
        self.dimension = simulation_config.dimension

        # learnable functions (same architecture as PDE_M2)
        self.substrate_func = mlp([2, hidden, msg_dim], activation=nn.Tanh)
        self.rate_func = mlp([msg_dim, hidden, 1], activation=nn.Tanh)
        self.softplus = nn.Softplus(beta=1.0)

        # per-reaction rate constants
        log_k = torch.empty(n_rxn)
        log_k.uniform_(-2.0, 1.0)
        self.log_k = nn.Parameter(log_k)

        # field_type for optional SIREN visual field
        self.field_type = model_config.field_type
        if 'visual' in self.field_type:
            print('use NNR for visual field reconstruction')
            self.NNR_f = Siren(
                in_features=model_config.input_size_nnr_f,
                out_features=model_config.output_size_nnr_f,
                hidden_features=model_config.hidden_dim_nnr_f,
                hidden_layers=model_config.n_layers_nnr_f,
                first_omega_0=model_config.omega_f,
                hidden_omega_0=model_config.omega_f,
                outermost_linear=model_config.outermost_linear_nnr_f,
            )
            self.NNR_f.to(device)
            self.NNR_f_xy_period = model_config.nnr_f_xy_period / (2 * np.pi)
            self.NNR_f_T_period = model_config.nnr_f_T_period / (2 * np.pi)

        self.p = torch.zeros(1)

    def load_stoich_graph(self, stoich_graph):
        """load bipartite graph structure and initialize learnable coefficients.

        Parameters
        ----------
        stoich_graph : dict
            'sub': (met_sub, rxn_sub, sto_sub)  substrate edges
            'all': (met_all, rxn_all, sto_all)  all stoichiometric edges
        """
        (met_sub, rxn_sub, gt_sto_sub) = stoich_graph['sub']
        (met_all, rxn_all, gt_sto_all) = stoich_graph['all']

        dev = met_sub.device

        # graph structure: fixed buffers
        self.register_buffer('met_sub', met_sub)
        self.register_buffer('rxn_sub', rxn_sub)
        self.register_buffer('met_all', met_all)
        self.register_buffer('rxn_all', rxn_all)

        # map substrate edges -> their index in the all-edges array
        # so |sto_all[sub_to_all]| gives substrate absolute coefficients
        all_edge_dict = {}
        for idx in range(met_all.shape[0]):
            key = (met_all[idx].item(), rxn_all[idx].item())
            all_edge_dict[key] = idx
        sub_to_all = torch.tensor(
            [all_edge_dict[(met_sub[i].item(), rxn_sub[i].item())]
             for i in range(met_sub.shape[0])],
            dtype=torch.long, device=dev,
        )
        self.register_buffer('sub_to_all', sub_to_all)

        # single learnable stoichiometric coefficient vector (all edges)
        n_all_edges = met_all.shape[0]
        self.sto_all = nn.Parameter(torch.randn(n_all_edges, device=dev) * 0.1)

        # pre-compute number of substrates per reaction for averaging
        n_sub_per_rxn = torch.zeros(self.n_rxn, dtype=torch.float32, device=dev)
        n_sub_per_rxn.index_add_(
            0, rxn_sub, torch.ones_like(rxn_sub, dtype=torch.float32)
        )
        n_sub_per_rxn.clamp_(min=1.0)
        self.register_buffer('n_sub_per_rxn', n_sub_per_rxn)

        # extract product edges from 'all' graph (positive ground-truth stoichiometry)
        prod_mask = gt_sto_all > 0
        met_prod = met_all[prod_mask]
        rxn_prod = rxn_all[prod_mask]
        self.register_buffer('met_prod', met_prod)
        self.register_buffer('rxn_prod', rxn_prod)

        # product-edge indices within the sto_all array (for multiplicative_product mode)
        prod_indices = torch.where(prod_mask)[0]
        self.register_buffer('prod_indices', prod_indices)

        # pre-compute number of products per reaction
        n_prod_per_rxn = torch.zeros(self.n_rxn, dtype=torch.float32, device=dev)
        n_prod_per_rxn.index_add_(
            0, rxn_prod, torch.ones_like(rxn_prod, dtype=torch.float32)
        )
        n_prod_per_rxn.clamp_(min=1.0)
        self.register_buffer('n_prod_per_rxn', n_prod_per_rxn)

    def forward_visual(self, x, k):
        """reconstruct external input via SIREN (position + time -> field).

        Parameters
        ----------
        x : Tensor (n_met, 8)
            metabolite features (positions at x[:, 1:3]).
        k : int or float
            frame index.

        Returns
        -------
        reconstructed_field : Tensor (n_input_metabolites, 1)
        """
        kk = torch.full(
            (x.size(0), 1), float(k), device=self.device, dtype=torch.float32
        )
        in_features = torch.cat(
            (x[:, 1:1 + self.dimension] / self.NNR_f_xy_period,
             kk / self.NNR_f_T_period),
            dim=1,
        )
        reconstructed_field = self.NNR_f(in_features[:self.n_input_metabolites]) ** 2
        return reconstructed_field

    def forward(self, data=None, has_field=False, frame=None):
        """compute dx/dt for all metabolites.

        same computation as PDE_M2.forward() but using learnable
        stoichiometric coefficients (sto_all). substrate |stoich| for
        messages is derived as |sto_all[sub_to_all]|.

        Returns
        -------
        dxdt : Tensor (n_met, 1)
        """
        x = data.x
        concentrations = x[:, 3]
        external_input = x[:, 4]

        # 1. gather substrate concentrations; derive |stoich| from sto_all
        x_src = concentrations[self.met_sub].unsqueeze(-1)
        s_abs = self.sto_all[self.sub_to_all].abs().unsqueeze(-1)
        msg_in = torch.cat([x_src, s_abs], dim=-1)

        # 2. compute messages
        msg = self.substrate_func(msg_in)

        # 3. aggregate messages per reaction
        h_rxn = torch.zeros(
            self.n_rxn, msg.shape[1], dtype=msg.dtype, device=msg.device
        )
        h_rxn.index_add_(0, self.rxn_sub, msg)

        # 4. compute base reaction rates
        k = torch.pow(10.0, self.log_k)
        base_v = self.softplus(self.rate_func(h_rxn).squeeze(-1))

        # 5. apply external modulation
        if self.external_input_mode == "multiplicative_substrate":
            ext_src = external_input[self.met_sub]
            ext_agg = torch.zeros(
                self.n_rxn, dtype=ext_src.dtype, device=ext_src.device
            )
            ext_agg.index_add_(0, self.rxn_sub, ext_src)
            ext_mean = ext_agg / self.n_sub_per_rxn
            v = k * ext_mean * base_v
        elif self.external_input_mode == "multiplicative_product":
            ext_src = external_input[self.met_prod]
            ext_agg = torch.zeros(
                self.n_rxn, dtype=ext_src.dtype, device=ext_src.device
            )
            ext_agg.index_add_(0, self.rxn_prod, ext_src)
            ext_mean = ext_agg / self.n_prod_per_rxn
            v = k * ext_mean * base_v
        else:
            v = k * base_v

        # 6. dx/dt via learnable stoichiometric matrix
        contrib = self.sto_all * v[self.rxn_all]
        dxdt = torch.zeros(
            self.n_met, dtype=contrib.dtype, device=contrib.device
        )
        dxdt.index_add_(0, self.met_all, contrib)

        # 7. additive external input
        if self.external_input_mode == "additive":
            dxdt = dxdt + external_input

        return dxdt.unsqueeze(-1)
