
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
        msg_dim = getattr(model_config, 'output_size_sub', getattr(model_config, 'output_size', 1))

        # per-MLP architecture from config (with fallbacks)
        hidden_sub = getattr(model_config, 'hidden_dim_sub', 64)
        n_layers_sub = getattr(model_config, 'n_layers_sub', 3)
        hidden_node = getattr(model_config, 'hidden_dim_node', 64)
        n_layers_node = getattr(model_config, 'n_layers_node', 3)

        self.n_met = n_met
        self.n_rxn = n_rxn
        self.device = device
        self.external_input_mode = getattr(
            simulation_config, 'external_input_mode', 'none'
        )
        self.n_input_metabolites = simulation_config.n_input_metabolites
        self.dimension = simulation_config.dimension

        # aggregation type: 'add' (sum) or 'mul' (product in log-space)
        self.aggr_type = getattr(model_config, 'aggr_type', 'add')

        # metabolite embeddings a_i
        embedding_dim = getattr(model_config, 'embedding_dim', 2)
        self.embedding_dim = embedding_dim
        self.a = nn.Parameter(torch.randn(n_met, embedding_dim) * 0.1)

        # MLP_node: (c_i, a_i) -> homeostasis term (learns -λ_i(c_i - c_baseline))
        # Hidden layers keep default (Kaiming) init for gradient flow;
        # only the output layer is zeroed so homeostasis starts inactive.
        node_sizes = [1 + embedding_dim] + [hidden_node] * (n_layers_node - 1) + [1]
        self.node_func = mlp(node_sizes, activation=nn.Tanh)
        with torch.no_grad():
            # Zero only the last Linear layer (output layer)
            for module in reversed(list(self.node_func.modules())):
                if isinstance(module, nn.Linear):
                    module.weight.zero_()
                    module.bias.zero_()
                    break

        # MLP_sub: (c_k, |s_kj|) -> substrate contribution (learns c^s)
        sub_sizes = [2] + [hidden_sub] * (n_layers_sub - 1) + [1]
        self.substrate_func = mlp(sub_sizes, activation=nn.Tanh)
        self.softplus = nn.Softplus(beta=1.0)

        # per-reaction rate constants k_j
        log_k = torch.empty(n_rxn)
        log_k.uniform_(-2.0, 1.0)
        self.log_k = nn.Parameter(log_k)

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

    def forward(self, data=None, has_field=False, frame=None):
        """Compute dx/dt for all metabolites.

        GNN parameterization from documentation:
        dc_i/dt = MLP_node(c_i, a_i) + Σ_j S_ij * k_j * aggr(MLP_sub(c_k, s_kj))

        Returns
        -------
        dxdt : Tensor (n_met, 1)
        """
        x = data.x
        concentrations = x[:, 3]
        external_input = x[:, 4]

        # ===== MLP_node: homeostasis term =====
        # MLP_node(c_i, a_i) learns -λ_i(c_i - c_baseline)
        node_in = torch.cat([concentrations.unsqueeze(-1), self.a], dim=-1)
        homeostasis = self.node_func(node_in).squeeze(-1)

        # ===== MLP_sub: substrate contribution =====
        # gather (c_k, |s_kj|) for each substrate edge
        c_sub = concentrations[self.met_sub].unsqueeze(-1)
        s_abs = self.sto_all[self.sub_to_all].abs().unsqueeze(-1)
        msg_in = torch.cat([c_sub, s_abs], dim=-1)

        # MLP_sub(c_k, s_kj) learns c^s
        msg = self.substrate_func(msg_in)

        # ===== Aggregation: sum or product =====
        if self.aggr_type == 'mul':
            # multiplicative: Π MLP_sub via log-space scatter
            eps = 1e-8
            log_msg = torch.log(msg.abs().clamp(min=eps))
            log_agg = torch.zeros(self.n_rxn, dtype=msg.dtype, device=msg.device)
            log_agg.index_add_(0, self.rxn_sub, log_msg.squeeze(-1))
            agg = torch.exp(log_agg)
        else:
            # additive: Σ MLP_sub
            agg = torch.zeros(self.n_rxn, dtype=msg.dtype, device=msg.device)
            agg.index_add_(0, self.rxn_sub, msg.squeeze(-1))

        # ===== Reaction rates: v = k * aggr =====
        k = torch.pow(10.0, self.log_k)
        base_v = self.softplus(agg)  # ensure positive rates

        # ===== External modulation (optional) =====
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

        # ===== dx/dt = Σ S_ij * v_j (stoichiometric scatter) =====
        contrib = self.sto_all * v[self.rxn_all]
        dxdt = torch.zeros(self.n_met, dtype=contrib.dtype, device=contrib.device)
        dxdt.index_add_(0, self.met_all, contrib)

        # ===== Add homeostasis term =====
        dxdt = dxdt + homeostasis

        # ===== External additive input (optional) =====
        if self.external_input_mode == "additive":
            dxdt = dxdt + external_input

        return dxdt.unsqueeze(-1)
