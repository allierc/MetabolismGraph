import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from MetabolismGraph.utils import to_numpy


# ---------------------------------------------------------------------------
#  Loss plotting color scheme
# ---------------------------------------------------------------------------

LOSS_COLORS = {
    'loss': ('blue', 4, 'loss (no regul)'),           # prediction loss only
    'regul_total': ('cyan', 2, 'total regularization'),
    'S_L1': ('red', 1.5, 'S L1 sparsity'),            # promotes zero entries
    'S_integer': ('orange', 1.5, 'S integer penalty'), # pushes to integer values
    'mass_conservation': ('green', 1.5, 'mass conservation'),  # column sums = 0
}
"""
Color scheme for loss.tif plot components:
- **blue (thick)**: prediction loss (MSE on dc/dt, without regularization)
- **cyan**: total regularization (sum of all regularization terms)
- **red**: S L1 sparsity penalty (coeff_S_L1 * ||sto_all||_1)
- **orange**: S integer penalty (coeff_S_integer * sin²(π·S))
- **green**: mass conservation penalty (coeff_mass * column_sum²)
"""


def plot_loss(loss_dict, log_dir, epoch=None, Niter=None, debug=False,
              current_loss=None, current_regul=None, total_loss=None,
              total_loss_regul=None):
    """
    plot stratified loss components over training iterations.

    creates a two-panel figure showing loss and regularization terms in both
    linear and log scale. saves to {log_dir}/tmp_training/loss.tif.

    Parameters
    -----------
    loss_dict : dict
        dictionary containing loss component lists with keys:
        - 'loss': loss without regularization
        - 'regul_total': total regularization loss
        - 'W_L1': W L1 sparsity penalty
        - 'W_L2': W L2 regularization penalty
        - 'edge_diff': edge monotonicity penalty
        - 'edge_norm': edge normalization
        - 'edge_weight': edge MLP weight regularization
        - 'phi_weight': phi MLP weight regularization
        - 'W_sign': W sign consistency penalty
    log_dir : str
        directory to save the figure
    epoch : int, optional
        current epoch number
    Niter : int, optional
        number of iterations per epoch
    debug : bool, optional
        if True, print debug information about loss components
    current_loss : float, optional
        current iteration total loss (for debug)
    current_regul : float, optional
        current iteration regularization (for debug)
    total_loss : float, optional
        accumulated total loss (for debug)
    total_loss_regul : float, optional
        accumulated regularization loss (for debug)
    """
    if len(loss_dict['loss']) == 0:
        return

    # debug output if requested
    if debug and current_loss is not None and current_regul is not None:
        current_pred_loss = current_loss - current_regul

        # get current iteration component values (last element in each list)
        debug_keys = ['W_L1', 'W_L2', 'edge_diff', 'edge_norm', 'edge_weight', 'phi_weight', 'W_sign']
        comp_sum = sum(loss_dict[k][-1] for k in debug_keys if k in loss_dict)

        print(f"\n=== debug loss components (epoch {epoch}, iter {Niter}) ===")
        print("current iteration:")
        print(f"  loss.item() (total): {current_loss:.6f}")
        print(f"  regul_this_iter: {current_regul:.6f}")
        print(f"  prediction_loss (loss - regul): {current_pred_loss:.6f}")
        print("\nregularization breakdown:")
        for k in debug_keys:
            if k in loss_dict:
                print(f"  {k}: {loss_dict[k][-1]:.6f}")
        print(f"  sum of components: {comp_sum:.6f}")
        if total_loss is not None and total_loss_regul is not None:
            print("\naccumulated (for reference):")
            print(f"  total_loss (accumulated): {total_loss:.6f}")
            print(f"  total_loss_regul (accumulated): {total_loss_regul:.6f}")
        if current_loss > 0:
            print(f"\nratio: regul / loss (current iter) = {current_regul / current_loss:.4f}")
        if current_pred_loss < 0:
            print("\n  warning: negative prediction loss! regul > total loss")
        print("="*60)

    fig_loss, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # build optional components list from LOSS_COLORS (metabolism-specific)
    # plus legacy components for NeuralGraph compatibility
    optional_components = [
        (k, v[0], v[1], v[2]) for k, v in LOSS_COLORS.items() if k != 'loss'
    ] + [
        # legacy NeuralGraph components (for compatibility)
        ('W_L1', 'r', 1.5, 'W L1 sparsity'),
        ('W_L2', 'darkred', 1.5, 'W L2 regul'),
        ('W_sign', 'navy', 1.5, 'W sign (Dale)'),
        ('phi_weight', 'lime', 1.5, 'MLP0 Weight Regul'),
        ('edge_diff', 'magenta', 1.5, 'MLP1 monotonicity'),
        ('edge_norm', 'brown', 1.5, 'MLP1 norm'),
        ('edge_weight', 'pink', 1.5, 'MLP1 weight regul'),
    ]

    # linear scale (main prediction loss in blue)
    loss_color, loss_lw, loss_label = LOSS_COLORS['loss']
    ax1.plot(loss_dict['loss'], color=loss_color, linewidth=loss_lw, label=loss_label, alpha=0.8)
    for key, color, lw, label in optional_components:
        if key in loss_dict:
            ax1.plot(loss_dict[key], color=color, linewidth=lw, label=label, alpha=0.7)
    ax1.set_xlabel('iteration', fontsize=14)
    ax1.set_ylabel('loss', fontsize=14)
    ax1.legend(fontsize=10, loc='best', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)

    # log scale
    ax2.plot(loss_dict['loss'], color=loss_color, linewidth=loss_lw, label=loss_label, alpha=0.8)
    for key, color, lw, label in optional_components:
        if key in loss_dict:
            ax2.plot(loss_dict[key], color=color, linewidth=lw, label=label, alpha=0.7)
    ax2.set_xlabel('iteration', fontsize=14)
    ax2.set_ylabel('loss (log)', fontsize=14)
    ax2.set_yscale('log')
    ax2.legend(fontsize=10, loc='best', ncol=2)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f'{log_dir}/tmp_training/loss.tif', dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
#  metabolism helpers
# ---------------------------------------------------------------------------

def init_reaction(n_metabolites, n_reactions, max_metabolites_per_reaction, device, seed=42,
                  cycle_fraction=0.0, cycle_length=4):
    """build a random sparse stoichiometric matrix as bipartite edge lists.

    for each reaction j:
      - sample 1..max_metabolites_per_reaction substrates  (stoich < 0)
      - sample 1..max_metabolites_per_reaction products    (stoich > 0)
      - substrates and products are disjoint
      - stoichiometric coefficients are integers in {1, 2}

    Optionally adds autocatalytic cycles for oscillatory dynamics:
      - cycle_fraction: fraction of reactions dedicated to cycles (0.0 to 1.0)
      - cycle_length: number of metabolites per cycle (default 4)
      - Creates autocatalytic chains: A + B → 2B, B + C → 2C, C + D → 2D, D + A → 2A
        where each species catalyzes the next, creating nonlinear feedback for oscillations

    Returns
    -------
    stoich_graph : dict with keys
        'sub'  : (met_sub, rxn_sub, sto_sub)   substrate edges (|coeff|)
        'all'  : (met_all, rxn_all, sto_all)   all edges (signed coeff)
    stoich_matrix : Tensor (n_metabolites, n_reactions)  dense S for saving
    """
    rng = np.random.RandomState(seed)

    sub_edges = []   # (metabolite, reaction, |coeff|)
    all_edges = []   # (metabolite, reaction, signed coeff)

    # Calculate number of cycle reactions vs random reactions
    n_cycle_reactions = int(n_reactions * cycle_fraction)
    n_random_reactions = n_reactions - n_cycle_reactions

    # --- Phase 1: Create autocatalytic cycles ---
    # Pattern: A + B → 2B, B + C → 2C, C + D → 2D, D + A → 2A
    # Each species catalyzes the growth of the next, creating oscillatory dynamics
    if n_cycle_reactions > 0:
        n_cycles = n_cycle_reactions // cycle_length
        metabolites_used_in_cycles = set()

        for cycle_idx in range(n_cycles):
            # Select metabolites for this cycle (avoiding reuse for cleaner cycles)
            available = [m for m in range(n_metabolites) if m not in metabolites_used_in_cycles]
            if len(available) < cycle_length:
                # Reuse metabolites if we run out (creates interconnected cycles)
                available = list(range(n_metabolites))

            cycle_mets = rng.choice(available, size=min(cycle_length, len(available)), replace=False)
            metabolites_used_in_cycles.update(cycle_mets)

            # Create autocatalytic cycle: A + B → 2B, B + C → 2C, etc.
            for i in range(len(cycle_mets)):
                rxn_idx = cycle_idx * cycle_length + i
                if rxn_idx >= n_cycle_reactions:
                    break

                # Reaction: m_i + m_{i+1} → 2 * m_{i+1}
                # m_i is consumed (food), m_{i+1} catalyzes and grows
                # Net stoichiometry: food -1, catalyst +1 (column sum = 0)
                food = cycle_mets[i]
                catalyst = cycle_mets[(i + 1) % len(cycle_mets)]

                # Substrates for rate computation: both food and catalyst
                sub_edges.append((int(food), rxn_idx, 1.0))
                sub_edges.append((int(catalyst), rxn_idx, 1.0))
                # Net stoichiometry for concentration update (mass conserved)
                all_edges.append((int(food), rxn_idx, -1.0))      # food consumed
                all_edges.append((int(catalyst), rxn_idx, 1.0))   # catalyst net +1

    # --- Phase 2: Create random reactions (as before) ---
    for j in range(n_cycle_reactions, n_reactions):
        n_participants = min(max_metabolites_per_reaction, n_metabolites // 2)
        # at least 1 substrate and 1 product
        n_sub = rng.randint(1, max(2, n_participants))
        n_prod = rng.randint(1, max(2, n_participants))

        participants = rng.choice(n_metabolites, size=n_sub + n_prod, replace=False)
        substrates = participants[:n_sub]
        products = participants[n_sub:]

        # assign substrate coefficients (random 1 or 2)
        sub_coeffs = [rng.randint(1, 3) for _ in substrates]
        total_consumed = sum(sub_coeffs)

        for m, coeff in zip(substrates, sub_coeffs):
            sub_edges.append((int(m), j, float(coeff)))
            all_edges.append((int(m), j, -float(coeff)))

        # distribute consumed mass among products (mass conservation: column sum = 0)
        # split total_consumed as evenly as possible using integer coefficients
        base_prod = total_consumed // n_prod
        remainder = total_consumed % n_prod
        prod_coeffs = [base_prod + (1 if i < remainder else 0) for i in range(n_prod)]
        # shuffle so the extra +1 isn't always on the first product
        rng.shuffle(prod_coeffs)

        for m, coeff in zip(products, prod_coeffs):
            if coeff > 0:
                all_edges.append((int(m), j, float(coeff)))

    # build tensors
    met_sub = torch.tensor([e[0] for e in sub_edges], dtype=torch.long, device=device)
    rxn_sub = torch.tensor([e[1] for e in sub_edges], dtype=torch.long, device=device)
    sto_sub = torch.tensor([e[2] for e in sub_edges], dtype=torch.float32, device=device)

    met_all = torch.tensor([e[0] for e in all_edges], dtype=torch.long, device=device)
    rxn_all = torch.tensor([e[1] for e in all_edges], dtype=torch.long, device=device)
    sto_all = torch.tensor([e[2] for e in all_edges], dtype=torch.float32, device=device)

    # dense stoichiometric matrix for saving / visualisation
    S = torch.zeros(n_metabolites, n_reactions, dtype=torch.float32, device=device)
    for (m, r, s) in all_edges:
        S[m, r] = s

    stoich_graph = {
        'sub': (met_sub, rxn_sub, sto_sub),
        'all': (met_all, rxn_all, sto_all),
    }

    return stoich_graph, S


def init_concentration(n_metabolites, device, mode='uniform', seed=42, c_min=2.5, c_max=7.5):
    """initialise metabolite concentrations.

    Parameters
    ----------
    n_metabolites : int
    device : torch.device
    mode : str   'uniform' | 'random'
    seed : int
    c_min : float   minimum concentration (for random mode)
    c_max : float   maximum concentration (for random mode)

    Returns
    -------
    concentrations : Tensor of shape (n_metabolites,)
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    if mode == 'random':
        concentrations = torch.rand(n_metabolites, generator=rng, device=device) * (c_max - c_min) + c_min
    else:  # uniform
        concentrations = torch.ones(n_metabolites, device=device) * ((c_max + c_min) / 2)

    return concentrations


def plot_metabolism_concentrations(x_list, n_metabolites, n_frames, dataset_name, delta_t, activity_rank=None):
    """plot metabolite concentration traces -- same layout as activity.png in signal.

    offset waterfall plot with metabolite indices on the left margin.
    saves to graphs_data/{dataset_name}/concentration.png
    """
    print('plot concentration ...')
    conc = x_list[:, :, 3:4].squeeze().T  # (n_met, T)

    # sample 100 traces if needed
    if n_metabolites > 100:
        sampled_indices = np.sort(np.random.choice(n_metabolites, 100, replace=False))
        conc_plot = conc[sampled_indices]
        n_plot = 100
    else:
        conc_plot = conc
        sampled_indices = np.arange(n_metabolites)
        n_plot = n_metabolites

    # offset traces vertically (same spacing logic as signal)
    spacing = np.std(conc_plot) * 3 if np.std(conc_plot) > 0 else 1.0
    conc_plot = conc_plot - spacing * np.arange(n_plot)[:, None] + spacing * n_plot / 2

    plt.figure(figsize=(16, 10))
    plt.plot(conc_plot.T, linewidth=2, alpha=0.7)

    for i in range(0, n_plot, 5):
        plt.text(-100, conc_plot[i, 0], str(sampled_indices[i]),
                 fontsize=10, va='center', ha='right')

    ax = plt.gca()
    ax.text(-n_frames * 0.12, conc_plot.mean(), 'metabolite index',
            fontsize=14, va='center', ha='center', rotation=90)
    plt.xlabel('time (min)', fontsize=14)
    plt.xticks(fontsize=12)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('right')
    ax.set_yticks([])
    plt.xlim([0, min(n_frames, conc_plot.shape[1])])

    # add activity rank in top left
    if activity_rank is not None:
        ax.text(0.02, 0.98, f'activity rank = {activity_rank}',
                transform=ax.transAxes, fontsize=12, va='top', ha='left')

    plt.tight_layout()
    plt.savefig(f'graphs_data/{dataset_name}/concentrations.png', dpi=150)
    plt.close()


def plot_stoichiometric_matrix(S, dataset_name):
    """heatmap of stoichiometric matrix S -- same style as connectivity_matrix.png.

    saves to graphs_data/{dataset_name}/connectivity_matrix.png
    """
    print('plot stoichiometric matrix ...')
    S_np = to_numpy(S) if torch.is_tensor(S) else np.asarray(S)
    n_met, n_rxn = S_np.shape

    vmax = max(np.max(np.abs(S_np)), 1)

    # main heatmap — aspect ratio matches matrix shape
    fig_w = max(8, min(12, n_rxn / n_met * 8))
    plt.figure(figsize=(fig_w, 8))
    ax = sns.heatmap(S_np, center=0, square=False, cmap='bwr',
                     cbar_kws={'fraction': 0.046}, vmin=-vmax, vmax=vmax)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    plt.xticks([0, n_rxn - 1], [1, n_rxn], fontsize=14)
    plt.yticks([0, n_met - 1], [1, n_met], fontsize=14)
    plt.xticks(rotation=0)
    plt.xlabel('reaction', fontsize=14)
    plt.ylabel('metabolite', fontsize=14)

    # zoom inset (top-left corner)
    zoom = min(20, n_met, n_rxn)
    if zoom > 0:
        plt.subplot(2, 2, 1)
        sns.heatmap(S_np[0:zoom, 0:zoom], cbar=False,
                    center=0, square=False, cmap='bwr', vmin=-vmax, vmax=vmax)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(f'graphs_data/{dataset_name}/connectivity_matrix.png', dpi=150)
    plt.close()


def plot_stoichiometric_eigenvalues(S, dataset_name):
    """SVD singular-value spectrum of stoichiometric matrix S.

    S is not square (n_met x n_rxn) so we use SVD instead of eigendecomposition.
    saves to graphs_data/{dataset_name}/eigenvalues.png
    """
    print('plot SVD spectrum of stoichiometric matrix ...')
    S_np = to_numpy(S) if torch.is_tensor(S) else np.asarray(S)

    U, sigma, Vt = np.linalg.svd(S_np, full_matrices=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (0) singular values
    axes[0].scatter(range(len(sigma)), sigma, s=50, c='k', alpha=0.7, edgecolors=None)
    axes[0].set_xlabel('index', fontsize=14)
    axes[0].set_ylabel('singular value', fontsize=14)
    axes[0].tick_params(labelsize=12)
    axes[0].text(0.05, 0.95, f'rank: {np.sum(sigma > 1e-6)}',
                 transform=axes[0].transAxes, fontsize=12, verticalalignment='top')

    # (1) singular values (log scale)
    axes[1].plot(sigma, c='k', linewidth=2)
    axes[1].set_xlabel('index', fontsize=14)
    axes[1].set_ylabel('singular value (log)', fontsize=14)
    axes[1].set_yscale('log')
    axes[1].tick_params(labelsize=12)

    # (2) cumulative variance explained
    cumvar = np.cumsum(sigma ** 2) / np.sum(sigma ** 2)
    rank_90 = np.searchsorted(cumvar, 0.90) + 1
    rank_99 = np.searchsorted(cumvar, 0.99) + 1
    axes[2].plot(cumvar, c='k', linewidth=2)
    axes[2].axhline(y=0.9, color='gray', linestyle='--', linewidth=1)
    axes[2].axhline(y=0.99, color='gray', linestyle=':', linewidth=1)
    axes[2].set_xlabel('index', fontsize=14)
    axes[2].set_ylabel('cumulative variance', fontsize=14)
    axes[2].tick_params(labelsize=12)
    axes[2].text(0.5, 0.5, f'rank(90%): {rank_90}\nrank(99%): {rank_99}',
                 transform=axes[2].transAxes, fontsize=12, verticalalignment='center')

    plt.tight_layout()
    plt.savefig(f'graphs_data/{dataset_name}/eigenvalues.png', dpi=150)
    plt.close()

    print(f'  SVD rank: {np.sum(sigma > 1e-6)}, rank(90%): {rank_90}, rank(99%): {rank_99}')


def plot_rate_distribution(model, dataset_name):
    """histogram of per-reaction rate constants k_j.

    plots both log10(k_j) distribution and the raw k_j distribution.
    saves to graphs_data/{dataset_name}/rate_distribution.png
    """
    print('plot rate distribution ...')
    log_k = to_numpy(model.log_k.detach())
    k = 10.0 ** log_k

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # log10(k_j) histogram
    ax1.hist(log_k, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
    ax1.set_xlabel('log$_{10}$(k$_j$)', fontsize=14)
    ax1.set_ylabel('count', fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.axvline(x=np.median(log_k), color='red', linestyle='--', linewidth=2,
                label=f'median = {np.median(k):.3f}')
    ax1.legend(fontsize=10)

    # raw k_j histogram (log x-axis)
    ax2.hist(k, bins=np.logspace(np.log10(k.min()), np.log10(k.max()), 20),
             color='coral', edgecolor='black', alpha=0.8)
    ax2.set_xscale('log')
    ax2.set_xlabel('k$_j$', fontsize=14)
    ax2.set_ylabel('count', fontsize=14)
    ax2.tick_params(labelsize=12)
    ax2.axvline(x=np.median(k), color='red', linestyle='--', linewidth=2)

    plt.tight_layout()
    plt.savefig(f'graphs_data/{dataset_name}/rate_distribution.png', dpi=150)
    plt.close()
    print(f'  k range: [{k.min():.4f}, {k.max():.4f}], median: {np.median(k):.4f}')


def plot_homeostasis_function(model, x_list, dataset_name, colormap='tab10'):
    """plot per-type homeostasis function: -λ_type * (c - c_baseline_type).

    For each metabolite type, plots the homeostasis term as a function of concentration.
    Includes a small concentration histogram in the top-right corner.
    Saves to graphs_data/{dataset_name}/homeostasis_function.png
    """
    if not hasattr(model, 'p') or model.p.shape[0] < 1:
        return

    n_types = model.p.shape[0]
    p = to_numpy(model.p.detach().cpu())

    # concentration range based on data
    conc_data = x_list[:, :, 3].flatten()
    c_min = np.min(conc_data) * 0.5
    c_max = np.max(conc_data) * 1.5
    c_range = np.linspace(c_min, c_max, 200)

    cmap = plt.cm.get_cmap(colormap)

    fig, ax = plt.subplots(figsize=(8, 8))

    for t in range(n_types):
        lambda_t = p[t, 0]
        c_baseline_t = p[t, 1]
        homeostasis = -lambda_t * (c_range - c_baseline_t)
        ax.plot(c_range, homeostasis, linewidth=2, color=cmap(t),
                label=f'type {t}: λ={lambda_t:.3f}, c₀={c_baseline_t:.2f}')
        # mark baseline
        ax.axvline(x=c_baseline_t, color=cmap(t), linestyle='--', alpha=0.5)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('concentration $c$', fontsize=14)
    ax.set_ylabel(r'$-\lambda (c - c^{\mathrm{baseline}})$', fontsize=14)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

    # add concentration histogram as inset in top-right (smaller)
    ax_inset = fig.add_axes([0.72, 0.72, 0.15, 0.15])
    ax_inset.hist(conc_data, bins=30, color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax_inset.set_xlabel('c', fontsize=7)
    ax_inset.set_ylabel('count', fontsize=7)
    ax_inset.tick_params(labelsize=6)

    plt.savefig(f'graphs_data/{dataset_name}/homeostasis_function.png', dpi=150)
    plt.close()
    print(f'  saved graphs_data/{dataset_name}/homeostasis_function.png')


def plot_metabolism_mlp_functions(model, x_list, dataset_name, device):
    """plot ground-truth substrate_func and rate_func functions for the metabolism generator.

    analogous to plot_synaptic_mlp_functions for signal models (MLP0/MLP1).

    saves:
      graphs_data/{dataset_name}/substrate_function.png
      graphs_data/{dataset_name}/rate_function.png
    """
    print('plot substrate_func and rate_func functions ...')
    import torch
    from torch_geometric.data import Data as pyg_Data

    n_pts = 500
    folder = f'graphs_data/{dataset_name}'

    # --- substrate_func: sweep concentration at fixed |stoich| values ---
    # use std-based range (like signal model) so Tanh region is visible
    conc_data = x_list[:, :, 3]
    conc_std = max(np.std(conc_data), 1e-6)
    conc_max = conc_std * 3.0

    conc_range = torch.linspace(0, conc_max, n_pts, device=device)

    stoich_values = [1, 2]
    colors = ['tab:blue', 'tab:orange']

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # true power laws c^s
    c_np = to_numpy(conc_range)
    for s_val, color in zip(stoich_values, colors):
        true_power = np.power(c_np + 1e-8, s_val)
        ax.plot(c_np, true_power, linewidth=2, color=color, label=f'$c^{{{s_val}}}$')

    ax.set_xlabel('concentration $c$', fontsize=14)
    ax.set_ylabel(r'$c^{|s|}$', fontsize=14)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f'{folder}/substrate_function.png', dpi=150)
    plt.close()

    # --- rate_func: compute actual h_rxn from a data frame, plot rate vs ||h_rxn|| ---
    # pick a frame near the middle of the simulation
    mid_frame = x_list.shape[0] // 2
    x = torch.tensor(x_list[mid_frame], dtype=torch.float32, device=device)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    with torch.no_grad():
        concentrations = x[:, 3]
        x_src = concentrations[model.met_sub].unsqueeze(-1)
        s_abs = model.sto_sub.unsqueeze(-1)
        msg_in = torch.cat([x_src, s_abs], dim=-1)
        msg = model.substrate_func(msg_in)

        h_rxn = torch.zeros(
            model.n_rxn, msg.shape[1], dtype=msg.dtype, device=msg.device
        )
        h_rxn.index_add_(0, model.rxn_sub, msg)

        base_rate = model.rate_func(h_rxn).squeeze(-1)
        k = torch.pow(10.0, model.log_k)
        full_rate = k * base_rate
        h_norm = h_rxn.norm(dim=-1)

    # left panel: base rate (before k scaling) vs ||h_rxn||
    axes[0].scatter(to_numpy(h_norm), to_numpy(base_rate), s=12, c='k', alpha=0.5, edgecolors=None)
    axes[0].set_xlabel(r'$\|h_{rxn}\|$', fontsize=14)
    axes[0].set_ylabel(r'rate\_func($h$)', fontsize=14)
    axes[0].tick_params(labelsize=12)

    # right panel: full rate (k * base) vs ||h_rxn||, colored by log_k
    sc = axes[1].scatter(to_numpy(h_norm), to_numpy(full_rate), s=12,
                         c=to_numpy(model.log_k.detach()), cmap='coolwarm', alpha=0.6, edgecolors=None)
    plt.colorbar(sc, ax=axes[1], label=r'$\log_{10}(k_j)$')
    axes[1].set_xlabel(r'$\|h_{rxn}\|$', fontsize=14)
    axes[1].set_ylabel(r'$k_j \cdot$ rate\_func($h$)', fontsize=14)
    axes[1].tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f'{folder}/rate_function.png', dpi=150)
    plt.close()
    print(f'  saved {folder}/substrate_function.png and {folder}/rate_function.png')


def plot_metabolism_kinograph(x_list, n_metabolites, n_frames, dataset_name, delta_t, c_center=None, c_range=1.0):
    """kinograph (imshow heatmap) of metabolite concentrations over time.

    rows = metabolites, columns = time frames.
    saves to graphs_data/{dataset_name}/kinograph.png

    Parameters
    ----------
    c_center : float or None
        Center of colormap range. If None, uses data-driven range [0, max].
    c_range : float
        Half-width of colormap range around c_center (i.e., vmin=c_center-c_range, vmax=c_center+c_range).
    """
    print('plot kinograph ...')
    conc = x_list[:, :, 3].T  # (n_met, T)

    n_frames_plot = conc.shape[1]

    if c_center is not None:
        vmin = c_center - c_range
        vmax = c_center + c_range
    else:
        vmin = 0
        vmax = np.abs(conc).max()

    plt.figure(figsize=(12, 8))
    plt.imshow(conc, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, origin='lower', interpolation='nearest')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('concentration', fontsize=14)
    plt.ylabel('metabolite', fontsize=14)
    plt.xlabel('time (min)', fontsize=14)
    plt.xticks([0, n_frames_plot - 1], [0, int(n_frames_plot * delta_t)], fontsize=12)
    plt.yticks([0, n_metabolites - 1], [1, n_metabolites], fontsize=12)
    plt.tight_layout()
    plt.savefig(f'graphs_data/{dataset_name}/kinograph.png', dpi=150)
    plt.close()


def plot_metabolism_external_input_kinograph(x_list, n_metabolites, n_frames, dataset_name, delta_t):
    """kinograph of external input (x[:, 4]) over time.

    rows = metabolites, columns = time frames.
    saves to graphs_data/{dataset_name}/kinograph_external_input.png
    """
    print('plot external input kinograph ...')
    ext_input = x_list[:, :, 4].T  # (n_met, T)

    n_frames_plot = ext_input.shape[1]

    plt.figure(figsize=(12, 8))
    plt.imshow(ext_input, aspect='auto', cmap='viridis', vmin=ext_input.min(), vmax=ext_input.max(), origin='lower', interpolation='nearest')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('external input', fontsize=14)
    plt.ylabel('metabolite', fontsize=14)
    plt.xlabel('time (min)', fontsize=14)
    plt.xticks([0, n_frames_plot - 1], [0, int(n_frames_plot * delta_t)], fontsize=12)
    plt.yticks([0, n_metabolites - 1], [1, n_metabolites], fontsize=12)
    plt.tight_layout()
    plt.savefig(f'graphs_data/{dataset_name}/kinograph_external_input.png', dpi=150)
    plt.close()
