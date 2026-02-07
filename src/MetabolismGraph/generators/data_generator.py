
import os

import numpy as np
import torch
from torch_geometric import data
from tqdm import trange

from MetabolismGraph.utils import to_numpy, get_equidistant_points
from MetabolismGraph.generators.utils import (
    init_reaction,
    init_concentration,
    plot_stoichiometric_matrix,
    plot_stoichiometric_eigenvalues,
    plot_rate_distribution,
    plot_metabolism_concentrations,
    plot_metabolism_kinograph,
    plot_metabolism_external_input_kinograph,
    plot_metabolism_mlp_functions,
    plot_homeostasis_function,
)


def data_generate(
    config,
    visualize=True,
    device=None,
    bSave=True,
):
    """generate synthetic metabolic dynamics data.

    builds a random stoichiometric matrix S, initialises metabolite
    concentrations, and integrates the PDE_M1 ODE forward in time using
    Euler steps.  saves x_list, y_list, and stoichiometric data.
    """

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(simulation_config.seed)
    np.random.seed(simulation_config.seed)

    dataset_name = config.dataset
    n_metabolites = simulation_config.n_metabolites
    n_reactions = simulation_config.n_reactions
    max_met_per_rxn = simulation_config.max_metabolites_per_reaction
    n_frames = simulation_config.n_frames
    delta_t = simulation_config.delta_t
    noise_model_level = simulation_config.noise_model_level
    measurement_noise_level = training_config.measurement_noise_level

    folder = f'./graphs_data/{dataset_name}/'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'{folder}/Fig/', exist_ok=True)

    # --- build stoichiometric graph ---
    cycle_fraction = getattr(simulation_config, 'cycle_fraction', 0.0)
    cycle_length = getattr(simulation_config, 'cycle_length', 4)
    stoich_graph, S = init_reaction(
        n_metabolites, n_reactions, max_met_per_rxn, device, seed=simulation_config.seed,
        cycle_fraction=cycle_fraction, cycle_length=cycle_length,
    )

    # --- initial concentrations ---
    c_min = getattr(simulation_config, 'concentration_min', 2.5)
    c_max = getattr(simulation_config, 'concentration_max', 7.5)
    concentrations = init_concentration(n_metabolites, device, mode='random', seed=simulation_config.seed, c_min=c_min, c_max=c_max)

    # --- positions for visualisation ---
    xc, yc = get_equidistant_points(n_points=n_metabolites)
    pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2

    # --- build model ---
    if "PDE_M2" in model_config.model_name:
        from MetabolismGraph.generators.PDE_M2 import PDE_M2
        model = PDE_M2(config=config, stoich_graph=stoich_graph, device=device)
    else:
        from MetabolismGraph.generators.PDE_M1 import PDE_M1
        model = PDE_M1(config=config, stoich_graph=stoich_graph, device=device)
    model.to(device)

    # --- external input (movie-driven enzyme modulation) ---
    external_input_type = getattr(simulation_config, 'external_input_type', 'none')
    has_visual_input = "visual" in external_input_type
    if has_visual_input:
        from imageio.v3 import imread
        im = imread(f"graphs_data/{simulation_config.node_value_map}").astype(np.float32)
        # normalize to [0, 1] if not already
        im_min, im_max = im.min(), im.max()
        if im_max > 1.0:
            im = (im - im_min) / (im_max - im_min + 1e-8)
        n_input_metabolites = min(simulation_config.n_input_metabolites, n_metabolites)
        # pre-compute 2D grid indices to spatially downsample image -> metabolites
        im_h, im_w = im[0].squeeze().shape[:2]
        n_side = int(np.sqrt(n_input_metabolites))
        im_rows = np.linspace(0, im_h - 1, n_side, dtype=int)
        im_cols = np.linspace(0, im_w - 1, n_side, dtype=int)

    S_np = to_numpy(S)

    # --- save stoichiometric data ---
    if bSave:
        torch.save(S, f'{folder}/stoichiometry.pt')
        torch.save(stoich_graph, f'{folder}/stoich_graph.pt')
        torch.save(model.state_dict(), f'{folder}/gt_model.pt')

    # --- plots: stoichiometric matrix + SVD + rates ---
    plot_stoichiometric_matrix(S, dataset_name)
    plot_stoichiometric_eigenvalues(S, dataset_name)
    plot_rate_distribution(model, dataset_name)

    # --- x tensor: 8-column layout ---
    x = torch.zeros((n_metabolites, 8), dtype=torch.float32, device=device)
    x[:, 0] = torch.arange(n_metabolites, dtype=torch.float32, device=device)
    x[:, 1:3] = pos.clone().detach()
    x[:, 3] = concentrations.clone().detach()

    # assign metabolite types (evenly distributed)
    n_metabolite_types = getattr(simulation_config, 'n_metabolite_types', 1)
    metabolite_type = torch.zeros(n_metabolites, dtype=torch.float32, device=device)
    for t in range(n_metabolite_types):
        start_idx = t * (n_metabolites // n_metabolite_types)
        end_idx = (t + 1) * (n_metabolites // n_metabolite_types) if t < n_metabolite_types - 1 else n_metabolites
        metabolite_type[start_idx:end_idx] = t
    x[:, 6] = metabolite_type

    # set per-type parameters if multiple types
    if n_metabolite_types > 1 and hasattr(model, 'p'):
        sim_cfg = config.simulation
        lambda_per_type = getattr(sim_cfg, 'homeostatic_lambda_per_type', None)
        baseline_per_type = getattr(sim_cfg, 'homeostatic_baseline_per_type', None)

        with torch.no_grad():
            # use per-type config if provided, otherwise randomize
            if lambda_per_type is not None and len(lambda_per_type) == n_metabolite_types:
                model.p[:, 0] = torch.tensor(lambda_per_type, dtype=model.p.dtype)
            else:
                # vary lambda: uniform in [0.5*base, 1.5*base]
                base_lambda = model.p[0, 0].item()
                if base_lambda > 0:
                    model.p[:, 0] = base_lambda * (0.5 + torch.rand(n_metabolite_types))

            if baseline_per_type is not None and len(baseline_per_type) == n_metabolite_types:
                model.p[:, 1] = torch.tensor(baseline_per_type, dtype=model.p.dtype)
            else:
                # vary c_baseline: uniform in [0.8*base, 1.2*base]
                base_c = model.p[0, 1].item()
                model.p[:, 1] = base_c * (0.8 + 0.4 * torch.rand(n_metabolite_types))

        print(f'metabolite types: {n_metabolite_types}')
        print(f'  lambda per type: {to_numpy(model.p[:, 0])}')
        print(f'  c_baseline per type: {to_numpy(model.p[:, 1])}')

    # --- Euler integration ---
    for run in range(training_config.n_runs):
        x_list = []
        y_list = []

        # reset concentrations per run
        x[:, 3] = concentrations.clone().detach()

        for it in trange(simulation_config.start_frame, n_frames + 1, ncols=150):

            # update external input from movie (2D grid subsample)
            if has_visual_input:
                im_idx = int(it % im.shape[0] - 1)
                im_ = im[im_idx].squeeze()
                im_ = np.rot90(im_, 3)
                im_down = im_[np.ix_(im_rows, im_cols)].flatten()[:n_input_metabolites]
                x[:n_input_metabolites, 4] = torch.tensor(
                    im_down, dtype=torch.float32, device=device
                )

            with torch.no_grad():
                dataset = data.Data(x=x, pos=x[:, 1:3])
                y = model(dataset, dt=delta_t)

            if (it >= 0) and bSave:
                x_list.append(to_numpy(x.clone()))
                y_list.append(to_numpy(y.clone()))

            # Euler step (flux-limited rates guarantee no substrate over-consumption)
            du = y.squeeze()
            x[:, 3] = x[:, 3] + du * delta_t
            # safety clamp (should rarely trigger with flux limiting)
            n_clamped = (x[:, 3] < 0).sum().item()
            x[:, 3] = torch.clamp(x[:, 3], min=0.0)

            if noise_model_level > 0:
                x[:, 3] = torch.clamp(
                    x[:, 3] + torch.randn(n_metabolites, device=device) * noise_model_level,
                    min=0.0,
                )

        if bSave:
            x_list = np.array(x_list)
            y_list = np.array(y_list)

            if measurement_noise_level > 0:
                np.save(f'{folder}/raw_x_list_{run}.npy', x_list)
                np.save(f'{folder}/raw_y_list_{run}.npy', y_list)
                for k in range(x_list.shape[0]):
                    x_list[k, :, 3] = np.maximum(
                        x_list[k, :, 3] + np.random.normal(0, measurement_noise_level, x_list.shape[1]),
                        0.0,
                    )
                for k in range(1, x_list.shape[0] - 1):
                    y_list[k] = (x_list[k + 1, :, 3:4] - x_list[k, :, 3:4]) / delta_t

            np.save(f'{folder}/x_list_{run}.npy', x_list)
            np.save(f'{folder}/y_list_{run}.npy', y_list)

        # --- activity plot + SVD analysis (first run) ---
        if run == 0:
            # compute SVD analysis first to get activity rank
            from MetabolismGraph.models.utils import analyze_data_svd
            svd_results = analyze_data_svd(x_list, folder, config=config, save_in_subfolder=False)
            activity_rank = svd_results.get('activity', {}).get('rank_99', None)

            plot_metabolism_concentrations(x_list, n_metabolites, n_frames, dataset_name, delta_t, activity_rank=activity_rank)
            # use data-driven vmin/vmax (no fixed c_center)
            plot_metabolism_kinograph(x_list, n_metabolites, n_frames, dataset_name, delta_t, c_center=None, c_range=1.0)
            if has_visual_input:
                plot_metabolism_external_input_kinograph(x_list, n_metabolites, n_frames, dataset_name, delta_t)

            plot_metabolism_mlp_functions(model, x_list, dataset_name, device)

            # plot per-type homeostasis functions
            if n_metabolite_types > 1:
                plot_homeostasis_function(model, x_list, dataset_name)

    torch.save(model.p, f'{folder}/model_p_0.pt')
