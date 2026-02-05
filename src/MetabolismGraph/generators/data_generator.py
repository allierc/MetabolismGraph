
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

    print(f'generating metabolism data ... {n_metabolites} metabolites {n_reactions} reactions')

    folder = f'./graphs_data/{dataset_name}/'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'{folder}/Fig/', exist_ok=True)

    # --- build stoichiometric graph ---
    stoich_graph, S = init_reaction(
        n_metabolites, n_reactions, max_met_per_rxn, device, seed=simulation_config.seed,
    )

    # --- initial concentrations ---
    concentrations = init_concentration(n_metabolites, device, mode='random', seed=simulation_config.seed)

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
        print(f'external input: {simulation_config.node_value_map} shape={im.shape}')

    S_np = to_numpy(S)

    # --- save stoichiometric data ---
    if bSave:
        torch.save(S, f'{folder}/stoichiometry.pt')
        torch.save(stoich_graph, f'{folder}/stoich_graph.pt')

    # --- plots: stoichiometric matrix + SVD + rates ---
    plot_stoichiometric_matrix(S, dataset_name)
    plot_stoichiometric_eigenvalues(S, dataset_name)
    plot_rate_distribution(model, dataset_name)

    # --- x tensor: 8-column layout ---
    x = torch.zeros((n_metabolites, 8), dtype=torch.float32, device=device)
    x[:, 0] = torch.arange(n_metabolites, dtype=torch.float32, device=device)
    x[:, 1:3] = pos.clone().detach()
    x[:, 3] = concentrations.clone().detach()
    x[:, 6] = 0  # metabolite type (single type for now)

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

        print(f'run {run}: generated {x_list.shape[0]} frames')

        # --- start vs end concentration check ---
        conc_start = x_list[0, :, 3]
        conc_end = x_list[-1, :, 3]
        delta = conc_end - conc_start
        n_changed = np.sum(np.abs(delta) > 1e-6)
        n_increased = np.sum(delta > 1e-6)
        n_decreased = np.sum(delta < -1e-6)
        n_unchanged = n_metabolites - n_changed
        print(f'  concentration change (start vs end):')
        print(f'    changed: {n_changed}/{n_metabolites} (increased: {n_increased}, decreased: {n_decreased}, unchanged: {n_unchanged})')
        print(f'    start: mean={conc_start.mean():.4f} std={conc_start.std():.4f} min={conc_start.min():.4f} max={conc_start.max():.4f}')
        print(f'    end:   mean={conc_end.mean():.4f} std={conc_end.std():.4f} min={conc_end.min():.4f} max={conc_end.max():.4f}')
        print(f'    delta: mean={delta.mean():.4f} std={delta.std():.4f} min={delta.min():.4f} max={delta.max():.4f}')

        # --- activity plot + SVD analysis (first run) ---
        if run == 0:
            plot_metabolism_concentrations(x_list, n_metabolites, n_frames, dataset_name, delta_t)
            plot_metabolism_kinograph(x_list, n_metabolites, n_frames, dataset_name, delta_t)
            if has_visual_input:
                plot_metabolism_external_input_kinograph(x_list, n_metabolites, n_frames, dataset_name, delta_t)

            from MetabolismGraph.models.utils import analyze_data_svd
            analyze_data_svd(x_list, folder, config=config, save_in_subfolder=False)

            plot_metabolism_mlp_functions(model, x_list, dataset_name, device)

    torch.save(model.p, f'{folder}/model_p_0.pt')
    print('data saved')
