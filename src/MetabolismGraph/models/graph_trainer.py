import os
import time
import glob
import shutil
import warnings
import logging

# suppress matplotlib/PDF warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*')
warnings.filterwarnings('ignore', message='.*Missing.*')

# suppress fontTools logging (PDF font subsetting messages)
logging.getLogger('fontTools').setLevel(logging.ERROR)
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import numpy as np
import random
import copy

from scipy.optimize import curve_fit

from tqdm import tqdm, trange

import torch_geometric as pyg
from torch_geometric.data import Data as pyg_Data
from torch_geometric.loader import DataLoader

from MetabolismGraph.utils import (
    to_numpy,
    create_log_dir,
    check_and_clear_memory,
    sort_key,
    fig_init,
    load_simulation_data,
    linear_model,
)
from MetabolismGraph.generators.utils import plot_loss
from MetabolismGraph.models.utils import (
    choose_inr_model,
    analyze_data_svd,
    LossRegularizer,
)


def data_train(config, erase, best_model, device, log_file=None, style='color'):
    """dispatcher that calls data_train_metabolism() directly."""
    return data_train_metabolism(config, erase, best_model, device, log_file=log_file, style=style)


def data_test(config, best_model=20, n_rollout_frames=300, device=None, log_file=None,
              visualize=False, verbose=False, run=0):
    """dispatcher that calls data_test_metabolism() directly."""
    return data_test_metabolism(config, best_model=best_model,
                                n_rollout_frames=n_rollout_frames,
                                device=device, log_file=log_file)


def data_train_metabolism(config, erase, best_model, device, log_file=None, style='color'):
    """train a model to recover stoichiometric weights and external modulation.

    combines the edge-weight recovery approach of data_train_flyvis with the
    SIREN-based external input learning of data_train_signal.

    the training model (Metabolism_Propagation) mirrors PDE_M2 but has learnable
    stoichiometric coefficients (sto_all) instead of fixed buffers.
    """
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    dataset_name = config.dataset
    n_epochs = train_config.n_epochs
    n_runs = train_config.n_runs
    n_frames = simulation_config.n_frames
    n_metabolites = simulation_config.n_metabolites
    n_input_metabolites = simulation_config.n_input_metabolites
    delta_t = simulation_config.delta_t
    data_augmentation_loop = train_config.data_augmentation_loop
    batch_size = train_config.batch_size
    time_step = train_config.time_step
    recurrent_training = train_config.recurrent_training
    noise_recurrent_level = train_config.noise_recurrent_level

    external_input_type = simulation_config.external_input_type
    learn_external_input = train_config.learn_external_input
    field_type = model_config.field_type

    has_visual_field = 'visual' in field_type

    log_dir, logger = create_log_dir(config, erase)

    # --- load data and move to GPU ---
    x_list = []
    y_list = []
    for run in trange(0, n_runs, ncols=50):
        x = load_simulation_data(f'graphs_data/{dataset_name}/x_list_{run}')
        y = load_simulation_data(f'graphs_data/{dataset_name}/y_list_{run}')
        # pre-load to GPU for faster training
        x_list.append(torch.tensor(x, dtype=torch.float32, device=device))
        y_list.append(torch.tensor(y, dtype=torch.float32, device=device))

    print(f'dataset: {len(x_list)} run, {len(x_list[0])} frames')
    print(f'data pre-loaded to GPU: {x_list[0].shape[0] * x_list[0].shape[1] * x_list[0].shape[2] * 4 * n_runs / 1024 / 1024:.1f} MB')

    # --- normalization ---
    concentration = x_list[0][:, :, 3:4].squeeze()  # already on GPU
    distrib = concentration.flatten()
    valid_distrib = distrib[~torch.isnan(distrib)]
    if len(valid_distrib) > 0:
        xnorm = 1.5 * torch.std(valid_distrib)
    else:
        xnorm = torch.tensor(1.0, device=device)
    torch.save(xnorm, os.path.join(log_dir, 'xnorm.pt'))
    ynorm = torch.tensor(1.0, device=device)
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    print(f'xnorm: {to_numpy(xnorm)}')
    logger.info(f'xnorm: {to_numpy(xnorm)}')

    # --- variance-weighted sampling (precompute probabilities) ---
    variance_weighted_sampling = getattr(train_config, 'variance_weighted_sampling', False)
    sampling_probs = []
    if variance_weighted_sampling:
        print('computing variance-weighted sampling probabilities...')
        for run in range(n_runs):
            # compute variance of dc/dt across metabolites at each timepoint
            y_var = torch.var(y_list[run], dim=1)  # (n_frames,)
            # add small epsilon to avoid zero weights
            weights = y_var + 1e-8
            # normalize to probability distribution (only valid frames)
            valid_range = n_frames - 4 - time_step
            probs = weights[:valid_range] / weights[:valid_range].sum()
            sampling_probs.append(probs)
        print(f'variance-weighted sampling enabled (80% weighted, 20% uniform)')
        logger.info('variance_weighted_sampling: True')

    # --- load stoichiometric graph ---
    stoich_graph = torch.load(
        f'graphs_data/{dataset_name}/stoich_graph.pt', map_location=device
    )
    gt_S = torch.load(
        f'graphs_data/{dataset_name}/stoichiometry.pt', map_location=device
    )
    print(f'stoichiometric matrix: {gt_S.shape}')
    logger.info(f'stoichiometric matrix: {gt_S.shape}')

    # --- load ground-truth generator model (for MLP comparison plots) ---
    gt_model = None
    gt_model_path = f'graphs_data/{dataset_name}/gt_model.pt'
    if os.path.exists(gt_model_path):
        # infer per-MLP architecture from checkpoint to avoid size mismatch
        gt_state = torch.load(gt_model_path, map_location=device)
        from copy import deepcopy
        gt_config = deepcopy(config)
        # infer hidden_dim and n_layers for each MLP from saved weights
        gt_config.graph_model.hidden_dim_sub = gt_state.get('substrate_func.0.weight', torch.zeros(64, 2)).shape[0]
        gt_config.graph_model.n_layers_sub = sum(1 for k in gt_state if k.startswith('substrate_func.') and k.endswith('.weight'))
        if 'rate_func.0.weight' in gt_state:
            gt_config.graph_model.hidden_dim_node = gt_state['rate_func.0.weight'].shape[0]
        gt_config.graph_model.n_layers_node = sum(1 for k in gt_state if k.startswith('rate_func.') and k.endswith('.weight'))
        if "PDE_M2" in config.graph_model.model_name:
            from MetabolismGraph.generators.PDE_M2 import PDE_M2
            gt_model = PDE_M2(config=gt_config, stoich_graph=stoich_graph, device=device)
        else:
            from MetabolismGraph.generators.PDE_M1 import PDE_M1
            gt_model = PDE_M1(config=gt_config, stoich_graph=stoich_graph, device=device)
        gt_model.load_state_dict(gt_state)
        gt_model.to(device)
        gt_model.eval()

    # --- create training model ---
    from MetabolismGraph.models.Metabolism_Propagation import Metabolism_Propagation
    model = Metabolism_Propagation(config=config, device=device)
    model.load_stoich_graph(stoich_graph)
    model = model.to(device)

    # --- freeze stoichiometry if configured (S given case) ---
    freeze_stoichiometry = getattr(train_config, 'freeze_stoichiometry', False)
    if freeze_stoichiometry:
        # initialize sto_all from ground truth
        gt_sto_all = stoich_graph['all'][2]
        with torch.no_grad():
            model.sto_all.data.copy_(gt_sto_all)
        model.sto_all.requires_grad = False
        print('stoichiometry frozen (S given mode)')
        logger.info('freeze_stoichiometry: True (S given mode)')

    # --- freeze embeddings if training_single_type (single metabolite type) ---
    training_single_type = getattr(train_config, 'training_single_type', False)
    if training_single_type:
        # set all embeddings to same value (single type)
        with torch.no_grad():
            model.a.data.zero_()
        model.a.requires_grad = False
        print('embeddings frozen (single type mode)')
        logger.info('training_single_type: True (embeddings frozen)')

    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total parameters: {n_total_params:,}')
    logger.info(f'total parameters: {n_total_params:,}')

    # --- SIREN for external input (optional) ---
    model_f = choose_inr_model(
        config=config, n_metabolites=n_metabolites, n_frames=n_frames,
        x_list=x_list, device=device,
    )
    optimizer_f = None
    if model_f is not None:
        omega_params = [
            (name, p) for name, p in model_f.named_parameters() if 'omega' in name
        ]
        other_params = [
            p for name, p in model_f.named_parameters() if 'omega' not in name
        ]
        if omega_params:
            print(f"model_f omega parameters: {[n for n, _ in omega_params]}")
            optimizer_f = torch.optim.Adam([
                {'params': other_params, 'lr': train_config.learning_rate_NNR_f},
                {'params': [p for _, p in omega_params],
                 'lr': getattr(train_config, 'learning_rate_omega_f', 0.0001)},
            ])
        else:
            optimizer_f = torch.optim.Adam(
                model_f.parameters(), lr=train_config.learning_rate_NNR_f
            )
        model_f.train()
        print(f'model_f ({type(model_f).__name__}): '
              f'{sum(p.numel() for p in model_f.parameters()):,} params')

    # --- load best model if resuming ---
    start_epoch = 0
    if best_model and best_model not in ('', 'None'):
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        print(f'loading state_dict from {net} ...')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'state_dict loaded, start_epoch={start_epoch}')

    # --- optimizer (custom param groups for metabolism) ---
    lr = train_config.learning_rate_start
    lr_S = train_config.learning_rate_S_start
    # separate learning rates for model components (fall back to lr if not specified)
    lr_k = getattr(train_config, 'learning_rate_k', lr)
    lr_node = getattr(train_config, 'learning_rate_node', lr)
    lr_sub = getattr(train_config, 'learning_rate_sub', lr)

    # separate parameters into groups: k, MLP_node, MLP_sub, stoichiometry
    k_params = []       # log_k (rate constants)
    node_params = []    # MLP_node (node_func + embeddings a)
    sub_params = []     # MLP_sub (substrate_func)
    stoich_params = []  # sto_all (stoichiometry)
    other_params = []   # anything else

    for name, p in model.named_parameters():
        if 'sto_' in name:
            if not freeze_stoichiometry:
                stoich_params.append(p)
        elif 'NNR_f' in name:
            continue  # handled by optimizer_f
        elif 'log_k' in name:
            k_params.append(p)
        elif 'node_func' in name or name == 'a':
            node_params.append(p)
        elif 'substrate_func' in name:
            sub_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if k_params:
        param_groups.append({'params': k_params, 'lr': lr_k, 'name': 'k'})
    if node_params:
        param_groups.append({'params': node_params, 'lr': lr_node, 'name': 'MLP_node'})
    if sub_params:
        param_groups.append({'params': sub_params, 'lr': lr_sub, 'name': 'MLP_sub'})
    if other_params:
        param_groups.append({'params': other_params, 'lr': lr, 'name': 'other'})
    if stoich_params and lr_S > 0:
        param_groups.append({'params': stoich_params, 'lr': lr_S, 'name': 'S'})
    elif stoich_params:
        # lr_S=0 means stoichiometry learns at same rate as MLPs
        param_groups[0]['params'].extend(stoich_params)

    optimizer = torch.optim.Adam(param_groups)
    n_total_params = sum(p.numel() for g in param_groups for p in g['params'])
    model.train()

    # --- S regularization coefficients ---
    coeff_S_L1 = train_config.coeff_S_L1
    coeff_S_integer = getattr(train_config, 'coeff_S_integer', 0.0)
    coeff_S_mass = train_config.coeff_mass_conservation
    n_epochs_init = getattr(train_config, 'n_epochs_init', 0)
    first_coeff_L1 = getattr(train_config, 'first_coeff_L1', coeff_S_L1)

    # --- MLP regularization coefficients ---
    coeff_MLP_sub_diff = getattr(train_config, 'coeff_MLP_sub_diff', 100.0)
    coeff_MLP_node_L1 = getattr(train_config, 'coeff_MLP_node_L1', 0.0)
    coeff_MLP_sub_norm = getattr(train_config, 'coeff_MLP_sub_norm', 0.0)
    coeff_k_floor = getattr(train_config, 'coeff_k_floor', 0.0)
    k_floor_threshold = getattr(train_config, 'k_floor_threshold', -3.0)

    # pre-compute per-reaction scatter indices for mass conservation penalty
    rxn_all = model.rxn_all

    print(f'learning rates: lr_k={lr_k}, lr_node={lr_node}, lr_sub={lr_sub}, lr_S={lr_S}')
    for g in param_groups:
        n_params = sum(p.numel() for p in g['params'])
        print(f"  {g.get('name', 'unnamed')}: {n_params:,} params, lr={g['lr']}")
    print(f'S regularization: coeff_S_L1={coeff_S_L1}, coeff_S_integer={coeff_S_integer}, coeff_mass={coeff_S_mass}')
    print(f'MLP regularization: coeff_MLP_sub_diff={coeff_MLP_sub_diff}, coeff_MLP_node_L1={coeff_MLP_node_L1}')
    if n_epochs_init > 0:
        print(f'two-phase: first {n_epochs_init} epochs with L1={first_coeff_L1}, then L1={coeff_S_L1}')

    list_loss = []
    list_loss_regul = []
    loss_components = {'loss': [], 'regul_total': [], 'S_L1': [], 'S_integer': [], 'mass_conservation': [], 'MLP_sub_diff': [], 'MLP_node_L1': [], 'MLP_sub_norm': [], 'k_floor': []}

    print("start training ...")
    training_start_time = time.time()
    check_and_clear_memory(
        device=device, iteration_number=0, every_n_iterations=1,
        memory_percentage_threshold=0.6,
    )

    for epoch in range(start_epoch, n_epochs):

        Niter = int(n_frames * data_augmentation_loop // batch_size * 0.2)
        plot_frequency = max(1, Niter // 20)
        print(f'{Niter} iterations per epoch, plot every {plot_frequency}')

        total_loss = 0
        total_loss_regul = 0

        # two-phase L1: use first_coeff_L1 during n_epochs_init, then coeff_S_L1
        if n_epochs_init > 0 and epoch < n_epochs_init:
            current_L1 = first_coeff_L1
        else:
            current_L1 = coeff_S_L1

        last_r2 = None
        pbar = trange(Niter, ncols=100)
        for N in pbar:

            optimizer.zero_grad()
            if optimizer_f is not None:
                optimizer_f.zero_grad()

            loss = torch.zeros(1, device=device)
            regul_loss_val = 0.0
            run = np.random.randint(n_runs)

            for batch in range(batch_size):

                # sample timepoint: 80% variance-weighted, 20% uniform
                if variance_weighted_sampling and np.random.rand() < 0.8:
                    k = torch.multinomial(sampling_probs[run], 1).item()
                else:
                    k = np.random.randint(n_frames - 4 - time_step)
                x = x_list[run][k].clone()  # already on GPU

                # inject external input from SIREN (when learning)
                if has_visual_field and hasattr(model, 'NNR_f'):
                    # SIREN embedded in model (position + time -> field)
                    visual_input = model.forward_visual(x, k)
                    x[:n_input_metabolites, 4:5] = visual_input
                    x[n_input_metabolites:, 4:5] = 0
                elif model_f is not None:
                    if external_input_type == 'visual':
                        # Siren_Network: time -> 2D spatial field
                        x[:n_input_metabolites, 4:5] = model_f(
                            time=k / n_frames
                        ) ** 2
                        x[n_input_metabolites:, 4:5] = 1
                    else:
                        # signal-type INR models
                        inr_type = model_config.inr_type
                        nnr_f_T_period = model_config.nnr_f_T_period
                        if inr_type == 'siren_t':
                            t_norm = torch.tensor(
                                [[k / nnr_f_T_period]],
                                dtype=torch.float32, device=device,
                            )
                            x[:, 4] = model_f(t_norm).squeeze()
                        elif inr_type == 'lowrank':
                            t_idx = torch.tensor(
                                [k], dtype=torch.long, device=device
                            )
                            x[:, 4] = model_f(t_idx).squeeze()
                        elif inr_type in ('ngp', 'siren_id', 'siren_x'):
                            t_norm = torch.tensor(
                                [[k / nnr_f_T_period]],
                                dtype=torch.float32, device=device,
                            )
                            x[:, 4] = model_f(t_norm).squeeze()

                if torch.isnan(x).any():
                    continue

                if recurrent_training and time_step > 1:
                    # multi-step rollout: predict and feed back for time_step iterations
                    # target: concentration at frame k + time_step
                    y_target = x_list[run][k + time_step, :, 3]  # already on GPU

                    # current concentration
                    pred_c = x[:, 3].clone()

                    for step in range(time_step):
                        # forward pass
                        dataset = pyg_Data(x=x.clone(), pos=x[:, 1:3])
                        pred = model(dataset)

                        # update concentration: c_new = c_old + delta_t * dx/dt + noise
                        pred_c = pred_c + delta_t * pred.squeeze()
                        if noise_recurrent_level > 0:
                            pred_c = pred_c + noise_recurrent_level * torch.randn_like(pred_c)

                        # update x for next step (concentrations in column 3)
                        x = x.clone()
                        x[:, 3] = pred_c

                    # loss on final concentration vs ground truth (normalized to derivative scale)
                    loss = loss + ((pred_c - y_target) / (delta_t * time_step)).norm(2)

                else:
                    # single-step: predict dx/dt directly
                    # target: dx/dt
                    y = y_list[run][k] / ynorm  # already on GPU

                    # forward pass (bipartite graph is internal to model)
                    dataset = pyg_Data(x=x, pos=x[:, 1:3])
                    pred = model(dataset)

                    # prediction loss
                    loss = loss + (pred.squeeze() - y.squeeze()).norm(2)

            # S regularization: L1 + integer on learnable stoichiometric coefficients
            if current_L1 > 0:
                regul_S_L1 = model.sto_all.norm(1) * current_L1
                loss = loss + regul_S_L1
                regul_loss_val += regul_S_L1.item()
            if coeff_S_integer > 0:
                regul_S_int = torch.sin(np.pi * model.sto_all).pow(2).mean() * coeff_S_integer
                loss = loss + regul_S_int
                regul_loss_val += regul_S_int.item()

            # mass conservation: penalize non-zero column sums of S
            # sum_i S[i,j] = 0 means substrates consumed = products produced
            if coeff_S_mass > 0:
                col_sums = torch.zeros(
                    model.n_rxn, dtype=model.sto_all.dtype,
                    device=model.sto_all.device,
                )
                col_sums.index_add_(0, rxn_all, model.sto_all)
                regul_mass = col_sums.pow(2).mean() * coeff_S_mass
                loss = loss + regul_mass
                regul_loss_val += regul_mass.item()

            # MLP_sub monotonicity: penalize decreasing output as concentration increases
            # MLP_sub learns c^s which should be increasing in c (for s > 0)
            regul_MLP_sub_diff = torch.tensor(0.0, device=device)
            if coeff_MLP_sub_diff > 0 and hasattr(model, 'substrate_func'):
                # sample at two concentration levels
                n_samples = 100
                c_max = xnorm.item() * 2.0
                c_vals = torch.linspace(0.1, c_max, n_samples, device=device)
                delta_c = c_vals[1] - c_vals[0]

                # test with |stoich| = 1 (most common case)
                s_abs = torch.ones(n_samples, 1, device=device)
                msg_in = torch.cat([c_vals.unsqueeze(-1), s_abs], dim=-1)
                msg_in_next = torch.cat([(c_vals + delta_c).unsqueeze(-1), s_abs], dim=-1)

                msg0 = model.substrate_func(msg_in)
                msg1 = model.substrate_func(msg_in_next)

                # penalize cases where output decreases (msg0 > msg1)
                # relu(msg0 - msg1) is positive only when output decreases
                regul_MLP_sub_diff = torch.relu(msg0.norm(dim=-1) - msg1.norm(dim=-1)).norm(2) * coeff_MLP_sub_diff
                loss = loss + regul_MLP_sub_diff
                regul_loss_val += regul_MLP_sub_diff.item()

            # MLP_node L1: penalize large homeostasis output to keep it small
            # relative to reaction terms
            regul_MLP_node_L1 = torch.tensor(0.0, device=device)
            if coeff_MLP_node_L1 > 0 and hasattr(model, 'node_func'):
                concentrations = x[:, 3]
                node_in = torch.cat([concentrations.unsqueeze(-1), model.a], dim=-1)
                node_out = model.node_func(node_in).squeeze(-1)
                regul_MLP_node_L1 = node_out.abs().mean() * coeff_MLP_node_L1
                loss = loss + regul_MLP_node_L1
                regul_loss_val += regul_MLP_node_L1.item()

            # MLP_sub normalization: enforce substrate_func(c=1, |s|=1) = 1
            regul_MLP_sub_norm = torch.tensor(0.0, device=device)
            if coeff_MLP_sub_norm > 0 and hasattr(model, 'substrate_func'):
                c_ref = torch.tensor([[1.0, 1.0]], device=device)
                output_ref = model.substrate_func(c_ref).norm()
                regul_MLP_sub_norm = (output_ref - 1.0) ** 2 * coeff_MLP_sub_norm
                loss = loss + regul_MLP_sub_norm
                regul_loss_val += regul_MLP_sub_norm.item()

            # k floor: penalize log_k values below threshold
            regul_k_floor = torch.tensor(0.0, device=device)
            if coeff_k_floor > 0 and hasattr(model, 'log_k'):
                violations = torch.relu(k_floor_threshold - model.log_k)
                regul_k_floor = (violations ** 2).sum() * coeff_k_floor
                loss = loss + regul_k_floor
                regul_loss_val += regul_k_floor.item()

            # SIREN omega regularization
            coeff_omega_f_L2 = getattr(train_config, 'coeff_omega_f_L2', 0.0)
            if model_f is not None and coeff_omega_f_L2 > 0:
                if hasattr(model_f, 'get_omega_L2_loss'):
                    loss = loss + coeff_omega_f_L2 * model_f.get_omega_L2_loss()

            loss.backward()
            optimizer.step()
            if optimizer_f is not None:
                optimizer_f.step()

            total_loss += loss.item()
            total_loss_regul += regul_loss_val

            # periodic plotting
            if N % plot_frequency == 0 or N == 0:
                current_loss = loss.item()
                loss_components['loss'].append(
                    (current_loss - regul_loss_val) / n_metabolites
                )
                loss_components['regul_total'].append(
                    regul_loss_val / n_metabolites
                )
                loss_components['S_L1'].append(
                    regul_S_L1.item() / n_metabolites if current_L1 > 0 else 0.0
                )
                loss_components['S_integer'].append(
                    regul_S_int.item() / n_metabolites if coeff_S_integer > 0 else 0.0
                )
                loss_components['mass_conservation'].append(
                    regul_mass.item() / n_metabolites if coeff_S_mass > 0 else 0.0
                )
                loss_components['MLP_sub_diff'].append(
                    regul_MLP_sub_diff.item() / n_metabolites if coeff_MLP_sub_diff > 0 else 0.0
                )
                loss_components['MLP_node_L1'].append(
                    regul_MLP_node_L1.item() / n_metabolites if coeff_MLP_node_L1 > 0 else 0.0
                )
                loss_components['MLP_sub_norm'].append(
                    regul_MLP_sub_norm.item() / n_metabolites if coeff_MLP_sub_norm > 0 else 0.0
                )
                loss_components['k_floor'].append(
                    regul_k_floor.item() / n_metabolites if coeff_k_floor > 0 else 0.0
                )
                plot_dict = {k: v for k, v in loss_components.items() if any(x != 0 for x in v) or k == 'loss'}
                plot_loss(
                    plot_dict, log_dir, epoch=epoch, Niter=N, debug=False,
                    current_loss=current_loss / n_metabolites,
                    current_regul=regul_loss_val / n_metabolites,
                    total_loss=total_loss, total_loss_regul=total_loss_regul,
                )

                # plot comparison: S (if learning) or k_j (if S frozen)
                with torch.no_grad():
                    if freeze_stoichiometry and gt_model is not None:
                        # S is given, plot k_j comparison
                        last_r2, _, _, _ = _plot_rate_constants_comparison(
                            model, gt_model, log_dir, epoch, N,
                            device=device,
                        )
                    else:
                        # S is learned, plot stoichiometry comparison
                        last_r2 = _plot_stoichiometry_comparison(
                            model, gt_S, stoich_graph, n_metabolites, log_dir,
                            epoch, N,
                        )

                # plot MLP_sub learned function
                _plot_metabolism_mlp_functions(
                    model, x, xnorm, log_dir, epoch, N, device,
                    gt_model=gt_model,
                )

                # update progress bar with color-coded R2
                if last_r2 is not None:
                    if last_r2 > 0.9:
                        r2_color = '\033[92m'   # green
                    elif last_r2 > 0.7:
                        r2_color = '\033[93m'   # yellow
                    elif last_r2 > 0.3:
                        r2_color = '\033[38;5;208m'  # orange
                    else:
                        r2_color = '\033[91m'   # red
                    r2_label = 'k' if freeze_stoichiometry else 'S'
                    pbar.set_postfix_str(
                        f'{r2_color}{r2_label} R\u00b2={last_r2:.3f}\033[0m'
                    )

                # save checkpoint
                torch.save(
                    {'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()},
                    os.path.join(
                        log_dir, 'models',
                        f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt',
                    ),
                )

        # epoch summary
        epoch_total = total_loss / n_metabolites
        epoch_regul = total_loss_regul / n_metabolites
        epoch_pred = (total_loss - total_loss_regul) / n_metabolites

        print(f"epoch {epoch}. loss: {epoch_total:.6f} "
              f"(pred: {epoch_pred:.6f}, regul: {epoch_regul:.6f})")
        logger.info(f"Epoch {epoch}. Loss: {epoch_total:.6f} "
                     f"(pred: {epoch_pred:.6f}, regul: {epoch_regul:.6f})")

        list_loss.append(epoch_pred)
        list_loss_regul.append(epoch_regul)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        # save epoch checkpoint
        torch.save(
            {'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(
                log_dir, 'models',
                f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt',
            ),
        )

    # --- final analysis: compute R2 and write to log ---
    final_trimmed_r2 = 0.0
    final_n_outliers = 0
    final_slope = 0.0
    with torch.no_grad():
        if freeze_stoichiometry and gt_model is not None:
            final_r2, final_trimmed_r2, final_n_outliers, final_slope = _plot_rate_constants_comparison(
                model, gt_model, log_dir, epoch='final', N=0,
                device=device,
            )
            r2_name = "rate_constants"
        else:
            final_r2 = _plot_stoichiometry_comparison(
                model, gt_S, stoich_graph, n_metabolites, log_dir,
                epoch='final', N=0,
            )
            r2_name = "stoichiometry"
    final_loss = list_loss[-1] if list_loss else 0.0

    # --- compare learned vs GT functions ---
    func_metrics = _compare_functions(model, gt_model, x, device)

    training_elapsed_min = (time.time() - training_start_time) / 60.0
    print(f"\n=== training complete ({training_elapsed_min:.1f} min) ===")
    print(f"  final prediction loss: {final_loss:.6f}")
    # color-coded R² print with slope and outliers
    if final_r2 > 0.9:
        _r2_color = '\033[92m'   # green
    elif final_r2 > 0.3:
        _r2_color = '\033[38;5;208m'  # orange
    else:
        _r2_color = '\033[91m'   # red
    _details = []
    if freeze_stoichiometry:
        _details.append(f'trimmed: {final_trimmed_r2:.4f}')
    if final_n_outliers > 0:
        _details.append(f'outliers: {final_n_outliers}')
    if freeze_stoichiometry:
        _details.append(f'slope: {final_slope:.3f}')
    _detail_str = f' ({", ".join(_details)})' if _details else ''
    print(f"  {_r2_color}{r2_name} R2: {final_r2:.4f}{_detail_str}\033[0m")
    print(f"  MLP_sub R2: {func_metrics['MLP_sub_r2']:.4f}")
    if 'MLP_node_slope_0' in func_metrics:
        for t in range(func_metrics.get('n_types', 0)):
            learned = func_metrics[f'MLP_node_slope_{t}']
            gt = func_metrics[f'MLP_node_gt_slope_{t}']
            print(f"  MLP_node type {t}: slope={learned:.6f} (GT: {gt:.6f})")
    logger.info(f"final prediction loss: {final_loss:.6f}")
    logger.info(f"{r2_name} R2: {final_r2:.4f}")
    logger.info(f"MLP_sub R2: {func_metrics['MLP_sub_r2']:.4f}")

    if log_file is not None:
        log_file.write(f"training_time_min: {training_elapsed_min:.1f}\n")
        log_file.write(f"final_loss: {final_loss:.6f}\n")
        log_file.write(f"{r2_name}_R2: {final_r2:.4f}\n")
        if freeze_stoichiometry:
            log_file.write(f"trimmed_R2: {final_trimmed_r2:.4f}\n")
            log_file.write(f"n_outliers: {final_n_outliers}\n")
            log_file.write(f"slope: {final_slope:.4f}\n")
        log_file.write(f"MLP_sub_R2: {func_metrics['MLP_sub_r2']:.4f}\n")
        log_file.write(f"MLP_sub_corr: {func_metrics['MLP_sub_corr']:.4f}\n")
        if 'MLP_node_slope_0' in func_metrics:
            for t in range(func_metrics.get('n_types', 0)):
                log_file.write(f"MLP_node_slope_{t}: {func_metrics[f'MLP_node_slope_{t}']:.6f}\n")
                log_file.write(f"MLP_node_gt_slope_{t}: {func_metrics[f'MLP_node_gt_slope_{t}']:.6f}\n")


def data_test_metabolism(config, best_model=20, n_rollout_frames=600, device=None, log_file=None):
    """test a trained metabolism model: rollout + stoichiometry comparison.

    loads the trained Metabolism_Propagation model, runs a rollout over
    n_rollout_frames, and computes metrics comparing predicted vs true
    concentration trajectories and learned vs true stoichiometric matrix.

    writes metrics to log_file (analysis.log) if provided.
    """
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    dataset_name = config.dataset
    n_runs = train_config.n_runs
    n_frames = simulation_config.n_frames
    n_metabolites = simulation_config.n_metabolites
    delta_t = simulation_config.delta_t

    # --- determine log_dir (must match training: ./log/{config_file}) ---
    log_dir = os.path.join('./log', config.config_file)
    print(f'log_dir: {log_dir}')

    # --- load data ---
    x_list = []
    y_list = []
    for run in range(n_runs):
        x = load_simulation_data(f'graphs_data/{dataset_name}/x_list_{run}')
        x_list.append(x)
        y = load_simulation_data(f'graphs_data/{dataset_name}/y_list_{run}')
        y_list.append(y)

    # --- load stoichiometric graph and ground truth ---
    stoich_graph = torch.load(
        f'graphs_data/{dataset_name}/stoich_graph.pt', map_location=device
    )
    gt_S = torch.load(
        f'graphs_data/{dataset_name}/stoichiometry.pt', map_location=device
    )

    # --- normalization (ynorm=1 always) ---
    ynorm = torch.tensor(1.0, device=device)

    # --- find best model if 'best' specified ---
    if best_model == 'best':
        files = glob.glob(f"{log_dir}/models/*")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]
        best_model = filename
        print(f'best model: {best_model}')

    # --- load trained model ---
    from MetabolismGraph.models.Metabolism_Propagation import Metabolism_Propagation
    model = Metabolism_Propagation(config=config, device=device)
    model.load_stoich_graph(stoich_graph)
    model = model.to(device)

    net = os.path.join(
        log_dir, 'models',
        f'best_model_with_{n_runs - 1}_graphs_{best_model}.pt',
    )
    print(f'loading model from {net}')
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    # --- load SIREN model_f if it exists ---
    model_f = None
    model_f_path = os.path.join(
        log_dir, 'models',
        f'best_model_f_with_{n_runs - 1}_graphs_{best_model}.pt',
    )
    if os.path.exists(model_f_path):
        model_f = choose_inr_model(
            config=config, n_metabolites=n_metabolites, n_frames=n_frames,
            x_list=x_list, device=device,
        )
        if model_f is not None:
            state_dict_f = torch.load(model_f_path, map_location=device)
            model_f.load_state_dict(state_dict_f['model_state_dict'])
            model_f.eval()
            print(f'loaded model_f from {model_f_path}')

    # --- check if S is frozen (S given mode) ---
    freeze_stoichiometry = getattr(train_config, 'freeze_stoichiometry', False)

    # --- stoichiometry R2 (learned vs true S) - skip if frozen ---
    stoich_r2 = 1.0 if freeze_stoichiometry else 0.0
    lin_fit = None
    n_edges = 0

    if not freeze_stoichiometry:
        with torch.no_grad():
            learned_S = torch.zeros_like(gt_S, device='cpu')
            met_all_cpu = stoich_graph['all'][0].cpu()
            rxn_all_cpu = stoich_graph['all'][1].cpu()
            learned_S[met_all_cpu, rxn_all_cpu] = model.sto_all.detach().cpu()

        # compare only edge coefficients (not full matrix with trivial zeros)
        gt_edges = to_numpy(gt_S.cpu()[met_all_cpu, rxn_all_cpu])
        learned_edges = to_numpy(model.sto_all.detach().cpu())
        n_edges = len(gt_edges)

        try:
            lin_fit, _ = curve_fit(linear_model, gt_edges, learned_edges)
            residuals = learned_edges - linear_model(gt_edges, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((learned_edges - np.mean(learned_edges)) ** 2)
            stoich_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        except Exception:
            pass

        print(f'stoichiometry R2: {stoich_r2:.4f} (n={n_edges} edges)')

    # --- scatter plot: true vs learned S (edges only) - skip if frozen ---
    if not freeze_stoichiometry:
        out_dir = os.path.join(log_dir, 'tmp_training', 'stoichiometry')
        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(gt_edges, learned_edges, s=2, c='k', alpha=0.5, edgecolors=None)
        ax.set_xlabel(r'true $S_{ij}$', fontsize=14)
        ax.set_ylabel(r'learned $S_{ij}$', fontsize=14)
        ax.text(0.05, 0.96, f'$R^2$: {stoich_r2:.3f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top')
        if lin_fit is not None:
            ax.text(0.05, 0.92, f'slope: {lin_fit[0]:.3f}', transform=ax.transAxes,
                    fontsize=12, verticalalignment='top')
        ax.text(0.05, 0.88, f'n={n_edges} edges', transform=ax.transAxes,
                fontsize=12, verticalalignment='top')
        lims = [min(gt_edges.min(), learned_edges.min()) - 0.2,
                max(gt_edges.max(), learned_edges.max()) + 0.2]
        ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'scatter_test.png'), dpi=150)
        plt.close()

    # --- rollout: predict concentration trajectories ---
    run = 0
    n_test_frames = min(n_rollout_frames, n_frames - 2)
    start_frame = 0

    # ground truth and predicted concentration trajectories
    activity_true = np.zeros((n_metabolites, n_test_frames))
    activity_pred = np.zeros((n_metabolites, n_test_frames))

    with torch.no_grad():
        # initialize from ground truth at start_frame
        x = torch.tensor(
            x_list[run][start_frame], dtype=torch.float32, device=device
        )

        for t in trange(n_test_frames, desc='rollout', ncols=100):
            frame_idx = start_frame + t

            # record ground truth concentrations
            x_gt = torch.tensor(
                x_list[run][frame_idx], dtype=torch.float32, device=device
            )
            activity_true[:, t] = to_numpy(x_gt[:, 3])
            activity_pred[:, t] = to_numpy(x[:, 3])

            # inject external input from SIREN if available
            if model_f is not None:
                external_input_type = simulation_config.external_input_type
                n_input_met = simulation_config.n_input_metabolites
                inr_type = model_config.inr_type
                nnr_f_T_period = model_config.nnr_f_T_period

                if external_input_type == 'visual':
                    if hasattr(model, 'NNR_f'):
                        x[:n_input_met, 4:5] = model.forward_visual(x, frame_idx)
                    else:
                        x[:n_input_met, 4:5] = model_f(
                            time=frame_idx / n_frames
                        ) ** 2
                        x[n_input_met:, 4:5] = 1
                elif inr_type == 'siren_t':
                    t_norm = torch.tensor(
                        [[frame_idx / nnr_f_T_period]],
                        dtype=torch.float32, device=device,
                    )
                    x[:, 4] = model_f(t_norm).squeeze()

            # forward pass: predict dx/dt
            dataset = pyg_Data(x=x, pos=x[:, 1:3])
            dxdt = model(dataset)

            # euler integration: c(t+1) = c(t) + dc/dt * delta_t
            x[:, 3:4] = x[:, 3:4] + dxdt * delta_t

    # --- compute test metrics ---
    # per-metabolite R2
    r2_list = []
    for i in range(n_metabolites):
        gt_i = activity_true[i, :]
        pred_i = activity_pred[i, :]
        ss_res = np.sum((gt_i - pred_i) ** 2)
        ss_tot = np.sum((gt_i - np.mean(gt_i)) ** 2)
        if ss_tot > 0:
            r2_list.append(1 - ss_res / ss_tot)
    test_r2 = np.mean(r2_list) if r2_list else 0.0

    # per-metabolite Pearson correlation
    pearson_list = []
    for i in range(n_metabolites):
        gt_i = activity_true[i, :]
        pred_i = activity_pred[i, :]
        if np.std(gt_i) > 1e-12 and np.std(pred_i) > 1e-12:
            r = np.corrcoef(gt_i, pred_i)[0, 1]
            if not np.isnan(r):
                pearson_list.append(r)
    test_pearson = np.mean(pearson_list) if pearson_list else 0.0

    print(f'\n=== test results ===')
    if not freeze_stoichiometry:
        print(f'  stoichiometry R2: {stoich_r2:.4f}')
    print(f'  test R2 (rollout): {test_r2:.4f}')
    print(f'  test Pearson: {test_pearson:.4f}')

    # --- create results folder ---
    results_dir = os.path.join(log_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # --- save kinograph (concentration heatmap) as npy ---
    np.save(os.path.join(results_dir, 'kinograph_gt.npy'), activity_true)
    np.save(os.path.join(results_dir, 'kinograph_pred.npy'), activity_pred)

    # --- kinograph montage 2x2: GT, pred, residual, scatter ---
    # use min/max of true concentrations for consistent color scale
    vmin_true = activity_true.min()
    vmax_true = activity_true.max()
    residual = activity_true - activity_pred
    vmax_res = np.abs(residual).max()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_gt, ax_pred, ax_res, ax_scat = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Top-left: GT
    im_gt = ax_gt.imshow(activity_true, aspect='auto', cmap='viridis', vmin=vmin_true, vmax=vmax_true, origin='lower', interpolation='nearest')
    ax_gt.set_ylabel('metabolites', fontsize=14)
    ax_gt.set_xticks([0, n_test_frames - 1]); ax_gt.set_xticklabels([0, n_test_frames], fontsize=12)
    ax_gt.set_yticks([0, n_metabolites - 1]); ax_gt.set_yticklabels([1, n_metabolites], fontsize=12)
    ax_gt.text(0.02, 0.97, 'ground truth', transform=ax_gt.transAxes, fontsize=9,
               color='white', va='top', ha='left')
    fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04).ax.tick_params(labelsize=12)

    # Top-right: GNN prediction (same color scale as GT for direct comparison)
    im_pred = ax_pred.imshow(activity_pred, aspect='auto', cmap='viridis', vmin=vmin_true, vmax=vmax_true, origin='lower', interpolation='nearest')
    ax_pred.set_xticks([0, n_test_frames - 1]); ax_pred.set_xticklabels([0, n_test_frames], fontsize=12)
    ax_pred.set_yticks([0, n_metabolites - 1]); ax_pred.set_yticklabels([1, n_metabolites], fontsize=12)
    ax_pred.text(0.02, 0.97, 'GNN prediction', transform=ax_pred.transAxes, fontsize=9,
                 color='white', va='top', ha='left')
    fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04).ax.tick_params(labelsize=12)

    # Bottom-left: Residual (same vmin/vmax as top panels)
    im_res = ax_res.imshow(residual, aspect='auto', cmap='RdBu_r', vmin=vmin_true, vmax=vmax_true, origin='lower', interpolation='nearest')
    ax_res.set_ylabel('metabolites', fontsize=14)
    ax_res.set_xlabel('time', fontsize=14)
    ax_res.set_xticks([0, n_test_frames - 1]); ax_res.set_xticklabels([0, n_test_frames], fontsize=12)
    ax_res.set_yticks([0, n_metabolites - 1]); ax_res.set_yticklabels([1, n_metabolites], fontsize=12)
    fig.colorbar(im_res, ax=ax_res, fraction=0.046, pad=0.04).ax.tick_params(labelsize=12)

    # Bottom-right: Scatter true vs predicted (use true concentration range)
    gt_flat = activity_true.flatten()
    pred_flat = activity_pred.flatten()
    ax_scat.scatter(gt_flat, pred_flat, s=1, alpha=0.1, c='k', rasterized=True, edgecolors=None)
    lims = [vmin_true, vmax_true]
    ax_scat.plot(lims, lims, 'r--', linewidth=2)
    ax_scat.set_xlim(lims); ax_scat.set_ylim(lims)
    ax_scat.set_xlabel('ground truth concentration', fontsize=14)
    ax_scat.set_ylabel('predicted concentration', fontsize=14)
    ax_scat.tick_params(labelsize=12)
    ax_scat.text(0.05, 0.95, f'R²={test_r2:.3f}\nPearson={test_pearson:.3f}',
                 transform=ax_scat.transAxes, fontsize=12, va='top')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'kinograph_montage.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- concentration traces plot (like concentrations.png but with GT overlay) ---
    n_traces = min(20, n_metabolites)
    trace_ids = np.linspace(0, n_metabolites - 1, n_traces, dtype=int)

    # compute per-trace R²
    r2_per_trace = []
    for idx in trace_ids:
        gt_trace = activity_true[idx]
        pred_trace = activity_pred[idx]
        ss_res = np.sum((gt_trace - pred_trace) ** 2)
        ss_tot = np.sum((gt_trace - np.mean(gt_trace)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r2_per_trace.append(r2)

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # compute offset based on data range
    offset = np.abs(activity_true[trace_ids]).max() * 1.5
    if offset == 0:
        offset = 1.0

    for j, n_idx in enumerate(trace_ids):
        y0 = j * offset
        baseline = np.mean(activity_true[n_idx])
        # Ground truth in green (thicker line)
        ax.plot(activity_true[n_idx] - baseline + y0, color='green', lw=3.0, alpha=0.9)
        # Prediction in red (thinner line)
        ax.plot(activity_pred[n_idx] - baseline + y0, color='red', lw=1.0, alpha=0.9)

        # Metabolite index on the left
        ax.text(-n_test_frames * 0.02, y0, str(n_idx), fontsize=10, va='center', ha='right')

        # R² on the right with color coding
        r2_val = r2_per_trace[j]
        r2_color = 'red' if r2_val < 0.5 else ('orange' if r2_val < 0.8 else 'green')
        ax.text(n_test_frames * 1.02, y0, f'R²:{r2_val:.2f}', fontsize=9, va='center', ha='left', color=r2_color)

    ax.set_xlim([-n_test_frames * 0.05, n_test_frames * 1.1])
    ax.set_ylim([-offset, n_traces * offset])
    ax.set_xlabel('frame', fontsize=14)
    ax.set_ylabel('metabolite', fontsize=14)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(0, n_test_frames)
    ax.set_yticks([])

    # legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='green', lw=3, label='ground truth'),
                       Line2D([0], [0], color='red', lw=1, label='GNN prediction')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'concentrations.png'), dpi=150)
    plt.close()

    # --- compute alpha (MLP_sub scale factor) ---
    freeze_stoichiometry = getattr(train_config, 'freeze_stoichiometry', False)
    alpha_val = None
    if hasattr(model, 'substrate_func'):
        with torch.no_grad():
            c_ref = torch.tensor([[1.0, 1.0]], device=device)
            alpha_val = model.substrate_func(c_ref).norm().item()

    # --- write to analysis log ---
    if log_file is not None:
        if not freeze_stoichiometry:
            log_file.write(f"stoichiometry_R2: {stoich_r2:.4f}\n")
        log_file.write(f"test_R2: {test_r2:.4f}\n")
        log_file.write(f"test_pearson: {test_pearson:.4f}\n")
        if alpha_val is not None:
            log_file.write(f"alpha: {alpha_val:.4f}\n")


def _plot_metabolism_mlp_functions(model, x, xnorm, log_dir, epoch, N, device,
                                  gt_model=None):
    """plot learned MLP_sub, MLP_node, and embeddings during metabolism training.

    MLP_sub: sweep concentration at several fixed |stoich| values -> ||output||.
    MLP_node: per-metabolite-type homeostasis function (c_i, a_i) -> homeostasis term.
    embeddings: scatter plot of learned metabolite embeddings a_i colored by type.
    if gt_model is provided, overlay ground-truth curves as dashed lines.

    saves to tmp_training/function/ and embedding/.
    """
    func_dir = f"./{log_dir}/tmp_training/function"
    sub_dir = f"{func_dir}/substrate_func"
    rate_dir = f"{func_dir}/rate_func"
    emb_dir = f"./{log_dir}/embedding"
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(rate_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    n_pts = 500

    # --- MLP_sub: concentration sweep at fixed |stoich| values ---
    conc_max = to_numpy(xnorm).item() * 3.0
    conc_range = torch.linspace(0, conc_max, n_pts, device=device)

    stoich_values = [1, 2]
    colors = ['tab:blue', 'tab:orange']

    fig, ax = plt.subplots(figsize=(8, 8))
    c_np = to_numpy(conc_range)
    scale_factors = {}
    with torch.no_grad():
        for s_val, color in zip(stoich_values, colors):
            s_abs = torch.full((n_pts, 1), float(s_val), device=device)
            msg_in = torch.cat([conc_range.unsqueeze(-1), s_abs], dim=-1)
            msg_out = model.substrate_func(msg_in)
            msg_norm = msg_out.norm(dim=-1)
            ax.plot(c_np, to_numpy(msg_norm),
                    linewidth=2, color=color, label=f'learned |s|={s_val}')

            # true power law c^s (normalized to match learned scale)
            true_power = np.power(c_np + 1e-8, s_val)
            scale = to_numpy(msg_norm).max() / (true_power.max() + 1e-8)
            scale_factors[s_val] = scale
            ax.plot(c_np, true_power * scale,
                    linewidth=2, color=color, linestyle='--', alpha=0.5,
                    label=f'$c^{{{s_val}}}$ (scaled)')

    # annotate scale factor α (used for log_k correction)
    alpha_text = ', '.join(f'|s|={s}: {scale_factors[s]:.3f}'
                           for s in stoich_values)
    ax.text(0.05, 0.96, f'$\\alpha$: {alpha_text}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top')

    ax.set_xlabel('concentration', fontsize=14)
    ax.set_ylabel(r'$\|\mathrm{MLP_{sub}}(c, |s|)\|$', fontsize=14)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(f"{sub_dir}/MLP_sub_{epoch}_{N}.png", dpi=87)
    plt.close()

    # --- MLP_node: per-metabolite homeostasis function ---
    # MLP_node(c_i, a_i) learns -λ_i(c_i - c_baseline)
    # Plot for ALL metabolites using their individual embeddings a_i
    with torch.no_grad():
        metabolite_types = x[:, 6].long()
        n_types = metabolite_types.max().item() + 1
        n_met = model.a.shape[0]
        cmap = plt.cm.get_cmap('tab10')

        fig, ax = plt.subplots(figsize=(8, 8))

        # plot MLP_node for each individual metabolite, collect slopes
        slopes_by_type = {}
        for i in range(n_met):
            a_i = model.a[i]  # embedding for metabolite i
            t = metabolite_types[i].item()

            # sweep concentration with this metabolite's embedding
            a_repeated = a_i.unsqueeze(0).expand(n_pts, -1)
            node_in = torch.cat([conc_range.unsqueeze(-1), a_repeated], dim=-1)
            homeostasis = model.node_func(node_in).squeeze(-1)

            h_np = to_numpy(homeostasis)
            ax.plot(c_np, h_np, linewidth=1, color=cmap(t), alpha=0.3)

            # linear fit y = a*x + b
            coeffs = np.polyfit(c_np, h_np, 1)
            slopes_by_type.setdefault(t, []).append(coeffs[0])

        # GT homeostasis if available: -λ_t * (c - c_baseline_t)
        if gt_model is not None and hasattr(gt_model, 'p'):
            p = to_numpy(gt_model.p.detach().cpu())
            for t in range(n_types):
                if t < p.shape[0]:
                    lambda_t = p[t, 0]
                    c_baseline_t = p[t, 1]
                    gt_homeostasis = -lambda_t * (c_np - c_baseline_t)
                    ax.plot(c_np, gt_homeostasis, linewidth=2, color=cmap(t),
                            linestyle='--', label=f'GT type {t}')
                    ax.axvline(x=c_baseline_t, color=cmap(t), linestyle=':',
                               alpha=0.3)

        # annotate mean learned slope vs GT slope per type
        slope_lines = []
        for t in sorted(slopes_by_type.keys()):
            mean_slope = np.mean(slopes_by_type[t])
            gt_slope_str = ''
            if gt_model is not None and hasattr(gt_model, 'p'):
                p = to_numpy(gt_model.p.detach().cpu())
                if t < p.shape[0]:
                    gt_slope_str = f', GT: {-p[t, 0]:.4f}'
            slope_lines.append(f'type {t}: slope={mean_slope:.4f}{gt_slope_str}')
        if slope_lines:
            ax.text(0.05, 0.96, '\n'.join(slope_lines),
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              alpha=0.8))

        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('concentration $c$', fontsize=14)
        ax.set_ylabel(r'$\mathrm{MLP_{node}}(c, a)$', fontsize=14)
        ax.legend(fontsize=10, loc='best')
        ax.tick_params(labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)

        # set y-axis limits from ground truth range (with margin)
        if gt_model is not None and hasattr(gt_model, 'p'):
            p = to_numpy(gt_model.p.detach().cpu())
            gt_vals = []
            for t in range(min(n_types, p.shape[0])):
                lambda_t = p[t, 0]
                c_baseline_t = p[t, 1]
                gt_vals.append(-lambda_t * (c_np[0] - c_baseline_t))
                gt_vals.append(-lambda_t * (c_np[-1] - c_baseline_t))
            gt_ymin, gt_ymax = min(gt_vals), max(gt_vals)
            margin = (gt_ymax - gt_ymin) * 0.3 + 1e-6
            ax.set_ylim(gt_ymin - margin, gt_ymax + margin)

    plt.tight_layout()
    plt.savefig(f"{rate_dir}/MLP_node_{epoch}_{N}.png", dpi=87)
    plt.close()

    # --- Embeddings: scatter plot of learned metabolite embeddings ---
    with torch.no_grad():
        metabolite_types = x[:, 6].long()
        n_types = metabolite_types.max().item() + 1
        cmap = plt.cm.get_cmap('tab10')

        fig, ax = plt.subplots(figsize=(8, 8))

        a_np = to_numpy(model.a.detach().cpu())
        for t in range(n_types):
            pos = torch.argwhere(metabolite_types == t).squeeze(-1)
            if len(pos) == 0:
                continue
            pos_np = to_numpy(pos)
            ax.scatter(a_np[pos_np, 0], a_np[pos_np, 1], s=50, color=cmap(t),
                       alpha=0.7, edgecolors=None)

        ax.set_xlabel('embedding 0', fontsize=14)
        ax.set_ylabel('embedding 1', fontsize=14)
        ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f"{emb_dir}/Fig_{epoch}_{N}.png", dpi=87)
    plt.close()


def _plot_stoichiometry_comparison(model, gt_S, stoich_graph, n_metabolites,
                                   log_dir, epoch, N):
    """plot learned vs ground-truth stoichiometric matrix + scatter with R2.

    returns
    -------
    r_squared : float
        R2 between true and learned stoichiometric coefficients.
    """
    out_dir = f"./{log_dir}/tmp_training/matrix"
    os.makedirs(out_dir, exist_ok=True)

    learned_S = torch.zeros_like(gt_S, device='cpu')
    met_all = stoich_graph['all'][0].cpu()
    rxn_all = stoich_graph['all'][1].cpu()
    learned_S[met_all, rxn_all] = model.sto_all.detach().cpu()
    gt_S_cpu = gt_S.cpu()

    # --- 3 panels: GT heatmap | Learned heatmap | Scatter ---
    gt_edges = to_numpy(gt_S_cpu[met_all, rxn_all])
    learned_edges = to_numpy(model.sto_all.detach().cpu())
    n_edges = len(gt_edges)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(
        to_numpy(gt_S_cpu), aspect='auto', cmap='bwr', vmin=-3, vmax=3,
    )
    axes[0].set_xlabel('reactions')
    axes[0].set_ylabel('metabolites')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(
        to_numpy(learned_S), aspect='auto', cmap='bwr', vmin=-3, vmax=3,
    )
    axes[1].set_xlabel('reactions')
    axes[1].set_ylabel('metabolites')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # panel 3: scatter plot true vs learned
    axes[2].scatter(gt_edges, learned_edges, s=2, c='k', alpha=0.5, edgecolors=None)
    axes[2].set_xlabel(r'true $S_{ij}$', fontsize=14)
    axes[2].set_ylabel(r'learned $S_{ij}$', fontsize=14)

    r_squared = 0.0
    try:
        lin_fit, _ = curve_fit(linear_model, gt_edges, learned_edges)
        residuals = learned_edges - linear_model(gt_edges, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((learned_edges - np.mean(learned_edges)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        axes[2].text(0.05, 0.96, f'$R^2$: {r_squared:.3f}', transform=axes[2].transAxes,
                     fontsize=12, verticalalignment='top')
        axes[2].text(0.05, 0.90, f'slope: {lin_fit[0]:.3f}', transform=axes[2].transAxes,
                     fontsize=12, verticalalignment='top')
        axes[2].text(0.05, 0.84, f'n={n_edges} edges', transform=axes[2].transAxes,
                     fontsize=12, verticalalignment='top')
    except Exception:
        pass

    lims = [min(gt_edges.min(), learned_edges.min()) - 0.2,
            max(gt_edges.max(), learned_edges.max()) + 0.2]
    axes[2].plot(lims, lims, 'r--', alpha=0.5, linewidth=1)
    axes[2].set_xlim(lims)
    axes[2].set_ylim(lims)
    axes[2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(
        f"{out_dir}/comparison_{epoch}_{N}.png", dpi=150,
        bbox_inches='tight',
    )
    plt.close()

    return r_squared


def _compute_scalar_correction(model, device):
    """Compute MLP_sub scale factor α (scalar correction).

    Evaluate substrate_func at c=1, |s|=1 where the true value is
    c^s = 1^1 = 1.  The learned output at this point gives α directly.

    If substrate_func learns α*c^s instead of c^s, with multiplicative
    aggregation the rate constants absorb the inverse:
        k_learned = k_true / α^{n_substrates}
    so the correction is:
        log_k_corrected = log_k_learned + n_j * log10(α)

    Returns
    -------
    log_alpha : float
        log10 of the scale factor α.
    n_sub_per_rxn : ndarray (n_rxn,)
        Number of substrate edges per reaction.
    """
    with torch.no_grad():
        # at c=1, true c^s = 1 for any s, so learned output = α
        c_ref = torch.tensor([[1.0, 1.0]], device=device)  # [c=1, |s|=1]
        alpha = model.substrate_func(c_ref).norm().item()

    log_alpha = np.log10(abs(alpha) + 1e-8)
    n_sub = to_numpy(model.n_sub_per_rxn)

    return log_alpha, n_sub


def _plot_rate_constants_comparison(model, gt_model, log_dir, epoch, N,
                                    device=None, outlier_threshold=0.3):
    """plot learned vs ground-truth rate constants k_j.

    When device is provided, computes a scalar correction from the MLP_sub
    scale factor (analogous to second_correction in NeuralGraph).

    Outlier reactions (|corrected_log_k - gt_log_k| > outlier_threshold)
    are highlighted in red and excluded from the trimmed R² computation.

    returns
    -------
    raw_r2 : float
        R2 on all reactions in log space.
    trimmed_r2 : float
        R2 excluding outliers in log space.
    n_outliers : int
        Number of outlier reactions excluded.
    slope : float
        Slope of the linear fit.
    """
    out_dir = f"./{log_dir}/tmp_training/rate_constants"
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        gt_log_k = to_numpy(gt_model.log_k.detach().cpu())
        learned_log_k = to_numpy(model.log_k.detach().cpu())
        n_rxn = len(gt_log_k)

    # --- compute scalar correction from MLP_sub scale factor ---
    corrected_log_k = learned_log_k
    has_correction = False
    if (device is not None
            and hasattr(model, 'substrate_func')
            and hasattr(model, 'n_sub_per_rxn')):
        log_alpha, n_sub = _compute_scalar_correction(model, device)
        corrected_log_k = learned_log_k + n_sub * log_alpha
        has_correction = True
        # save scalar correction (like NeuralGraph second_correction)
        np.save(f"./{log_dir}/scalar_correction.npy",
                np.array([log_alpha, 10.0 ** log_alpha]))

    # use corrected values for R² computation
    plot_log_k = corrected_log_k if has_correction else learned_log_k

    # --- outlier detection ---
    identity_errors = np.abs(plot_log_k - gt_log_k)
    outlier_mask = identity_errors > outlier_threshold
    n_outliers = int(np.sum(outlier_mask))
    # if >50% are outliers, model failed — still show red but don't trim R²
    trimmed = n_outliers <= n_rxn * 0.5
    keep_mask = ~outlier_mask if trimmed else np.ones(n_rxn, dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 8))

    if has_correction:
        # faded uncorrected points
        ax.scatter(gt_log_k, learned_log_k, s=15, c='gray', alpha=0.3,
                   edgecolors='none', label='uncorrected')
        # corrected inlier points
        ax.scatter(gt_log_k[~outlier_mask], corrected_log_k[~outlier_mask],
                   s=20, c='k', alpha=0.6, edgecolors='none', label='corrected')
        # corrected outlier points in red
        if n_outliers > 0:
            ax.scatter(gt_log_k[outlier_mask], corrected_log_k[outlier_mask],
                       s=20, c='red', alpha=0.6, edgecolors='none',
                       label=f'outlier ({n_outliers})')
    else:
        ax.scatter(gt_log_k[~outlier_mask], learned_log_k[~outlier_mask],
                   s=20, c='k', alpha=0.6, edgecolors='none')
        if n_outliers > 0:
            ax.scatter(gt_log_k[outlier_mask], learned_log_k[outlier_mask],
                       s=20, c='red', alpha=0.6, edgecolors='none',
                       label=f'outlier ({n_outliers})')

    ax.set_xlabel(r'true $\log_{10}(k_j)$', fontsize=14)
    ax.set_ylabel(r'learned $\log_{10}(k_j)$', fontsize=14)

    raw_r2 = 0.0
    trimmed_r2 = 0.0
    slope = 0.0
    dy = 0.04  # text line spacing
    try:
        # raw R² on ALL reactions
        lin_fit_all, _ = curve_fit(linear_model, gt_log_k, plot_log_k)
        res_all = plot_log_k - linear_model(gt_log_k, *lin_fit_all)
        ss_res_all = np.sum(res_all ** 2)
        ss_tot_all = np.sum((plot_log_k - np.mean(plot_log_k)) ** 2)
        raw_r2 = 1 - (ss_res_all / ss_tot_all) if ss_tot_all > 0 else 0.0

        # trimmed R² (excluding outliers)
        gt_kept = gt_log_k[keep_mask]
        pred_kept = plot_log_k[keep_mask]
        lin_fit, _ = curve_fit(linear_model, gt_kept, pred_kept)
        residuals = pred_kept - linear_model(gt_kept, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((pred_kept - np.mean(pred_kept)) ** 2)
        trimmed_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        slope = lin_fit[0]

        y_text = 0.98
        ax.text(0.05, y_text, f'$R^2$: {raw_r2:.3f} (trimmed: {trimmed_r2:.3f})',
                transform=ax.transAxes, fontsize=12, verticalalignment='top')
        y_text -= dy
        ax.text(0.05, y_text, f'slope: {slope:.3f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top')
    except Exception:
        y_text = 0.90

    y_text -= dy
    ax.text(0.05, y_text, f'n={n_rxn} reactions', transform=ax.transAxes,
            fontsize=12, verticalalignment='top')
    if n_outliers > 0:
        y_text -= dy
        ax.text(0.05, y_text, f'outliers: {n_outliers}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', color='red')
    if has_correction:
        y_text -= dy
        ax.text(0.05, y_text,
                f'$\\log_{{10}}\\alpha$: {log_alpha:.3f}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top')
        ax.legend(fontsize=10, loc='lower right')
    elif n_outliers > 0:
        ax.legend(fontsize=10, loc='lower right')

    # fixed axis limits
    lims = [-2.5, -0.75]
    ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/comparison_{epoch}_{N}.png", dpi=150, bbox_inches='tight')
    plt.close()

    return raw_r2, trimmed_r2, n_outliers, slope


def _compare_functions(model, gt_model, x, device):
    """Compare learned vs ground truth MLP_sub and MLP_node functions.

    Evaluates MLP_sub on a grid of inputs and computes correlation metrics.
    For MLP_node, fits y = a*x + b per metabolite and compares slopes to GT.

    Returns
    -------
    dict with keys:
        MLP_sub_r2: R² between GT and learned MLP_sub outputs
        MLP_sub_corr: Pearson correlation for MLP_sub
        MLP_node_slope_t: mean learned slope for type t (one key per type)
        MLP_node_gt_slope_t: GT slope (-λ_t) for type t (one key per type)
        n_types: number of metabolite types
    """
    model.eval()
    gt_model.eval()

    with torch.no_grad():
        # --- MLP_sub comparison ---
        # Create grid of inputs: [concentration, |stoich|]
        # concentration range: 0 to 10, stoich range: 0 to 3
        n_pts = 50
        conc = torch.linspace(0, 10, n_pts, device=device)
        stoich = torch.linspace(0, 3, n_pts, device=device)
        conc_grid, stoich_grid = torch.meshgrid(conc, stoich, indexing='ij')
        inputs = torch.stack([conc_grid.flatten(), stoich_grid.flatten()], dim=-1)

        # Evaluate both models
        learned_out = model.substrate_func(inputs)
        gt_out = gt_model.substrate_func(inputs)

        # Flatten to 1D for correlation
        learned_flat = learned_out.flatten()
        gt_flat = gt_out.flatten()

        # Compute metrics
        ss_res = ((learned_flat - gt_flat) ** 2).sum()
        ss_tot = ((gt_flat - gt_flat.mean()) ** 2).sum()
        sub_r2 = 1 - ss_res / (ss_tot + 1e-8)

        # Pearson correlation
        learned_centered = learned_flat - learned_flat.mean()
        gt_centered = gt_flat - gt_flat.mean()
        sub_corr = (learned_centered * gt_centered).sum() / (
            learned_centered.norm() * gt_centered.norm() + 1e-8
        )

    result = {
        'MLP_sub_r2': float(sub_r2.cpu()),
        'MLP_sub_corr': float(sub_corr.cpu()),
    }

    # --- MLP_node comparison ---
    # For each metabolite, sweep concentration, fit y = a*x + b
    # Compare learned slope to GT slope (-λ_type)
    if hasattr(model, 'node_func') and gt_model is not None and hasattr(gt_model, 'p'):
        p_node = gt_model.p.detach().cpu()
        n_met = model.a.shape[0]
        metabolite_types = x[:, 6].long()
        n_types = metabolite_types.max().item() + 1

        conc_sweep = torch.linspace(0, 40, 50, device=device)
        c_np_sweep = conc_sweep.cpu().numpy()

        slopes_by_type = {}  # {type_id: [slopes]}
        for i in range(n_met):
            a_i = model.a[i]
            t = metabolite_types[i].item()
            a_rep = a_i.unsqueeze(0).expand(50, -1)
            node_in = torch.cat([conc_sweep.unsqueeze(-1), a_rep], dim=-1)
            out = model.node_func(node_in).squeeze(-1)
            coeffs = np.polyfit(c_np_sweep, out.cpu().numpy(), 1)  # [slope, intercept]
            slopes_by_type.setdefault(t, []).append(coeffs[0])

        result['n_types'] = n_types
        for t in range(n_types):
            if t < p_node.shape[0]:
                gt_slope = -p_node[t, 0].item()
                learned_slopes = slopes_by_type.get(t, [0.0])
                result[f'MLP_node_slope_{t}'] = float(np.mean(learned_slopes))
                result[f'MLP_node_gt_slope_{t}'] = float(gt_slope)

    return result
