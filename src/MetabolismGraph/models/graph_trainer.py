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


def data_test(config, best_model=20, n_rollout_frames=2000, device=None, log_file=None,
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
    homeostasis_training = getattr(train_config, 'homeostasis_training', False)
    skip_phase1 = getattr(train_config, 'skip_phase1', False)
    homeostasis_time_step = getattr(train_config, 'homeostasis_time_step', 32)
    lr_node_homeo = getattr(train_config, 'learning_rate_node_homeostasis', 0.0)
    lr_embedding_homeo = getattr(train_config, 'learning_rate_embedding_homeostasis', 0.0)

    external_input_type = simulation_config.external_input_type
    learn_external_input = train_config.learn_external_input
    field_type = model_config.field_type

    has_visual_field = 'visual' in field_type

    log_dir, logger = create_log_dir(config, erase, keep_model=best_model)

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
        try:
            start_epoch = int(best_model.split('_')[0])
        except ValueError:
            start_epoch = n_epochs  # non-numeric (e.g. 'phase2'): skip Phase 1
        print(f'state_dict loaded, start_epoch={start_epoch}')

    # --- skip Phase 1 if configured ---
    if skip_phase1:
        start_epoch = n_epochs
        print('skip_phase1: skipping Phase 1 training')

    # --- optimizer (custom param groups for metabolism) ---
    lr = train_config.learning_rate_start
    lr_S = train_config.learning_rate_S_start
    # separate learning rates for model components (fall back to lr if not specified)
    lr_k = getattr(train_config, 'learning_rate_k', lr)
    lr_node = getattr(train_config, 'learning_rate_node', lr)
    lr_sub = getattr(train_config, 'learning_rate_sub', lr)
    if lr_node_homeo == 0.0:
        lr_node_homeo = lr_node
    if lr_embedding_homeo == 0.0:
        lr_embedding_homeo = lr_node_homeo

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

    # ===== Phase 2: Homeostasis training (recurrent) =====
    # Strategy 2: Signal Amplification + Slope-Weighted Loss (Bengio et al. 2009, Goyal et al. 2017)
    #
    # MOTIVATION: Strategy 1 (Residual Direct Supervision) failed because:
    #   - The residual (true_dcdt - reaction_dcdt) is dominated by Phase 1 model errors (R²=0.73)
    #   - These errors are NOT concentration-dependent, so MLP_node learns offset but not slope
    #   - Slope gradient ∝ covariance(concentration, residual), which is weak when errors dominate
    #
    # NEW APPROACH:
    #   1. Use rollout-based loss (trajectory prediction), which worked for offset learning
    #   2. AMPLIFY MLP_node output by large factor (50x) during forward pass
    #      - This makes homeostatic contribution comparable to reaction terms
    #      - Amplification affects slope gradient more than offset because:
    #        slope_gradient ∝ amplification × concentration_variance
    #        offset_gradient ∝ amplification × 1
    #      - With 50x amplification, slope signal becomes resolvable
    #   3. Add auxiliary "slope-encouraging" loss that penalizes flat MLP_node outputs
    #      - Compute MLP_node at multiple concentration values
    #      - Penalize if outputs are too similar (variance too low)
    #      - This breaks the flat initialization without requiring huge LRs
    #
    # After training, the learned weights implicitly include the 1/amplification scaling,
    # so the final MLP_node output is already calibrated to the correct magnitude.
    #
    # References:
    #   - Curriculum learning / loss scaling (Bengio et al. 2009)
    #   - Mixed-precision loss scaling (Micikevicius et al. 2018)
    #   - Gradient magnitude manipulation for weak signals (Goyal et al. 2017)
    if homeostasis_training:
        # ===== Phase 2: Fresh start (no supervised losses) =====
        #
        # Training strategies retained from previous exploration:
        # - Signal amplification (10x) to make homeostatic gradient comparable to reaction gradient
        # - Offset penalty to suppress constant-output solutions
        # - Gradient accumulation (4 micro-batches) for variance reduction
        # - Gradient clipping for BPTT stability
        #
        # REMOVED: supervised contrastive loss (used GT metabolite type labels — label leakage)
        # Embeddings must self-organize from MLP_node behavioral differences only.
        #
        OFFSET_PENALTY_WEIGHT = 2.0  # Keep from Block 3
        GRADIENT_CLIP_NORM = 1.0  # Clip gradient norm to prevent explosion
        AMPLIFICATION_FACTOR = 10.0  # Keep from Block 3
        GRADIENT_ACCUMULATION_STEPS = 4  # NEW: accumulate 4 batches before step

        print(f'\n===== Phase 2: homeostasis training (GRAD ACCUM x{GRADIENT_ACCUMULATION_STEPS}, amp={AMPLIFICATION_FACTOR}x, offset_pen={OFFSET_PENALTY_WEIGHT}, grad_clip={GRADIENT_CLIP_NORM}, {homeostasis_time_step} steps, lr_node={lr_node_homeo}, lr_emb={lr_embedding_homeo}) =====')
        logger.info(f'Phase 2: homeostasis training (GRAD ACCUM x{GRADIENT_ACCUMULATION_STEPS}, amp={AMPLIFICATION_FACTOR}x, grad_clip={GRADIENT_CLIP_NORM})')

        # Freeze reaction params
        for p in k_params:
            p.requires_grad = False
        for p in sub_params:
            p.requires_grad = False
        if hasattr(model, 'sto_all'):
            model.sto_all.requires_grad = False

        # Re-initialize node_func hidden layers (checkpoint has all-zero weights
        # from original init, which kills gradient flow through Tanh).
        # Keep output layer at zero so MLP_node starts with zero output.
        with torch.no_grad():
            modules = list(model.node_func.modules())
            linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
            for layer in linear_layers[:-1]:  # all except last (output) layer
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='tanh')
                nn.init.zeros_(layer.bias)
            # Output layer stays zero (or re-zero if needed)
            linear_layers[-1].weight.zero_()
            linear_layers[-1].bias.zero_()
        print(f'  Re-initialized node_func hidden layers (Kaiming), output layer zeroed')

        # Ensure node params trainable (respect training_single_type for embeddings)
        for name, p in model.named_parameters():
            if 'node_func' in name:
                p.requires_grad = True
            elif name == 'a' and not training_single_type:
                p.requires_grad = True

        # Fresh optimizer for Phase 2 with separate lr for embeddings
        node_func_params = []
        embedding_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name == 'a':
                embedding_params.append(p)
            else:
                node_func_params.append(p)
        optimizer_p2 = torch.optim.Adam([
            {'params': node_func_params, 'lr': lr_node_homeo},
            {'params': embedding_params, 'lr': lr_embedding_homeo},
        ])
        n_p2_params = sum(p.numel() for p in node_func_params) + sum(p.numel() for p in embedding_params)
        print(f'  Phase 2 trainable params: {n_p2_params:,}')

        # Helper function to compute AMPLIFIED forward pass for Phase 2
        # We temporarily scale MLP_node output by AMPLIFICATION_FACTOR to make
        # the homeostatic signal comparable to reaction dynamics
        def compute_dcdt_amplified(model, x_tensor, amplification):
            """Compute dc/dt with amplified MLP_node contribution.

            The amplification factor makes homeostatic gradient comparable to reaction gradient,
            allowing the optimizer to learn the slope (not just offset) of homeostatic regulation.
            """
            concentrations = x_tensor[:, 3]
            external_input = x_tensor[:, 4]

            # === MLP_sub: substrate contribution (frozen) ===
            c_sub = concentrations[model.met_sub].unsqueeze(-1)
            s_abs = model.sto_all[model.sub_to_all].abs().unsqueeze(-1)
            msg_in = torch.cat([c_sub, s_abs], dim=-1)
            msg = model.substrate_func(msg_in)

            # Aggregation
            if model.aggr_type == 'mul':
                eps = 1e-8
                log_msg = torch.log(msg.abs().clamp(min=eps))
                log_agg = torch.zeros(model.n_rxn, dtype=msg.dtype, device=msg.device)
                log_agg.index_add_(0, model.rxn_sub, log_msg.squeeze(-1))
                agg = torch.exp(log_agg)
            else:
                agg = torch.zeros(model.n_rxn, dtype=msg.dtype, device=msg.device)
                agg.index_add_(0, model.rxn_sub, msg.squeeze(-1))

            # Reaction rates
            k = torch.pow(10.0, model.log_k)
            base_v = model.softplus(agg)

            # External modulation
            if model.external_input_mode == "multiplicative_substrate":
                ext_src = external_input[model.met_sub]
                ext_agg = torch.zeros(model.n_rxn, dtype=ext_src.dtype, device=ext_src.device)
                ext_agg.index_add_(0, model.rxn_sub, ext_src)
                ext_mean = ext_agg / model.n_sub_per_rxn
                v = k * ext_mean * base_v
            elif model.external_input_mode == "multiplicative_product":
                ext_src = external_input[model.met_prod]
                ext_agg = torch.zeros(model.n_rxn, dtype=ext_src.dtype, device=ext_src.device)
                ext_agg.index_add_(0, model.rxn_prod, ext_src)
                ext_mean = ext_agg / model.n_prod_per_rxn
                v = k * ext_mean * base_v
            else:
                v = k * base_v

            # dc/dt from reactions
            contrib = model.sto_all * v[model.rxn_all]
            dxdt_reaction = torch.zeros(model.n_met, dtype=contrib.dtype, device=contrib.device)
            dxdt_reaction.index_add_(0, model.met_all, contrib)

            # Add external additive input if applicable
            if model.external_input_mode == "additive":
                dxdt_reaction = dxdt_reaction + external_input

            # === MLP_node: homeostatic contribution (trainable, AMPLIFIED) ===
            node_in = torch.cat([concentrations.unsqueeze(-1), model.a], dim=-1)
            node_out = model.node_func(node_in).squeeze(-1)
            # AMPLIFY the homeostatic signal so it's comparable to reaction dynamics
            dxdt_homeo = node_out * amplification

            return dxdt_reaction + dxdt_homeo

        def compute_offset_penalty(model, n_met, device):
            """Penalize large mean MLP_node output to prevent offset explosion.

            The trajectory loss has O(1) gradient for offset learning but only
            O(concentration_variance) gradient for slope learning. This penalty
            suppresses the easy offset solution, forcing the optimizer to learn slope.

            Returns: mean squared output (offset penalty)
            """
            # Sample at middle concentration (baseline region)
            c_mid = torch.full((n_met,), 5.0, device=device)
            node_in = torch.cat([c_mid.unsqueeze(-1), model.a], dim=-1)
            out = model.node_func(node_in).squeeze(-1)
            # Penalize large mean output (offset) - we want output near zero at baseline
            offset_penalty = (out ** 2).mean()
            return offset_penalty

        Niter_p2 = int(n_frames * data_augmentation_loop // batch_size * 0.2) // homeostasis_time_step
        # Adjust iterations for gradient accumulation (more micro-batches, same effective updates)
        Niter_p2 = Niter_p2 * GRADIENT_ACCUMULATION_STEPS
        plot_frequency_p2 = max(1, Niter_p2 // 20)
        print(f'  {Niter_p2} micro-iterations ({Niter_p2 // GRADIENT_ACCUMULATION_STEPS} optimizer steps), plot/save every {plot_frequency_p2} iterations')
        total_loss_p2 = 0
        p2_label = f'p2_{n_epochs - 1}'
        pbar = trange(Niter_p2, ncols=100, desc='phase2')

        for N in pbar:
            # GRADIENT ACCUMULATION: only zero_grad at start of accumulation window
            if N % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer_p2.zero_grad()
            loss = torch.zeros(1, device=device)
            run = np.random.randint(n_runs)

            for batch in range(batch_size):
                # Sample starting frame for rollout
                k = np.random.randint(n_frames - 4 - homeostasis_time_step)
                x = x_list[run][k].clone()

                # inject external input from SIREN (same as Phase 1)
                if has_visual_field and hasattr(model, 'NNR_f'):
                    visual_input = model.forward_visual(x, k)
                    x[:n_input_metabolites, 4:5] = visual_input
                    x[n_input_metabolites:, 4:5] = 0
                elif model_f is not None:
                    if external_input_type == 'visual':
                        x[:n_input_metabolites, 4:5] = model_f(
                            time=k / n_frames
                        ) ** 2
                        x[n_input_metabolites:, 4:5] = 1
                    else:
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

                # BLOCK 4 STRATEGY: Offset Suppression + Direct Slope Supervision
                # Euler rollout with reduced amplification (10x instead of 50x)
                batch_loss = torch.zeros(1, device=device)
                x_curr = x.clone()

                for step in range(homeostasis_time_step):
                    frame_idx = k + step
                    if frame_idx + 1 >= n_frames:
                        break

                    # Ground truth next concentration
                    c_true_next = x_list[run][frame_idx + 1, :, 3]

                    # Compute dc/dt with AMPLIFIED MLP_node
                    dcdt_amplified = compute_dcdt_amplified(model, x_curr, AMPLIFICATION_FACTOR)

                    # Euler step: c_next = c_curr + dcdt * delta_t
                    c_pred_next = x_curr[:, 3] + dcdt_amplified * delta_t

                    # Trajectory prediction loss
                    batch_loss = batch_loss + ((c_pred_next - c_true_next) ** 2).mean()

                    # Update x_curr for next step (use PREDICTED concentration, not true)
                    # NOTE: Removed .detach() to enable BPTT through full rollout
                    # This allows gradient to accumulate across time steps for slope learning
                    x_curr = x_curr.clone()
                    x_curr[:, 3] = c_pred_next  # allow gradient flow through time steps

                loss = loss + batch_loss / homeostasis_time_step

            # Block 3 Strategy: Offset suppression + Scheduled Contrastive + Grad Clip
            # 1. Offset penalty: suppress large constant outputs (reduced weight)
            offset_penalty = compute_offset_penalty(model, n_metabolites, device)
            loss = loss + OFFSET_PENALTY_WEIGHT * offset_penalty

            # GRADIENT ACCUMULATION: scale loss by 1/N before backward
            # This ensures gradients are averaged across accumulation window
            scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaled_loss.backward()

            total_loss_p2 += loss.item()

            # Only step optimizer at end of accumulation window
            if (N + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # 3. Gradient clipping to prevent explosion (Pascanu et al. 2013)
                # This is critical for BPTT stability through long rollouts
                torch.nn.utils.clip_grad_norm_(
                    list(node_func_params) + list(embedding_params),
                    GRADIENT_CLIP_NORM
                )

                # --- DEBUG: print gradient magnitudes for Phase 2 parameters ---
                if N % max(1, Niter_p2 // 10) == 0:
                    print(f'\n  [DEBUG iter {N}] loss={loss.item():.6f}  offset_pen={offset_penalty.item():.6f}  accum_step={GRADIENT_ACCUMULATION_STEPS}')
                    # embedding gradient
                    if model.a.grad is not None:
                        print(f'    embedding a: grad_norm={model.a.grad.norm().item():.8f}, '
                              f'grad_max={model.a.grad.abs().max().item():.8f}, '
                              f'param_norm={model.a.data.norm().item():.6f}')
                    else:
                        print(f'    embedding a: NO GRADIENT')
                    # node_func layer-by-layer
                    for i, layer in enumerate(model.node_func):
                        if hasattr(layer, 'weight'):
                            w = layer.weight
                            b = layer.bias
                            wg = w.grad.norm().item() if w.grad is not None else 0
                            bg = b.grad.norm().item() if b.grad is not None else 0
                            print(f'    node_func[{i}] Linear: '
                                  f'weight={w.data.norm().item():.8f} grad={wg:.8f}, '
                                  f'bias={b.data.norm().item():.8f} grad={bg:.8f}, '
                                  f'hidden_act_zero={((w.data.norm(dim=1) == 0).sum().item())}')

                optimizer_p2.step()

            # --- Phase 2 periodic plots + checkpoint ---
            if N % plot_frequency_p2 == 0 or N == 0:
                with torch.no_grad():
                    # rate constant comparison (should be stable — k frozen)
                    if freeze_stoichiometry and gt_model is not None:
                        p2_r2, _, _, _ = _plot_rate_constants_comparison(
                            model, gt_model, log_dir, p2_label, N,
                            device=device,
                        )
                    # MLP functions (MLP_sub frozen, MLP_node evolving)
                    plot_x = x_list[0][0]
                    _plot_metabolism_mlp_functions(
                        model, plot_x, xnorm, log_dir, p2_label, N,
                        device, gt_model=gt_model,
                    )
                # save checkpoint
                torch.save(
                    {'model_state_dict': model.state_dict()},
                    os.path.join(
                        log_dir, 'models',
                        f'best_model_with_{n_runs - 1}_graphs_{p2_label}_{N}.pt',
                    ),
                )
                # progress bar
                if freeze_stoichiometry and gt_model is not None:
                    pbar.set_postfix_str(f'R2={p2_r2:.3f} loss={loss.item()/n_metabolites:.4f}')
                else:
                    pbar.set_postfix_str(f'loss={loss.item()/n_metabolites:.4f}')

        # Phase 2 summary
        p2_loss = total_loss_p2 / n_metabolites
        with torch.no_grad():
            sample_x = x_list[0][0]
            node_in = torch.cat([sample_x[:, 3:4], model.a], dim=-1)
            node_mag = model.node_func(node_in).squeeze(-1).abs().mean().item()
        print(f'  phase2: loss={p2_loss:.6f}, MLP_node_mag={node_mag:.6f}')
        logger.info(f'Phase2: loss={p2_loss:.6f}, MLP_node_mag={node_mag:.6f}')

        # Unfreeze all for downstream analysis (_compare_functions, data_test)
        for p in k_params:
            p.requires_grad = True
        for p in sub_params:
            p.requires_grad = True
        if not freeze_stoichiometry and hasattr(model, 'sto_all'):
            model.sto_all.requires_grad = True

        if log_file is not None:
            log_file.write(f"phase2_loss: {p2_loss:.6f}\n")
            log_file.write(f"phase2_node_magnitude: {node_mag:.6f}\n")

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
    eps = getattr(train_config, 'cluster_distance_threshold', 0.1)
    func_metrics = _compare_functions(model, gt_model, x, device,
                                      cluster_distance_threshold=eps)

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
            sr = func_metrics[f'MLP_node_slope_ratio_{t}']
            or_ = func_metrics[f'MLP_node_offset_ratio_{t}']
            ls = func_metrics[f'MLP_node_slope_{t}']
            gs = func_metrics[f'MLP_node_gt_slope_{t}']
            lo = func_metrics[f'MLP_node_offset_{t}']
            go = func_metrics[f'MLP_node_gt_offset_{t}']
            print(f"  MLP_node type {t}: slope_ratio={sr:.4f} offset_ratio={or_:.4f} (slope={ls:.6f}/{gs:.6f}, offset={lo:.6f}/{go:.6f})")
    if 'embedding_cluster_acc' in func_metrics:
        acc = func_metrics['embedding_cluster_acc']
        nc = func_metrics['embedding_n_clusters']
        sil_str = ''
        if 'embedding_silhouette' in func_metrics:
            sil_str = f', silhouette={func_metrics["embedding_silhouette"]:.4f}'
        print(f"  embedding clusters: {nc} found, accuracy={acc:.4f}{sil_str}")
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
                log_file.write(f"MLP_node_slope_ratio_{t}: {func_metrics[f'MLP_node_slope_ratio_{t}']:.4f}\n")
                log_file.write(f"MLP_node_offset_ratio_{t}: {func_metrics[f'MLP_node_offset_ratio_{t}']:.4f}\n")
                log_file.write(f"MLP_node_slope_{t}: {func_metrics[f'MLP_node_slope_{t}']:.6f}\n")
                log_file.write(f"MLP_node_gt_slope_{t}: {func_metrics[f'MLP_node_gt_slope_{t}']:.6f}\n")
                log_file.write(f"MLP_node_offset_{t}: {func_metrics[f'MLP_node_offset_{t}']:.6f}\n")
                log_file.write(f"MLP_node_gt_offset_{t}: {func_metrics[f'MLP_node_gt_offset_{t}']:.6f}\n")
        if 'embedding_cluster_acc' in func_metrics:
            log_file.write(f"embedding_cluster_acc: {func_metrics['embedding_cluster_acc']:.4f}\n")
            log_file.write(f"embedding_n_clusters: {func_metrics['embedding_n_clusters']}\n")
            if 'embedding_silhouette' in func_metrics:
                log_file.write(f"embedding_silhouette: {func_metrics['embedding_silhouette']:.4f}\n")


def data_test_metabolism(config, best_model=20, n_rollout_frames=2000, device=None, log_file=None):
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

    # --- one-step derivative kinograph: feed true state at each timestep ---
    print('one-step derivative kinograph ...')
    deriv_gt_arr = np.zeros((n_metabolites, n_test_frames))
    deriv_pred_arr = np.zeros((n_metabolites, n_test_frames))

    run = 0
    with torch.no_grad():
        for t in trange(n_test_frames, desc='one-step', ncols=100):
            frame_idx = start_frame + t
            if frame_idx >= n_frames - 2:
                break

            # true state and true derivative at this frame
            x = torch.tensor(x_list[run][frame_idx], dtype=torch.float32, device=device)
            y_gt = torch.tensor(y_list[run][frame_idx], dtype=torch.float32, device=device)

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

            # model prediction: dc/dt
            dataset = pyg_Data(x=x, pos=x[:, 1:3])
            pred = model(dataset)

            deriv_gt_arr[:, t] = to_numpy(y_gt[:n_metabolites].squeeze())
            deriv_pred_arr[:, t] = to_numpy(pred[:n_metabolites].squeeze())

    # save derivative kinograph matrices
    np.save(os.path.join(results_dir, 'deriv_kinograph_gt.npy'), deriv_gt_arr)
    np.save(os.path.join(results_dir, 'deriv_kinograph_pred.npy'), deriv_pred_arr)

    # compute derivative metrics
    from MetabolismGraph.models.utils import compute_kinograph_metrics
    deriv_metrics = compute_kinograph_metrics(deriv_gt_arr, deriv_pred_arr)
    print(f"deriv R²={deriv_metrics['r2']:.4f}, SSIM={deriv_metrics['ssim']:.4f}, WD={deriv_metrics['mean_wasserstein']:.4f}")

    # 2x2 derivative kinograph montage
    vmax_deriv = max(np.abs(deriv_gt_arr).max(), np.abs(deriv_pred_arr).max())
    deriv_residual = deriv_gt_arr - deriv_pred_arr
    vmax_deriv_res = np.abs(deriv_residual).max()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_gt_d, ax_pred_d, ax_res_d, ax_scat_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    im_gt_d = ax_gt_d.imshow(deriv_gt_arr, aspect='auto', cmap='viridis', vmin=-vmax_deriv, vmax=vmax_deriv, origin='lower', interpolation='nearest')
    ax_gt_d.set_ylabel('metabolites', fontsize=14)
    ax_gt_d.set_xticks([0, n_test_frames - 1]); ax_gt_d.set_xticklabels([0, n_test_frames], fontsize=12)
    ax_gt_d.set_yticks([0, n_metabolites - 1]); ax_gt_d.set_yticklabels([1, n_metabolites], fontsize=12)
    ax_gt_d.text(0.02, 0.97, 'true dc/dt', transform=ax_gt_d.transAxes, fontsize=9,
                 color='white', va='top', ha='left')
    fig.colorbar(im_gt_d, ax=ax_gt_d, fraction=0.046, pad=0.04).ax.tick_params(labelsize=12)

    im_pred_d = ax_pred_d.imshow(deriv_pred_arr, aspect='auto', cmap='viridis', vmin=-vmax_deriv, vmax=vmax_deriv, origin='lower', interpolation='nearest')
    ax_pred_d.set_xticks([0, n_test_frames - 1]); ax_pred_d.set_xticklabels([0, n_test_frames], fontsize=12)
    ax_pred_d.set_yticks([0, n_metabolites - 1]); ax_pred_d.set_yticklabels([1, n_metabolites], fontsize=12)
    ax_pred_d.text(0.02, 0.97, 'GNN one-step dc/dt', transform=ax_pred_d.transAxes, fontsize=9,
                   color='white', va='top', ha='left')
    fig.colorbar(im_pred_d, ax=ax_pred_d, fraction=0.046, pad=0.04).ax.tick_params(labelsize=12)

    im_res_d = ax_res_d.imshow(deriv_residual, aspect='auto', cmap='RdBu_r', vmin=-vmax_deriv_res, vmax=vmax_deriv_res, origin='lower', interpolation='nearest')
    ax_res_d.set_ylabel('metabolites', fontsize=14)
    ax_res_d.set_xlabel('time', fontsize=14)
    ax_res_d.set_xticks([0, n_test_frames - 1]); ax_res_d.set_xticklabels([0, n_test_frames], fontsize=12)
    ax_res_d.set_yticks([0, n_metabolites - 1]); ax_res_d.set_yticklabels([1, n_metabolites], fontsize=12)
    fig.colorbar(im_res_d, ax=ax_res_d, fraction=0.046, pad=0.04).ax.tick_params(labelsize=12)

    gt_deriv_flat = deriv_gt_arr.flatten()
    pred_deriv_flat = deriv_pred_arr.flatten()
    ax_scat_d.scatter(gt_deriv_flat, pred_deriv_flat, s=1, alpha=0.1, c='k', rasterized=True, edgecolors=None)
    lim_d = [min(gt_deriv_flat.min(), pred_deriv_flat.min()), max(gt_deriv_flat.max(), pred_deriv_flat.max())]
    ax_scat_d.plot(lim_d, lim_d, 'r--', linewidth=2)
    ax_scat_d.set_xlim(lim_d); ax_scat_d.set_ylim(lim_d)
    ax_scat_d.set_xlabel('true dc/dt', fontsize=14)
    ax_scat_d.set_ylabel('predicted dc/dt', fontsize=14)
    ax_scat_d.tick_params(labelsize=12)
    ax_scat_d.text(0.05, 0.95, f'R²={deriv_metrics["r2"]:.3f}\nSSIM={deriv_metrics["ssim"]:.3f}\nWD={deriv_metrics["mean_wasserstein"]:.3f}',
                   transform=ax_scat_d.transAxes, fontsize=12, va='top')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'deriv_kinograph_montage.png'), dpi=150, bbox_inches='tight')
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

    # --- derivative traces plot: true dc/dt vs GNN one-step dc/dt ---
    deriv_r2_per_trace = []
    for idx in trace_ids:
        gt_d = deriv_gt_arr[idx]
        pred_d = deriv_pred_arr[idx]
        ss_res = np.sum((gt_d - pred_d) ** 2)
        ss_tot = np.sum((gt_d - np.mean(gt_d)) ** 2)
        deriv_r2_per_trace.append(1 - (ss_res / ss_tot) if ss_tot > 0 else 0)

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    offset_d = np.abs(deriv_gt_arr[trace_ids]).max() * 1.5
    if offset_d == 0:
        offset_d = 1.0

    for j, n_idx in enumerate(trace_ids):
        y0 = j * offset_d
        baseline_d = np.mean(deriv_gt_arr[n_idx])
        ax.plot(deriv_gt_arr[n_idx] - baseline_d + y0, color='green', lw=3.0, alpha=0.9)
        ax.plot(deriv_pred_arr[n_idx] - baseline_d + y0, color='red', lw=1.0, alpha=0.9)

        ax.text(-n_test_frames * 0.02, y0, str(n_idx), fontsize=10, va='center', ha='right')

        r2_val = deriv_r2_per_trace[j]
        r2_color = 'red' if r2_val < 0.5 else ('orange' if r2_val < 0.8 else 'green')
        ax.text(n_test_frames * 1.02, y0, f'R²:{r2_val:.2f}', fontsize=9, va='center', ha='left', color=r2_color)

    ax.set_xlim([-n_test_frames * 0.05, n_test_frames * 1.1])
    ax.set_ylim([-offset_d, n_traces * offset_d])
    ax.set_xlabel('frame', fontsize=14)
    ax.set_ylabel('metabolite', fontsize=14)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(0, n_test_frames)
    ax.set_yticks([])

    from matplotlib.lines import Line2D
    legend_d = [Line2D([0], [0], color='green', lw=3, label='true dc/dt'),
                Line2D([0], [0], color='red', lw=1, label='GNN one-step dc/dt')]
    ax.legend(handles=legend_d, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'deriv_traces.png'), dpi=150)
    plt.close()

    # --- derivative residual traces plot: (GNN one-step dc/dt) - (true dc/dt) ---
    residual_arr = deriv_pred_arr - deriv_gt_arr  # (n_metabolites, n_frames)

    # compute GT homeostasis baseline: λ_type(i) * (c_i(t) - c_base_type(i))
    # residual should equal +λ(c - c_base) since pred - true = -homeostasis = +λ(c - c_base)
    lambda_per_type = getattr(simulation_config, 'homeostatic_lambda_per_type', None)
    baseline_per_type = getattr(simulation_config, 'homeostatic_baseline_per_type', None)
    gt_homeostasis_arr = None
    if lambda_per_type is not None and baseline_per_type is not None:
        gt_homeostasis_arr = np.zeros((n_metabolites, n_test_frames))
        metabolite_types = x_list[0][0][:n_metabolites, 6].astype(int)
        for t in range(n_test_frames):
            frame_idx = start_frame + t
            if frame_idx >= n_frames - 2:
                break
            conc = x_list[0][frame_idx][:n_metabolites, 3]
            for i in range(n_metabolites):
                mt = metabolite_types[i]
                gt_homeostasis_arr[i, t] = lambda_per_type[mt] * (conc[i] - baseline_per_type[mt])

    n_plot_frames = min(600, n_test_frames)  # show 600 frames like kinograph
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # scale traces: use median of per-trace max amplitudes for tighter spacing
    trace_amps = [np.abs(residual_arr[n_idx, :n_plot_frames]).max() for n_idx in trace_ids]
    median_amp = np.median(trace_amps) if np.median(trace_amps) > 0 else 1.0
    scale = 0.8 / median_amp  # normalize so median trace fills ~80% of spacing
    offset_r = 1.0  # fixed unit spacing between traces

    for j, n_idx in enumerate(trace_ids):
        y0 = j * offset_r
        ax.axhline(y=y0, color='gray', lw=0.3, alpha=0.5)
        ax.plot(residual_arr[n_idx, :n_plot_frames] * scale + y0, color='#e74c3c', lw=1.0, alpha=0.9)
        if gt_homeostasis_arr is not None:
            ax.plot(gt_homeostasis_arr[n_idx, :n_plot_frames] * scale + y0, color='#2ecc71', lw=1.5, alpha=0.8)

        ax.text(-n_plot_frames * 0.02, y0, str(n_idx), fontsize=10, va='center', ha='right')

        mae_val = np.mean(np.abs(residual_arr[n_idx, :n_plot_frames]))
        ax.text(n_plot_frames * 1.02, y0, f'MAE:{mae_val:.3f}', fontsize=9, va='center', ha='left', color='#e74c3c')

    ax.set_xlim([-n_plot_frames * 0.05, n_plot_frames * 1.1])
    ax.set_ylim([-offset_r, n_traces * offset_r])
    ax.set_xlabel('frame', fontsize=14)
    ax.set_ylabel('metabolite', fontsize=14)
    ax.set_title('Derivative residuals: GNN one-step dc/dt − true dc/dt', fontsize=14)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(0, n_plot_frames)
    ax.set_yticks([])

    from matplotlib.lines import Line2D
    legend_r = [Line2D([0], [0], color='#e74c3c', lw=1, label='pred − true dc/dt'),
                Line2D([0], [0], color='#2ecc71', lw=1.5, label='GT: λ(c − c_base)'),
                Line2D([0], [0], color='gray', lw=0.3, label='zero baseline')]
    ax.legend(handles=legend_r, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'deriv_residual_traces.png'), dpi=150)
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
        log_file.write(f"deriv_kinograph_R2: {deriv_metrics['r2']:.4f}\n")
        log_file.write(f"deriv_kinograph_SSIM: {deriv_metrics['ssim']:.4f}\n")
        log_file.write(f"deriv_kinograph_Wasserstein: {deriv_metrics['mean_wasserstein']:.4f}\n")


def _plot_metabolism_mlp_functions(model, x, xnorm, log_dir, epoch, N, device,
                                  gt_model=None):
    """plot learned MLP_sub, MLP_node, and embeddings during metabolism training.

    MLP_sub: sweep concentration at several fixed |stoich| values -> ||output||.
    MLP_node: per-metabolite-type homeostasis function (c_i, a_i) -> homeostasis term.
    embeddings: scatter plot of learned metabolite embeddings a_i colored by type.
    if gt_model is provided, overlay ground-truth curves as dashed lines.

    saves to tmp_training/function/ and tmp_training/embedding/.
    """
    func_dir = f"./{log_dir}/tmp_training/function"
    sub_dir = f"{func_dir}/MLP_sub"
    rate_dir = f"{func_dir}/MLP_node"
    emb_dir = f"./{log_dir}/tmp_training/embedding"
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
    type_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    with torch.no_grad():
        metabolite_types = x[:, 6].long()
        n_types = metabolite_types.max().item() + 1
        n_met = model.a.shape[0]

        fig, ax = plt.subplots(figsize=(8, 8))

        # plot MLP_node for each individual metabolite, collect linear fits
        fits_by_type = {}  # {type: [(slope, intercept), ...]}
        for i in range(n_met):
            a_i = model.a[i]  # embedding for metabolite i
            t = metabolite_types[i].item()

            # sweep concentration with this metabolite's embedding
            a_repeated = a_i.unsqueeze(0).expand(n_pts, -1)
            node_in = torch.cat([conc_range.unsqueeze(-1), a_repeated], dim=-1)
            homeostasis = model.node_func(node_in).squeeze(-1)

            h_np = to_numpy(homeostasis)
            ax.plot(c_np, h_np, linewidth=1, color=type_colors[t % len(type_colors)],
                    alpha=0.3)

            # linear fit y = a*x + b
            coeffs = np.polyfit(c_np, h_np, 1)  # [slope, intercept]
            fits_by_type.setdefault(t, []).append(coeffs)

        # plot individual linear fits with transparency
        for t in sorted(fits_by_type.keys()):
            for slope, intercept in fits_by_type[t]:
                fit_line = slope * c_np + intercept
                ax.plot(c_np, fit_line, linewidth=0.5,
                        color=type_colors[t % len(type_colors)],
                        alpha=0.15, linestyle='-')

        # plot mean linear fit per type (thick)
        for t in sorted(fits_by_type.keys()):
            slopes = [f[0] for f in fits_by_type[t]]
            intercepts = [f[1] for f in fits_by_type[t]]
            mean_slope = np.mean(slopes)
            mean_intercept = np.mean(intercepts)
            mean_fit = mean_slope * c_np + mean_intercept
            ax.plot(c_np, mean_fit, linewidth=2.5,
                    color=type_colors[t % len(type_colors)],
                    linestyle='-', label=f'fit type {t}: slope={mean_slope:.4f}')

        # GT homeostasis if available: -λ_t * (c - c_baseline_t)
        if gt_model is not None and hasattr(gt_model, 'p'):
            p = to_numpy(gt_model.p.detach().cpu())
            for t in range(n_types):
                if t < p.shape[0]:
                    lambda_t = p[t, 0]
                    c_baseline_t = p[t, 1]
                    gt_homeostasis = -lambda_t * (c_np - c_baseline_t)
                    ax.plot(c_np, gt_homeostasis, linewidth=2,
                            color=type_colors[t % len(type_colors)],
                            linestyle='--',
                            label=f'GT type {t}: slope={-lambda_t:.4f}')
                    ax.axvline(x=c_baseline_t,
                               color=type_colors[t % len(type_colors)],
                               linestyle=':', alpha=0.3)

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

        fig, ax = plt.subplots(figsize=(8, 8))

        a_np = to_numpy(model.a.detach().cpu())
        for t in range(n_types):
            pos = torch.argwhere(metabolite_types == t).squeeze(-1)
            if len(pos) == 0:
                continue
            pos_np = to_numpy(pos)
            ax.scatter(a_np[pos_np, 0], a_np[pos_np, 1], s=50,
                       color=type_colors[t % len(type_colors)],
                       alpha=0.7, edgecolors=None, label=f'type {t}')

        ax.set_xlabel('embedding 0', fontsize=14)
        ax.set_ylabel('embedding 1', fontsize=14)
        ax.legend(fontsize=10)
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


def _compare_functions(model, gt_model, x, device, cluster_distance_threshold=0.1):
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
        MLP_node_offset_t: mean learned offset (intercept) for type t
        MLP_node_gt_offset_t: GT offset (λ_t * c_baseline_t) for type t
        MLP_node_slope_ratio_t: learned_slope / gt_slope (best = 1.0)
        MLP_node_offset_ratio_t: learned_offset / gt_offset (best = 1.0)
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
      with torch.no_grad():
        p_node = gt_model.p.detach().cpu()
        n_met = model.a.shape[0]
        metabolite_types = x[:, 6].long()
        n_types = metabolite_types.max().item() + 1

        conc_sweep = torch.linspace(0, 40, 50, device=device)
        c_np_sweep = conc_sweep.cpu().numpy()

        fits_by_type = {}  # {type_id: [(slope, intercept), ...]}
        for i in range(n_met):
            a_i = model.a[i]
            t = metabolite_types[i].item()
            a_rep = a_i.unsqueeze(0).expand(50, -1)
            node_in = torch.cat([conc_sweep.unsqueeze(-1), a_rep], dim=-1)
            out = model.node_func(node_in).squeeze(-1)
            coeffs = np.polyfit(c_np_sweep, out.cpu().numpy(), 1)  # [slope, intercept]
            fits_by_type.setdefault(t, []).append((coeffs[0], coeffs[1]))

        result['n_types'] = n_types
        for t in range(n_types):
            if t < p_node.shape[0]:
                lambda_t = p_node[t, 0].item()
                c_base_t = p_node[t, 1].item()
                gt_slope = -lambda_t
                gt_offset = lambda_t * c_base_t
                fits = fits_by_type.get(t, [(0.0, 0.0)])
                learned_slope = float(np.mean([f[0] for f in fits]))
                learned_offset = float(np.mean([f[1] for f in fits]))
                slope_ratio = learned_slope / gt_slope if abs(gt_slope) > 1e-10 else 0.0
                offset_ratio = learned_offset / gt_offset if abs(gt_offset) > 1e-10 else 0.0
                result[f'MLP_node_slope_{t}'] = learned_slope
                result[f'MLP_node_gt_slope_{t}'] = gt_slope
                result[f'MLP_node_offset_{t}'] = learned_offset
                result[f'MLP_node_gt_offset_{t}'] = gt_offset
                result[f'MLP_node_slope_ratio_{t}'] = slope_ratio
                result[f'MLP_node_offset_ratio_{t}'] = offset_ratio

    # --- Embedding cluster accuracy ---
    # DBSCAN clustering of learned embeddings a_i, compared to GT type labels
    if hasattr(model, 'a') and x is not None:
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.metrics import accuracy_score, silhouette_score
            from scipy.optimize import linear_sum_assignment

            a_np = model.a.detach().cpu().numpy()
            true_labels = x[:, 6].long().cpu().numpy().flatten()

            eps = cluster_distance_threshold
            dbscan = DBSCAN(eps=eps, min_samples=5)
            cluster_labels = dbscan.fit_predict(a_np)

            n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

            # assign noise to separate cluster
            cluster_labels_clean = cluster_labels.copy()
            cluster_labels_clean[cluster_labels_clean == -1] = n_clusters_found

            # optimal label mapping via Hungarian algorithm
            n_true = len(np.unique(true_labels))
            n_found = len(np.unique(cluster_labels_clean))
            confusion = np.zeros((n_true, n_found))
            for i in range(len(true_labels)):
                ti = int(true_labels[i])
                ci = int(cluster_labels_clean[i])
                if 0 <= ti < n_true and 0 <= ci < n_found:
                    confusion[ti, ci] += 1
            row_ind, col_ind = linear_sum_assignment(-confusion)
            mapping = {col_ind[i]: row_ind[i] for i in range(len(col_ind))}
            mapped_labels = np.array([mapping.get(l, -1) for l in cluster_labels_clean])

            accuracy = accuracy_score(true_labels, mapped_labels)
            result['embedding_cluster_acc'] = float(accuracy)
            result['embedding_n_clusters'] = n_clusters_found

            if n_clusters_found > 1:
                sil = silhouette_score(a_np, cluster_labels_clean)
                result['embedding_silhouette'] = float(sil)
        except ImportError:
            pass  # sklearn not available

    return result
