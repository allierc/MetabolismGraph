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


def data_test(config, best_model=20, n_rollout_frames=600, device=None, log_file=None):
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

    external_input_type = simulation_config.external_input_type
    learn_external_input = train_config.learn_external_input
    field_type = model_config.field_type

    has_visual_field = 'visual' in field_type

    log_dir, logger = create_log_dir(config, erase)

    # --- load data ---
    x_list = []
    y_list = []
    for run in trange(0, n_runs, ncols=50):
        x = load_simulation_data(f'graphs_data/{dataset_name}/x_list_{run}')
        y = load_simulation_data(f'graphs_data/{dataset_name}/y_list_{run}')
        x_list.append(x)
        y_list.append(y)

    print(f'dataset: {len(x_list)} run, {len(x_list[0])} frames')

    # --- normalization ---
    activity = torch.tensor(x_list[0][:, :, 3:4], device=device).squeeze()
    distrib = activity.flatten()
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
        if "PDE_M2" in config.graph_model.model_name:
            from MetabolismGraph.generators.PDE_M2 import PDE_M2
            gt_model = PDE_M2(config=config, stoich_graph=stoich_graph, device=device)
        else:
            from MetabolismGraph.generators.PDE_M1 import PDE_M1
            gt_model = PDE_M1(config=config, stoich_graph=stoich_graph, device=device)
        gt_model.load_state_dict(torch.load(gt_model_path, map_location=device))
        gt_model.to(device)
        gt_model.eval()

    # --- create training model ---
    from MetabolismGraph.models.Metabolism_Propagation import Metabolism_Propagation
    model = Metabolism_Propagation(config=config, device=device)
    model.load_stoich_graph(stoich_graph)
    model = model.to(device)

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

    # separate stoichiometric params from MLP/rate params
    stoich_params = []
    other_params = []
    for name, p in model.named_parameters():
        if 'sto_' in name:
            stoich_params.append(p)
        elif 'NNR_f' in name:
            continue  # handled by optimizer_f
        else:
            other_params.append(p)

    param_groups = [{'params': other_params, 'lr': lr}]
    if stoich_params and lr_S > 0:
        param_groups.append({'params': stoich_params, 'lr': lr_S})
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

    # pre-compute per-reaction scatter indices for mass conservation penalty
    rxn_all = model.rxn_all

    print(f'learning rates: lr={lr}, lr_S={lr_S}')
    print(f'S regularization: coeff_S_L1={coeff_S_L1}, coeff_S_integer={coeff_S_integer}, coeff_mass={coeff_S_mass}')
    if n_epochs_init > 0:
        print(f'two-phase: first {n_epochs_init} epochs with L1={first_coeff_L1}, then L1={coeff_S_L1}')

    list_loss = []
    list_loss_regul = []
    loss_components = {'loss': [], 'S_L1': [], 'S_integer': [], 'mass_conservation': []}

    print("start training ...")
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

        last_S_r2 = None
        pbar = trange(Niter, ncols=100)
        for N in pbar:

            optimizer.zero_grad()
            if optimizer_f is not None:
                optimizer_f.zero_grad()

            loss = torch.zeros(1, device=device)
            regul_loss_val = 0.0
            run = np.random.randint(n_runs)

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 4 - time_step)
                x = torch.tensor(
                    x_list[run][k], dtype=torch.float32, device=device
                )

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

                # target: dx/dt
                y = torch.tensor(
                    y_list[run][k], device=device, dtype=torch.float32,
                ) / ynorm

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
                loss_components['S_L1'].append(
                    regul_S_L1.item() / n_metabolites if current_L1 > 0 else 0.0
                )
                loss_components['S_integer'].append(
                    regul_S_int.item() / n_metabolites if coeff_S_integer > 0 else 0.0
                )
                loss_components['mass_conservation'].append(
                    regul_mass.item() / n_metabolites if coeff_S_mass > 0 else 0.0
                )
                plot_dict = {k: v for k, v in loss_components.items() if any(x != 0 for x in v) or k == 'loss'}
                plot_loss(
                    plot_dict, log_dir, epoch=epoch, Niter=N, debug=False,
                    current_loss=current_loss / n_metabolites,
                    current_regul=regul_loss_val / n_metabolites,
                    total_loss=total_loss, total_loss_regul=total_loss_regul,
                )

                # plot learned stoichiometry vs ground truth (scatter + heatmaps)
                with torch.no_grad():
                    last_S_r2 = _plot_stoichiometry_comparison(
                        model, gt_S, stoich_graph, n_metabolites, log_dir,
                        epoch, N,
                    )

                # plot substrate_func and rate_func learned functions
                _plot_metabolism_mlp_functions(
                    model, x, xnorm, log_dir, epoch, N, device,
                    gt_model=gt_model,
                )

                # update progress bar with color-coded R2
                if last_S_r2 is not None:
                    if last_S_r2 > 0.9:
                        r2_color = '\033[92m'   # green
                    elif last_S_r2 > 0.7:
                        r2_color = '\033[93m'   # yellow
                    elif last_S_r2 > 0.3:
                        r2_color = '\033[38;5;208m'  # orange
                    else:
                        r2_color = '\033[91m'   # red
                    pbar.set_postfix_str(
                        f'{r2_color}R\u00b2={last_S_r2:.3f}\033[0m'
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

        # epoch-end plot: loss + stoichiometry comparison
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(list_loss, color='cyan', linewidth=1)
        axes[0].set_xlim([0, n_epochs])
        axes[0].set_ylabel('prediction loss', fontsize=12)
        axes[0].set_xlabel('epochs', fontsize=12)
        axes[0].set_title('Loss', fontsize=14)

        # reconstruct learned S matrix from sto_all
        learned_S = torch.zeros_like(gt_S, device='cpu')
        met_all_cpu = stoich_graph['all'][0].cpu()
        rxn_all_cpu = stoich_graph['all'][1].cpu()
        with torch.no_grad():
            learned_S[met_all_cpu, rxn_all_cpu] = model.sto_all.detach().cpu()

        im1 = axes[1].imshow(
            to_numpy(gt_S.cpu()), aspect='auto', cmap='bwr',
            vmin=-3, vmax=3,
        )
        axes[1].set_title('Ground Truth S', fontsize=14)
        axes[1].set_xlabel('reactions')
        axes[1].set_ylabel('metabolites')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(
            to_numpy(learned_S), aspect='auto', cmap='bwr',
            vmin=-3, vmax=3,
        )
        axes[2].set_title(f'Learned S (epoch {epoch})', fontsize=14)
        axes[2].set_xlabel('reactions')
        axes[2].set_ylabel('metabolites')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        plt.tight_layout()
        plt.savefig(
            f"./{log_dir}/tmp_training/epoch_{epoch}.png", dpi=150,
            bbox_inches='tight',
        )
        plt.close()

    # --- final analysis: compute R2 and write to log ---
    with torch.no_grad():
        final_r2 = _plot_stoichiometry_comparison(
            model, gt_S, stoich_graph, n_metabolites, log_dir,
            epoch='final', N=0,
        )
    final_loss = list_loss[-1] if list_loss else 0.0

    print(f"\n=== training complete ===")
    print(f"  final prediction loss: {final_loss:.6f}")
    print(f"  stoichiometry R2: {final_r2:.4f}")
    logger.info(f"final prediction loss: {final_loss:.6f}")
    logger.info(f"stoichiometry R2: {final_r2:.4f}")

    if log_file is not None:
        log_file.write(f"final_loss: {final_loss:.6f}\n")
        log_file.write(f"stoichiometry_R2: {final_r2:.4f}\n")


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

    # --- stoichiometry R2 (learned vs true S) ---
    with torch.no_grad():
        learned_S = torch.zeros_like(gt_S, device='cpu')
        met_all_cpu = stoich_graph['all'][0].cpu()
        rxn_all_cpu = stoich_graph['all'][1].cpu()
        learned_S[met_all_cpu, rxn_all_cpu] = model.sto_all.detach().cpu()

    # compare only edge coefficients (not full matrix with trivial zeros)
    gt_edges = to_numpy(gt_S.cpu()[met_all_cpu, rxn_all_cpu])
    learned_edges = to_numpy(model.sto_all.detach().cpu())
    n_edges = len(gt_edges)

    stoich_r2 = 0.0
    lin_fit = None
    try:
        lin_fit, _ = curve_fit(linear_model, gt_edges, learned_edges)
        residuals = learned_edges - linear_model(gt_edges, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((learned_edges - np.mean(learned_edges)) ** 2)
        stoich_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    except Exception:
        pass

    print(f'stoichiometry R2: {stoich_r2:.4f} (n={n_edges} edges)')

    # --- scatter plot: true vs learned S (edges only) ---
    out_dir = os.path.join(log_dir, 'tmp_training', 'stoichiometry')
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(gt_edges, learned_edges, s=2, c='k', alpha=0.5)
    ax.set_xlabel(r'true $S_{ij}$', fontsize=18)
    ax.set_ylabel(r'learned $S_{ij}$', fontsize=18)
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
    print(f'  stoichiometry R2: {stoich_r2:.4f}')
    print(f'  test R2 (rollout): {test_r2:.4f}')
    print(f'  test Pearson: {test_pearson:.4f}')

    # --- plot rollout: a few example metabolites ---
    out_dir_rollout = os.path.join(log_dir, 'tmp_training', 'rollout')
    os.makedirs(out_dir_rollout, exist_ok=True)

    n_plot = min(10, n_metabolites)
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 2 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    for i in range(n_plot):
        axes[i].plot(activity_true[i, :], 'k-', linewidth=1, label='true')
        axes[i].plot(activity_pred[i, :], 'r--', linewidth=1, label='predicted')
        axes[i].set_ylabel(f'met {i}', fontsize=8)
        if i == 0:
            axes[i].legend(fontsize=8)
    axes[-1].set_xlabel('frame')
    plt.suptitle(f'rollout (R2={test_r2:.3f}, Pearson={test_pearson:.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_rollout, 'rollout_test.png'), dpi=150)
    plt.close()

    # --- write to analysis log ---
    if log_file is not None:
        log_file.write(f"stoichiometry_R2: {stoich_r2:.4f}\n")
        log_file.write(f"test_R2: {test_r2:.4f}\n")
        log_file.write(f"test_pearson: {test_pearson:.4f}\n")


def _plot_metabolism_mlp_functions(model, x, xnorm, log_dir, epoch, N, device,
                                  gt_model=None):
    """plot learned substrate_func and rate_func functions during metabolism training.

    substrate_func: sweep concentration at several fixed |stoich| values -> ||output||.
    rate_func: compute actual h_rxn for all reactions -> scatter rate vs ||h_rxn||,
               plus 1D sweep along mean h_rxn direction.
    if gt_model is provided, overlay ground-truth curves as dashed lines.

    saves to tmp_training/function/substrate_func/ and tmp_training/function/rate_func/.
    """
    msg_dir = f"./{log_dir}/tmp_training/function/substrate_func"
    rate_dir = f"./{log_dir}/tmp_training/function/rate_func"
    os.makedirs(msg_dir, exist_ok=True)
    os.makedirs(rate_dir, exist_ok=True)

    n_pts = 500

    # --- substrate_func: concentration sweep at fixed |stoich| values ---
    conc_max = to_numpy(xnorm).item() * 3.0
    conc_range = torch.linspace(0, conc_max, n_pts, device=device)

    stoich_values = [0.5, 1.0, 2.0]
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    fig, ax = plt.subplots(figsize=(8, 8))
    with torch.no_grad():
        for s_val, color in zip(stoich_values, colors):
            s_abs = torch.full((n_pts, 1), s_val, device=device)
            msg_in = torch.cat([conc_range.unsqueeze(-1), s_abs], dim=-1)
            msg_out = model.substrate_func(msg_in)
            msg_norm = msg_out.norm(dim=-1)
            ax.plot(to_numpy(conc_range), to_numpy(msg_norm),
                    linewidth=2, color=color, label=f'learned |s|={s_val}')

            if gt_model is not None:
                gt_msg_out = gt_model.substrate_func(msg_in)
                gt_msg_norm = gt_msg_out.norm(dim=-1)
                ax.plot(to_numpy(conc_range), to_numpy(gt_msg_norm),
                        linewidth=2, color=color, linestyle='--', alpha=0.7,
                        label=f'GT |s|={s_val}')

    ax.set_xlabel('concentration', fontsize=24)
    ax.set_ylabel(r'$\|\mathrm{substrate\_func}\|$', fontsize=24)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(f"{msg_dir}/func_{epoch}_{N}.png", dpi=87)
    plt.close()

    # --- rate_func: actual h_rxn scatter + 1D sweep ---
    with torch.no_grad():
        concentrations = x[:, 3]
        x_src = concentrations[model.met_sub].unsqueeze(-1)
        s_abs = model.sto_all[model.sub_to_all].abs().unsqueeze(-1)
        msg_in = torch.cat([x_src, s_abs], dim=-1)
        msg = model.substrate_func(msg_in)

        h_rxn = torch.zeros(
            model.n_rxn, msg.shape[1], dtype=msg.dtype, device=msg.device
        )
        h_rxn.index_add_(0, model.rxn_sub, msg)

        rate = model.softplus(model.rate_func(h_rxn).squeeze(-1))
        h_norm = h_rxn.norm(dim=-1)

        # 1D sweep along mean h_rxn direction
        h_mean = h_rxn.mean(dim=0)
        h_mean_norm = h_mean.norm()
        if h_mean_norm > 1e-8:
            h_dir = h_mean / h_mean_norm
            sweep_scale = torch.linspace(0, h_norm.max().item() * 1.5, n_pts, device=device)
            h_sweep = sweep_scale.unsqueeze(-1) * h_dir.unsqueeze(0)
            rate_sweep = model.softplus(model.rate_func(h_sweep).squeeze(-1))
        else:
            sweep_scale = None

        # ground truth rate_func
        gt_rate = None
        gt_h_norm = None
        gt_sweep_scale = None
        gt_rate_sweep = None
        if gt_model is not None:
            gt_x_src = concentrations[gt_model.met_sub].unsqueeze(-1)
            gt_s_abs = gt_model.sto_sub.unsqueeze(-1)
            gt_msg_in = torch.cat([gt_x_src, gt_s_abs], dim=-1)
            gt_msg = gt_model.substrate_func(gt_msg_in)

            gt_h_rxn = torch.zeros(
                gt_model.n_rxn, gt_msg.shape[1], dtype=gt_msg.dtype, device=gt_msg.device
            )
            gt_h_rxn.index_add_(0, gt_model.rxn_sub, gt_msg)

            gt_rate = gt_model.softplus(gt_model.rate_func(gt_h_rxn).squeeze(-1))
            gt_h_norm = gt_h_rxn.norm(dim=-1)

            gt_h_mean = gt_h_rxn.mean(dim=0)
            gt_h_mean_norm = gt_h_mean.norm()
            if gt_h_mean_norm > 1e-8:
                gt_h_dir = gt_h_mean / gt_h_mean_norm
                gt_sweep_scale = torch.linspace(0, gt_h_norm.max().item() * 1.5, n_pts, device=device)
                gt_h_sweep = gt_sweep_scale.unsqueeze(-1) * gt_h_dir.unsqueeze(0)
                gt_rate_sweep = gt_model.softplus(gt_model.rate_func(gt_h_sweep).squeeze(-1))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(to_numpy(h_norm), to_numpy(rate), s=10, c='k', alpha=0.4,
               label=f'learned (n={model.n_rxn})')
    if sweep_scale is not None:
        ax.plot(to_numpy(sweep_scale), to_numpy(rate_sweep),
                'r-', linewidth=2, alpha=0.8, label='learned sweep')
    if gt_rate is not None:
        ax.scatter(to_numpy(gt_h_norm), to_numpy(gt_rate), s=10, c='tab:blue',
                   alpha=0.3, marker='o', label=f'GT (n={gt_model.n_rxn})')
        if gt_sweep_scale is not None:
            ax.plot(to_numpy(gt_sweep_scale), to_numpy(gt_rate_sweep),
                    'b--', linewidth=2, alpha=0.7, label='GT sweep')
    ax.set_xlabel(r'$\|h_{rxn}\|$ (aggregated message)', fontsize=20)
    ax.set_ylabel(r'$\mathrm{softplus}(\mathrm{rate\_func}(h))$', fontsize=20)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(f"{rate_dir}/func_{epoch}_{N}.png", dpi=87)
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
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].set_xlabel('reactions')
    axes[0].set_ylabel('metabolites')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(
        to_numpy(learned_S), aspect='auto', cmap='bwr', vmin=-3, vmax=3,
    )
    axes[1].set_title(f'Learned (epoch {epoch}, iter {N})', fontsize=12)
    axes[1].set_xlabel('reactions')
    axes[1].set_ylabel('metabolites')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # panel 3: scatter plot true vs learned
    axes[2].scatter(gt_edges, learned_edges, s=2, c='k', alpha=0.5)
    axes[2].set_xlabel(r'true $S_{ij}$', fontsize=12)
    axes[2].set_ylabel(r'learned $S_{ij}$', fontsize=12)

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
