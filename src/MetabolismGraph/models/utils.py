import matplotlib.pyplot as plt
import os
import re
import torch
import numpy as np

from MetabolismGraph.utils import to_numpy, fig_init

# optional siren / lowrank imports - may not be available yet
try:
    from MetabolismGraph.models.Siren_Network import Siren, Siren_Network
except ImportError:
    Siren = None
    Siren_Network = None

try:
    from MetabolismGraph.models.LowRank_INR import LowRankINR
except ImportError:
    LowRankINR = None


def choose_inr_model(config=None, n_metabolites=None, n_frames=None, x_list=None, device=None):
    """
    create inr model for external input reconstruction.

    hierarchy: visual > signal > none
    for signal input: use inr_type to select representation (siren_t, siren_id, siren_x, ngp, lowrank)
    for visual input: use Siren_Network with nnr_f params

    returns None if learn_external_input is False or external_input_type is 'none'
    """
    simulation_config = config.simulation
    model_config = config.graph_model
    train_config = config.training

    external_input_type = simulation_config.external_input_type
    learn_external_input = train_config.learn_external_input
    inr_type = model_config.inr_type

    if not learn_external_input or external_input_type == 'none':
        return None

    model_f = None

    if external_input_type == 'visual':
        # visual input: use Siren_Network with nnr_f params
        n_input_metabolites = simulation_config.n_input_metabolites
        n_input_metabolites_per_axis = int(np.sqrt(n_input_metabolites))
        model_f = Siren_Network(
            image_width=n_input_metabolites_per_axis,
            in_features=model_config.input_size_nnr_f,
            out_features=model_config.output_size_nnr_f,
            hidden_features=model_config.hidden_dim_nnr_f,
            hidden_layers=model_config.n_layers_nnr_f,
            outermost_linear=model_config.outermost_linear_nnr_f,
            device=device,
            first_omega_0=model_config.omega_f,
            hidden_omega_0=model_config.omega_f
        )

    elif external_input_type == 'signal':
        # signal input: use inr_type to select representation
        learnable_omega = model_config.omega_f_learning
        if inr_type == 'siren_t':
            model_f = Siren(
                in_features=1,
                hidden_features=model_config.hidden_dim_nnr_f,
                hidden_layers=model_config.n_layers_nnr_f,
                out_features=n_metabolites,
                outermost_linear=model_config.outermost_linear_nnr_f,
                first_omega_0=model_config.omega_f,
                hidden_omega_0=model_config.omega_f,
                learnable_omega=learnable_omega
            )
        elif inr_type == 'siren_id':
            model_f = Siren(
                in_features=2,  # (t, id)
                hidden_features=model_config.hidden_dim_nnr_f,
                hidden_layers=model_config.n_layers_nnr_f,
                out_features=1,
                outermost_linear=model_config.outermost_linear_nnr_f,
                first_omega_0=model_config.omega_f,
                hidden_omega_0=model_config.omega_f,
                learnable_omega=learnable_omega
            )
        elif inr_type == 'siren_x':
            model_f = Siren(
                in_features=3,  # (t, x, y)
                hidden_features=model_config.hidden_dim_nnr_f,
                hidden_layers=model_config.n_layers_nnr_f,
                out_features=1,
                outermost_linear=model_config.outermost_linear_nnr_f,
                first_omega_0=model_config.omega_f,
                hidden_omega_0=model_config.omega_f,
                learnable_omega=learnable_omega
            )
        elif inr_type == 'lowrank':
            # extract external_input for svd init
            external_input_data = x_list[0][:, :, 4] if x_list is not None else None
            init_data = external_input_data if model_config.lowrank_svd_init else None
            model_f = LowRankINR(
                n_frames=n_frames,
                n_metabolites=n_metabolites,
                rank=model_config.lowrank_rank,
                init_data=init_data
            )

    if model_f is not None:
        model_f.to(device=device)
        print(f'external input model: {inr_type}, external_input_type={external_input_type}')

    return model_f


def constant_batch_size(batch_size):
    def get_batch_size(epoch):
        return batch_size

    return get_batch_size

def increasing_batch_size(batch_size):
    def get_batch_size(epoch):
        return 1 if epoch < 1 else batch_size

    return get_batch_size


def set_trainable_parameters(model=[], lr_embedding=[], lr=[],  lr_update=[], lr_W=[], lr_modulation=[], learning_rate_NNR=[], learning_rate_NNR_f=[], learning_rate_NNR_E=[], learning_rate_NNR_b=[]):

    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params)

    # only count model.a if it exists and requires gradients (not frozen by training_single_type)
    if hasattr(model, 'a') and model.a.requires_grad:
        n_total_params = n_total_params + torch.numel(model.a)


    if lr_update==[]:
        lr_update = lr

    param_groups = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if name == 'a':
                param_groups.append({'params': parameter, 'lr': lr_embedding})
            elif (name=='b') or ('lin_modulation' in name):
                param_groups.append({'params': parameter, 'lr': lr_modulation})
            elif 'lin_phi' in name:
                param_groups.append({'params': parameter, 'lr': lr_update})
            elif 'W' in name:
                param_groups.append({'params': parameter, 'lr': lr_W})
            elif 'NNR_f' in name:
                param_groups.append({'params': parameter, 'lr': learning_rate_NNR_f})
            elif 'NNR' in name:
                param_groups.append({'params': parameter, 'lr': learning_rate_NNR})
            else:
                param_groups.append({'params': parameter, 'lr': lr})

    # use foreach=False to avoid cuda device mismatch issues with multi-gpu setups
    optimizer = torch.optim.Adam(param_groups, foreach=False)

    return optimizer, n_total_params


def analyze_data_svd(x_list, output_folder, config=None, max_components=100, logger=None, max_data_size=10_000_000, max_metabolites=1024, style=None, save_in_subfolder=True, log_file=None):
    """
    perform svd analysis on activity data and external_input (if present).
    uses randomized svd for large datasets for efficiency.
    subsamples frames if data is too large.

    args:
        x_list: numpy array of shape (n_frames, n_metabolites, n_features)
                features: [id, x, y, u, external_input, ...]
        output_folder: path to save plots
        config: config object (optional, for metadata)
        max_components: maximum number of svd components to compute
        logger: optional logger (for training)
        max_data_size: maximum data size before subsampling (default 10M elements)
        max_metabolites: maximum number of metabolites before subsampling (default 1024)
        style: matplotlib style to use (e.g., 'dark_background' for dark mode)
        save_in_subfolder: if True, save to results/ subfolder; if False, save directly to output_folder
        log_file: optional file handle to write results

    returns:
        dict with svd analysis results
    """
    from sklearn.utils.extmath import randomized_svd

    n_frames, n_metabolites, n_features = x_list.shape
    results = {}

    def log_print(msg):
        if logger:
            logger.info(msg)
        if log_file:
            # strip ansi color codes for log file
            clean_msg = re.sub(r'\033\[[0-9;]*m', '', msg)
            log_file.write(clean_msg + '\n')

    # subsample metabolites if too many
    if n_metabolites > max_metabolites:
        metabolite_subsample = int(np.ceil(n_metabolites / max_metabolites))
        metabolite_indices = np.arange(0, n_metabolites, metabolite_subsample)
        x_list = x_list[:, metabolite_indices, :]
        n_metabolites_sampled = len(metabolite_indices)
        log_print(f"subsampling metabolites: {n_metabolites} -> {n_metabolites_sampled} (every {metabolite_subsample}th)")
        n_metabolites = n_metabolites_sampled

    # subsample frames if data is too large
    data_size = n_frames * n_metabolites
    if data_size > max_data_size:
        subsample_factor = int(np.ceil(data_size / max_data_size))
        frame_indices = np.arange(0, n_frames, subsample_factor)
        x_list_sampled = x_list[frame_indices]
        n_frames_sampled = len(frame_indices)
        log_print(f"subsampling frames: {n_frames} -> {n_frames_sampled} (every {subsample_factor}th)")
        data_size_sampled = n_frames_sampled * n_metabolites
    else:
        x_list_sampled = x_list
        n_frames_sampled = n_frames
        data_size_sampled = data_size
        subsample_factor = 1

    # decide whether to use randomized svd
    use_randomized = data_size_sampled > 1e6  # use randomized for > 1M elements

    # store data size info for later printing with results
    if subsample_factor > 1:
        data_info = f"using {n_frames_sampled:,} of {n_frames:,} frames ({n_metabolites:,} metabolites)"
    else:
        data_info = f"using full data ({n_frames:,} frames, {n_metabolites:,} metabolites)"

    # save current style context and apply new style if provided
    if style:
        plt.style.use(style)

    # main color based on style
    mc = 'w' if style == 'dark_background' else 'k'
    bg_color = 'k' if style == 'dark_background' else 'w'

    # font sizes
    TITLE_SIZE = 16
    LABEL_SIZE = 14
    TICK_SIZE = 12
    LEGEND_SIZE = 12

    # prepare figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=bg_color)
    for ax in axes.flat:
        ax.set_facecolor(bg_color)

    # 1. analyze activity (u) - column 3
    activity = x_list_sampled[:, :, 3]  # shape: (n_frames_sampled, n_metabolites)
    log_print("--- activity ---")
    log_print(f"  shape: {activity.shape}")
    log_print(f"  range: [{activity.min():.3f}, {activity.max():.3f}]")

    k = min(max_components, min(n_frames_sampled, n_metabolites) - 1)

    try:
        if use_randomized:
            U_act, S_act, Vt_act = randomized_svd(activity, n_components=k, random_state=42)
        else:
            U_act, S_act, Vt_act = np.linalg.svd(activity, full_matrices=False)
            S_act = S_act[:k]
    except np.linalg.LinAlgError:
        log_print("  SVD did not converge — skipping SVD analysis")
        return {'activity': {'rank_90': None, 'rank_99': None}}

    # compute cumulative variance
    cumvar_act = np.cumsum(S_act**2) / np.sum(S_act**2)
    rank_90_act = np.searchsorted(cumvar_act, 0.90) + 1
    rank_99_act = np.searchsorted(cumvar_act, 0.99) + 1

    log_print(f"  effective rank (90% var): {rank_90_act}")
    log_print(f"  effective rank (99% var): \033[92m{rank_99_act}\033[0m")

    # compression ratio
    if rank_99_act < k:
        compression_act = (n_frames * n_metabolites) / (rank_99_act * (n_frames + n_metabolites))
        log_print(f"  compression (rank-{rank_99_act}): {compression_act:.1f}x")
    else:
        log_print("  compression: need more components to reach 99% variance")

    results['activity'] = {
        'singular_values': S_act,
        'cumulative_variance': cumvar_act,
        'rank_90': rank_90_act,
        'rank_99': rank_99_act,
    }

    # plot activity svd
    ax = axes[0, 0]
    ax.semilogy(S_act, color=mc, lw=1.5)
    ax.set_xlabel('component', fontsize=LABEL_SIZE)
    ax.set_ylabel('singular value', fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(cumvar_act, color=mc, lw=1.5)
    ax.axhline(0.90, color='orange', ls='--', label='90%')
    ax.axhline(0.99, color='green', ls='--', label='99%')
    ax.axvline(rank_90_act, color='orange', ls=':', alpha=0.7)
    ax.axvline(rank_99_act, color='green', ls=':', alpha=0.7)
    ax.set_xlabel('component', fontsize=LABEL_SIZE)
    ax.set_ylabel('cumulative variance', fontsize=LABEL_SIZE)
    ax.legend(loc='lower right', fontsize=LEGEND_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    # 2. analyze external_input (if present and non-zero) - column 4
    input_label = "external input"

    if n_features > 4:
        external_input = x_list_sampled[:, :, 4]  # shape: (n_frames_sampled, n_metabolites)

        # check if external_input has actual signal
        ext_range = external_input.max() - external_input.min()
        if ext_range > 1e-6:
            log_print(f"--- {input_label} ---")
            log_print(f"  shape: {external_input.shape}")
            log_print(f"  range: [{external_input.min():.3f}, {external_input.max():.3f}]")

            if use_randomized:
                U_ext, S_ext, Vt_ext = randomized_svd(external_input, n_components=k, random_state=42)
            else:
                U_ext, S_ext, Vt_ext = np.linalg.svd(external_input, full_matrices=False)
                S_ext = S_ext[:k]

            cumvar_ext = np.cumsum(S_ext**2) / np.sum(S_ext**2)
            rank_90_ext = np.searchsorted(cumvar_ext, 0.90) + 1
            rank_99_ext = np.searchsorted(cumvar_ext, 0.99) + 1

            log_print(f"  effective rank (90% var): {rank_90_ext}")
            log_print(f"  effective rank (99% var): \033[92m{rank_99_ext}\033[0m")

            if rank_99_ext < k:
                compression_ext = (n_frames * n_metabolites) / (rank_99_ext * (n_frames + n_metabolites))
                log_print(f"  compression (rank-{rank_99_ext}): {compression_ext:.1f}x")
            else:
                log_print("  compression: need more components to reach 99% variance")

            results['external_input'] = {
                'singular_values': S_ext,
                'cumulative_variance': cumvar_ext,
                'rank_90': rank_90_ext,
                'rank_99': rank_99_ext,
            }

            # plot external_input svd
            ax = axes[1, 0]
            ax.semilogy(S_ext, color=mc, lw=1.5)
            ax.set_xlabel('component', fontsize=LABEL_SIZE)
            ax.set_ylabel('singular value', fontsize=LABEL_SIZE)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.3)

            ax = axes[1, 1]
            ax.plot(cumvar_ext, color=mc, lw=1.5)
            ax.axhline(0.90, color='orange', ls='--', label='90%')
            ax.axhline(0.99, color='green', ls='--', label='99%')
            ax.axvline(rank_90_ext, color='orange', ls=':', alpha=0.7)
            ax.axvline(rank_99_ext, color='green', ls=':', alpha=0.7)
            ax.set_xlabel('component', fontsize=LABEL_SIZE)
            ax.set_ylabel('cumulative variance', fontsize=LABEL_SIZE)
            ax.legend(loc='lower right', fontsize=LEGEND_SIZE)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.3)
        else:
            log_print(f"--- {input_label} ---")
            log_print("  no external input found (range < 1e-6)")
            axes[1, 0].set_visible(False)
            axes[1, 1].set_visible(False)
            results['external_input'] = None
    else:
        log_print(f"--- {input_label} ---")
        log_print("  not present in data")
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)
        results['external_input'] = None

    plt.tight_layout()

    # save plot
    if save_in_subfolder:
        save_folder = os.path.join(output_folder, 'results')
        os.makedirs(save_folder, exist_ok=True)
    else:
        save_folder = output_folder
    save_path = os.path.join(save_folder, 'svd_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=bg_color)
    plt.close()

    # print svd results: data info + rank results
    if results.get('external_input'):
        print(f"{data_info}, \033[92mactivity rank(99%)={results['activity']['rank_99']}, external_input rank(99%)={results['external_input']['rank_99']}\033[0m")
    else:
        print(f"{data_info}, \033[92mactivity rank(99%)={results['activity']['rank_99']}\033[0m")

    return results


class LossRegularizer:
    """
    handles all regularization terms, coefficient annealing, and history tracking.

    usage:
        regularizer = LossRegularizer(train_config, model_config, activity_column=6,
                                       plot_frequency=100, n_metabolites=1000, trainer_type='signal')

        for epoch in range(n_epochs):
            regularizer.set_epoch(epoch)

            for N in range(Niter):
                regularizer.reset_iteration()

                pred, in_features, msg = model(batch, data_id=data_id, return_all=True)

                regul_loss = regularizer.compute(model, x, in_features, ids, ids_batch, edges, device)
                loss = pred_loss + regul_loss
    """

    # components tracked in history
    COMPONENTS = [
        'W_L1', 'W_L2', 'W_sign',
        'edge_diff', 'edge_norm', 'edge_weight', 'phi_weight',
        'phi_zero', 'update_diff', 'update_msg_diff', 'update_u_diff', 'update_msg_sign',
        'missing_activity', 'model_a', 'model_b', 'modulation'
    ]

    def __init__(self, train_config, model_config, activity_column: int,
                 plot_frequency: int, n_metabolites: int, trainer_type: str = 'signal'):
        """
        args:
            train_config: TrainingConfig with coeff_* values
            model_config: GraphModelConfig with model settings
            activity_column: column index for activity (6 for signal, 3 for flyvis)
            plot_frequency: how often to record to history
            n_metabolites: number of metabolites for normalization
            trainer_type: 'signal' or 'flyvis' - controls annealing behavior
        """
        self.train_config = train_config
        self.model_config = model_config
        self.activity_column = activity_column
        self.plot_frequency = plot_frequency
        self.n_metabolites = n_metabolites
        self.trainer_type = trainer_type

        # current epoch (for annealing)
        self.epoch = 0

        # iteration counter
        self.iter_count = 0

        # per-iteration accumulator
        self._iter_total = 0.0
        self._iter_tracker = {}

        # history for plotting
        self._history = {comp: [] for comp in self.COMPONENTS}
        self._history['regul_total'] = []

        # cache coefficients
        self._coeffs = {}
        self._update_coeffs()

    def _update_coeffs(self):
        """recompute coefficients based on current epoch (annealing for flyvis only)."""
        tc = self.train_config
        epoch = self.epoch

        # two-phase training support
        n_epochs_init = getattr(tc, 'n_epochs_init', 0)
        first_coeff_L1 = getattr(tc, 'first_coeff_L1', tc.coeff_W_L1)

        if self.trainer_type == 'flyvis':
            # flyvis: annealed coefficients
            self._coeffs['W_L1'] = tc.coeff_W_L1 * (1 - np.exp(-tc.coeff_W_L1_rate * epoch))
            self._coeffs['edge_weight_L1'] = tc.coeff_edge_weight_L1 * (1 - np.exp(-tc.coeff_edge_weight_L1_rate ** epoch))
            self._coeffs['phi_weight_L1'] = tc.coeff_phi_weight_L1 * (1 - np.exp(-tc.coeff_phi_weight_L1_rate * epoch))
        else:
            # signal: two-phase training if n_epochs_init > 0
            if n_epochs_init > 0 and epoch < n_epochs_init:
                # phase 1: use first_coeff_L1 (typically 0 or small)
                self._coeffs['W_L1'] = first_coeff_L1
            else:
                # phase 2: use coeff_W_L1 (target L1)
                self._coeffs['W_L1'] = tc.coeff_W_L1
            self._coeffs['edge_weight_L1'] = tc.coeff_edge_weight_L1
            self._coeffs['phi_weight_L1'] = tc.coeff_phi_weight_L1

        # non-annealed coefficients (same for both)
        self._coeffs['W_L2'] = tc.coeff_W_L2
        self._coeffs['W_sign'] = tc.coeff_W_sign
        # two-phase: edge_diff is active in phase 1, disabled in phase 2
        if n_epochs_init > 0 and epoch >= n_epochs_init:
            self._coeffs['edge_diff'] = 0  # phase 2: no monotonicity constraint
        else:
            self._coeffs['edge_diff'] = tc.coeff_edge_diff
        self._coeffs['edge_norm'] = tc.coeff_edge_norm
        self._coeffs['edge_weight_L2'] = tc.coeff_edge_weight_L2
        self._coeffs['phi_weight_L2'] = tc.coeff_phi_weight_L2
        self._coeffs['phi_zero'] = tc.coeff_lin_phi_zero
        self._coeffs['update_diff'] = tc.coeff_update_diff
        self._coeffs['update_msg_diff'] = tc.coeff_update_msg_diff
        self._coeffs['update_u_diff'] = tc.coeff_update_u_diff
        self._coeffs['update_msg_sign'] = tc.coeff_update_msg_sign
        self._coeffs['missing_activity'] = tc.coeff_missing_activity
        self._coeffs['model_a'] = tc.coeff_model_a
        self._coeffs['model_b'] = tc.coeff_model_b
        self._coeffs['modulation'] = tc.coeff_lin_modulation

    def set_epoch(self, epoch: int, plot_frequency: int = None):
        """set current epoch and update annealed coefficients."""
        self.epoch = epoch
        self._update_coeffs()
        if plot_frequency is not None:
            self.plot_frequency = plot_frequency
        # reset iteration counter at epoch start
        self.iter_count = 0

    def reset_iteration(self):
        """reset per-iteration accumulator."""
        self.iter_count += 1
        self._iter_total = 0.0
        self._iter_tracker = {comp: 0.0 for comp in self.COMPONENTS}
        # flag to ensure W_L1 is only applied once per iteration (not per batch item)
        self._W_L1_applied_this_iter = False

    def should_record(self) -> bool:
        """check if we should record to history this iteration."""
        return (self.iter_count % self.plot_frequency == 0) or (self.iter_count == 1)

    def needs_update_regul(self) -> bool:
        """check if update regularization is needed (update_diff, update_msg_diff, update_u_diff, or update_msg_sign)."""
        return (self._coeffs['update_diff'] > 0 or
                self._coeffs['update_msg_diff'] > 0 or
                self._coeffs['update_u_diff'] > 0 or
                self._coeffs['update_msg_sign'] > 0)

    def _add(self, name: str, term):
        """internal: accumulate a regularization term."""
        if term is None:
            return
        val = term.item() if hasattr(term, 'item') else float(term)
        self._iter_total += val
        if name in self._iter_tracker:
            self._iter_tracker[name] += val

    def compute(self, model, x, in_features, ids, ids_batch, edges, device,
                xnorm=1.0, index_weight=None):
        """
        compute all regularization terms internally.

        args:
            model: the neural network model
            x: input tensor
            in_features: features for lin_phi (from model forward pass, can be None)
            ids: sample indices for regularization
            ids_batch: batch indices
            edges: edge tensor
            device: torch device
            xnorm: normalization value
            index_weight: index for W_sign computation (signal only)

        returns:
            total regularization loss tensor
        """
        tc = self.train_config
        mc = self.model_config
        n_metabolites = self.n_metabolites
        total_regul = torch.tensor(0.0, device=device)

        # --- W regularization ---

        low_rank = getattr(model, 'low_rank_factorization', False)
        if low_rank and hasattr(model, 'WL') and hasattr(model, 'WR'):

            if self._coeffs['W_L1'] > 0 and not self._W_L1_applied_this_iter:
                regul_term = (model.WL.norm(1) + model.WR) * self._coeffs['W_L1']
                total_regul = total_regul + regul_term
                self._add('W_L1', regul_term)
                self._W_L1_applied_this_iter = True
        else:

            # W_L1: apply only once per iteration (not per batch item)
            if self._coeffs['W_L1'] > 0 and not self._W_L1_applied_this_iter:
                regul_term = model.W.norm(1) * self._coeffs['W_L1']
                total_regul = total_regul + regul_term
                self._add('W_L1', regul_term)
                self._W_L1_applied_this_iter = True

            if self._coeffs['W_L2'] > 0 and not self._W_L1_applied_this_iter:
                regul_term = model.W.norm(2) * self._coeffs['W_L2']
                total_regul = total_regul + regul_term
                self._add('W_L2', regul_term)

        # --- edge/phi weight regularization ---
        if (self._coeffs['edge_weight_L1'] + self._coeffs['edge_weight_L2']) > 0:
            for param in model.lin_edge.parameters():
                regul_term = param.norm(1) * self._coeffs['edge_weight_L1'] + param.norm(2) * self._coeffs['edge_weight_L2']
                total_regul = total_regul + regul_term
                self._add('edge_weight', regul_term)

        if (self._coeffs['phi_weight_L1'] + self._coeffs['phi_weight_L2']) > 0:
            for param in model.lin_phi.parameters():
                regul_term = param.norm(1) * self._coeffs['phi_weight_L1'] + param.norm(2) * self._coeffs['phi_weight_L2']
                total_regul = total_regul + regul_term
                self._add('phi_weight', regul_term)

        # --- phi_zero regularization ---
        if self._coeffs['phi_zero'] > 0:
            in_features_phi = get_in_features_update(rr=None, model=model, device=device)
            func_phi = model.lin_phi(in_features_phi[ids].float())
            regul_term = func_phi.norm(2) * self._coeffs['phi_zero']
            total_regul = total_regul + regul_term
            self._add('phi_zero', regul_term)

        # --- edge diff/norm regularization ---
        if (self._coeffs['edge_diff'] > 0) | (self._coeffs['edge_norm'] > 0):
            in_features_edge, in_features_edge_next = get_in_features_lin_edge(x, model, mc, xnorm, n_metabolites, device)

            if self._coeffs['edge_diff'] > 0:
                if mc.lin_edge_positive:
                    msg0 = model.lin_edge(in_features_edge[ids].clone().detach()) ** 2
                    msg1 = model.lin_edge(in_features_edge_next[ids].clone().detach()) ** 2
                else:
                    msg0 = model.lin_edge(in_features_edge[ids].clone().detach())
                    msg1 = model.lin_edge(in_features_edge_next[ids].clone().detach())
                regul_term = torch.relu(msg0 - msg1).norm(2) * self._coeffs['edge_diff']
                total_regul = total_regul + regul_term
                self._add('edge_diff', regul_term)

            if self._coeffs['edge_norm'] > 0:
                in_features_edge_norm = in_features_edge.clone()
                in_features_edge_norm[:, 0] = 2 * xnorm
                if mc.lin_edge_positive:
                    msg_norm = model.lin_edge(in_features_edge_norm[ids].clone().detach()) ** 2
                else:
                    msg_norm = model.lin_edge(in_features_edge_norm[ids].clone().detach())
                # different normalization target for signal vs flyvis
                if self.trainer_type == 'signal':
                    regul_term = (msg_norm - 1).norm(2) * self._coeffs['edge_norm']
                else:  # flyvis
                    regul_term = (msg_norm - 2 * xnorm).norm(2) * self._coeffs['edge_norm']
                total_regul = total_regul + regul_term
                self._add('edge_norm', regul_term)

        # --- W_sign (Dale's Law) regularization ---
        if self._coeffs['W_sign'] > 0 and self.epoch > 0:
            W_sign_temp = getattr(tc, 'W_sign_temperature', 10.0)

            if self.trainer_type == 'signal' and index_weight is not None:
                # signal version: uses index_weight
                if self.iter_count % 4 == 0:
                    W_sign = torch.tanh(5 * model.W)
                    loss_contribs = []
                    for i in range(n_metabolites):
                        indices = index_weight[int(i)]
                        if indices.numel() > 0:
                            values = W_sign[indices, i]
                            std = torch.std(values, unbiased=False)
                            loss_contribs.append(std)
                    if loss_contribs:
                        regul_term = torch.stack(loss_contribs).norm(2) * self._coeffs['W_sign']
                        total_regul = total_regul + regul_term
                        self._add('W_sign', regul_term)
            else:
                # flyvis version: uses scatter_add
                weights = model.W.squeeze()
                source_nodes = edges[0]

                n_pos = torch.zeros(n_metabolites, device=device)
                n_neg = torch.zeros(n_metabolites, device=device)
                n_total = torch.zeros(n_metabolites, device=device)

                pos_mask = torch.sigmoid(W_sign_temp * weights)
                neg_mask = torch.sigmoid(-W_sign_temp * weights)

                n_pos.scatter_add_(0, source_nodes, pos_mask)
                n_neg.scatter_add_(0, source_nodes, neg_mask)
                n_total.scatter_add_(0, source_nodes, torch.ones_like(weights))

                violation = torch.where(n_total > 0,
                                        (n_pos / n_total) * (n_neg / n_total),
                                        torch.zeros_like(n_total))
                regul_term = violation.sum() * self._coeffs['W_sign']
                total_regul = total_regul + regul_term
                self._add('W_sign', regul_term)

        # note: update function regularizations (update_msg_diff, update_u_diff, update_msg_sign)
        # are handled by compute_update_regul() which should be called after the model forward pass.
        # call finalize_iteration() after all regularizations are computed to record to history.

        return total_regul

    def _record_to_history(self):
        """append current iteration values to history."""
        n = self.n_metabolites
        self._history['regul_total'].append(self._iter_total / n)
        for comp in self.COMPONENTS:
            self._history[comp].append(self._iter_tracker.get(comp, 0) / n)

    def compute_update_regul(self, model, in_features, ids_batch, device,
                              x=None, xnorm=None, ids=None):
        """
        compute update function regularizations (update_diff, update_msg_diff, update_u_diff, update_msg_sign).

        this method should be called after the model forward pass when in_features is available.

        args:
            model: the neural network model
            in_features: features from model forward pass
            ids_batch: batch indices
            device: torch device
            x: input tensor (required for update_diff with 'generic' update_type)
            xnorm: normalization value (required for update_diff)
            ids: sample indices (required for update_diff)

        returns:
            total update regularization loss tensor
        """
        mc = self.model_config
        embedding_dim = mc.embedding_dim
        n_metabolites = self.n_metabolites
        total_regul = torch.tensor(0.0, device=device)

        # update_diff: for 'generic' update_type only
        if (self._coeffs['update_diff'] > 0) and (model.update_type == 'generic') and (x is not None):
            in_features_edge, in_features_edge_next = get_in_features_lin_edge(
                x, model, mc, xnorm, n_metabolites, device)
            if mc.lin_edge_positive:
                msg0 = model.lin_edge(in_features_edge[ids].clone().detach()) ** 2
                msg1 = model.lin_edge(in_features_edge_next[ids].clone().detach()) ** 2
            else:
                msg0 = model.lin_edge(in_features_edge[ids].clone().detach())
                msg1 = model.lin_edge(in_features_edge_next[ids].clone().detach())
            in_feature_update = torch.cat((torch.zeros((n_metabolites, 1), device=device),
                                           model.a[:n_metabolites], msg0,
                                           torch.ones((n_metabolites, 1), device=device)), dim=1)
            in_feature_update = in_feature_update[ids]
            in_feature_update_next = torch.cat((torch.zeros((n_metabolites, 1), device=device),
                                                model.a[:n_metabolites], msg1,
                                                torch.ones((n_metabolites, 1), device=device)), dim=1)
            in_feature_update_next = in_feature_update_next[ids]
            regul_term = torch.relu(model.lin_phi(in_feature_update) - model.lin_phi(in_feature_update_next)).norm(2) * self._coeffs['update_diff']
            total_regul = total_regul + regul_term
            self._add('update_diff', regul_term)

        if in_features is None:
            return total_regul

        if self._coeffs['update_msg_diff'] > 0:
            pred_msg = model.lin_phi(in_features.clone().detach())
            in_features_msg_next = in_features.clone().detach()
            in_features_msg_next[:, embedding_dim + 1] = in_features_msg_next[:, embedding_dim + 1] * 1.05
            pred_msg_next = model.lin_phi(in_features_msg_next)
            regul_term = torch.relu(pred_msg[ids_batch] - pred_msg_next[ids_batch]).norm(2) * self._coeffs['update_msg_diff']
            total_regul = total_regul + regul_term
            self._add('update_msg_diff', regul_term)

        if self._coeffs['update_u_diff'] > 0:
            pred_u = model.lin_phi(in_features.clone().detach())
            in_features_u_next = in_features.clone().detach()
            in_features_u_next[:, 0] = in_features_u_next[:, 0] * 1.05
            pred_u_next = model.lin_phi(in_features_u_next)
            regul_term = torch.relu(pred_u_next[ids_batch] - pred_u[ids_batch]).norm(2) * self._coeffs['update_u_diff']
            total_regul = total_regul + regul_term
            self._add('update_u_diff', regul_term)

        if self._coeffs['update_msg_sign'] > 0:
            in_features_modified = in_features.clone().detach()
            in_features_modified[:, 0] = 0
            pred_msg = model.lin_phi(in_features_modified)
            msg_col = in_features[:, embedding_dim + 1].clone().detach()
            regul_term = (torch.tanh(pred_msg / 0.1) - torch.tanh(msg_col.unsqueeze(-1) / 0.1)).norm(2) * self._coeffs['update_msg_sign']
            total_regul = total_regul + regul_term
            self._add('update_msg_sign', regul_term)

        return total_regul

    def finalize_iteration(self):
        """
        finalize the current iteration by recording to history if appropriate.

        this should be called after all regularization computations (compute + compute_update_regul).
        """
        if self.should_record():
            self._record_to_history()

    def get_iteration_total(self) -> float:
        """get total regularization for current iteration."""
        return self._iter_total

    def get_history(self) -> dict:
        """get history dictionary for plotting."""
        return self._history


# --- helper functions used by LossRegularizer ---

def get_in_features_update(rr=None, model=None, embedding=None, device=None):

    n_metabolites = model.n_metabolites
    model_update_type = model.update_type

    if embedding is None:
        embedding = model.a[0:n_metabolites]
        if model.embedding_trial:
            embedding = torch.cat((embedding, model.b[0].repeat(n_metabolites, 1)), dim=1)

    if rr is None:
        if 'generic' in model_update_type:
            if 'excitation' in model_update_type:
                in_features = torch.cat((
                    torch.zeros((n_metabolites, 1), device=device),
                    embedding,
                    torch.zeros((n_metabolites, 1), device=device),
                    torch.ones((n_metabolites, 1), device=device),
                    torch.zeros((n_metabolites, model.excitation_dim), device=device)
                ), dim=1)
            else:
                in_features = torch.cat((
                    torch.zeros((n_metabolites, 1), device=device),
                    embedding,
                    torch.ones((n_metabolites, 1), device=device),
                    torch.ones((n_metabolites, 1), device=device)
                ), dim=1)
        else:
            in_features = torch.cat((torch.zeros((n_metabolites, 1), device=device), embedding), dim=1)
    else:
        if 'generic' in model_update_type:
            if 'excitation' in model_update_type:
                in_features = torch.cat((
                    rr,
                    embedding,
                    torch.zeros((rr.shape[0], 1), device=device),
                    torch.ones((rr.shape[0], 1), device=device),
                    torch.zeros((rr.shape[0], model.excitation_dim), device=device)
                ), dim=1)
            else:
                in_features = torch.cat((
                    rr,
                    embedding,
                    torch.ones((rr.shape[0], 1), device=device),
                    torch.ones((rr.shape[0], 1), device=device)
                ), dim=1)
        else:
            in_features = torch.cat((rr, embedding), dim=1)

    return in_features

def get_in_features_lin_edge(x, model, model_config, xnorm, n_metabolites, device):

    model_name = model_config.model_name

    if model_name in ['PDE_N4', 'PDE_N7', 'PDE_N11']:
        # in_features for lin_edge: [u_j, embedding_j] where u is x[:,3:4]
        in_features_prev = torch.cat((x[:n_metabolites, 3:4] - xnorm / 150, model.a[:n_metabolites]), dim=1)
        in_features = torch.cat((x[:n_metabolites, 3:4], model.a[:n_metabolites]), dim=1)
        in_features_next = torch.cat((x[:n_metabolites, 3:4] + xnorm / 150, model.a[:n_metabolites]), dim=1)
        if model.embedding_trial:
            in_features_prev = torch.cat((in_features_prev, model.b[0].repeat(n_metabolites, 1)), dim=1)
            in_features = torch.cat((in_features, model.b[0].repeat(n_metabolites, 1)), dim=1)
            in_features_next = torch.cat((in_features_next, model.b[0].repeat(n_metabolites, 1)), dim=1)
    elif model_name == 'PDE_N5':
        # in_features for lin_edge: [u_j, embedding_i, embedding_j] where u is x[:,3:4]
        if model.embedding_trial:
            in_features = torch.cat((x[:n_metabolites, 3:4], model.a[:n_metabolites], model.b[0].repeat(n_metabolites, 1), model.a[:n_metabolites], model.b[0].repeat(n_metabolites, 1)), dim=1)
            in_features_next = torch.cat((x[:n_metabolites, 3:4] + xnorm / 150, model.a[:n_metabolites], model.b[0].repeat(n_metabolites, 1), model.a[:n_metabolites], model.b[0].repeat(n_metabolites, 1)), dim=1)
        else:
            in_features = torch.cat((x[:n_metabolites, 3:4], model.a[:n_metabolites], model.a[:n_metabolites]), dim=1)
            in_features_next = torch.cat((x[:n_metabolites, 3:4] + xnorm / 150, model.a[:n_metabolites], model.a[:n_metabolites]), dim=1)
    elif ('PDE_N9_A' in model_name) | (model_name == 'PDE_N9_C') | (model_name == 'PDE_N9_D'):
        in_features = torch.cat((x[:, 3:4], model.a), dim=1)
        in_features_next = torch.cat((x[:,3:4] * 1.05, model.a), dim=1)
    elif model_name == 'PDE_N9_B':
        perm_indices = torch.randperm(n_metabolites, device=model.a.device)
        in_features = torch.cat((x[:, 3:4], x[:, 3:4], model.a, model.a[perm_indices]), dim=1)
        in_features_next = torch.cat((x[:, 3:4], x[:, 3:4] * 1.05, model.a, model.a[perm_indices]), dim=1)
    elif model_name == 'PDE_N8':
        # in_features for lin_edge: [u_i, u_j, embedding_i, embedding_j] where u is x[:,3:4]
        if model.embedding_trial:
            perm_indices = torch.randperm(n_metabolites, device=model.a.device)
            in_features = torch.cat((x[:n_metabolites, 3:4], x[:n_metabolites, 3:4], model.a[:n_metabolites], model.b[0].repeat(n_metabolites, 1), model.a[perm_indices[:n_metabolites]], model.b[0].repeat(n_metabolites, 1)), dim=1)
            in_features_next = torch.cat((x[:n_metabolites, 3:4], x[:n_metabolites, 3:4]*1.05, model.a[:n_metabolites], model.b[0].repeat(n_metabolites, 1), model.a[perm_indices[:n_metabolites]], model.b[0].repeat(n_metabolites, 1)), dim=1)
        else:
            perm_indices = torch.randperm(n_metabolites, device=model.a.device)
            in_features = torch.cat((x[:n_metabolites, 3:4], x[:n_metabolites, 3:4], model.a[:n_metabolites], model.a[perm_indices[:n_metabolites]]), dim=1)
            in_features_next = torch.cat((x[:n_metabolites, 3:4], x[:n_metabolites, 3:4] * 1.05, model.a[:n_metabolites], model.a[perm_indices[:n_metabolites]]), dim=1)
    else:
        # default: just u (signal) where u is x[:,3:4]
        in_features = x[:n_metabolites, 3:4]
        in_features_next = x[:n_metabolites, 3:4] + xnorm / 150

    return in_features, in_features_next


def compute_kinograph_metrics(gt, pred):
    """Compare two kinograph matrices [n_metabolites, n_frames].
    Returns dict: r2, ssim, mean_wasserstein.

    r2: mean per-frame R² (consistent with data_test rollout R²).
    Wasserstein: time-unaligned comparison of population activity modes.
    Projects population snapshots (columns) onto top PCs via SVD, then
    compares the marginal distributions along each PC axis using 1D
    Wasserstein distance normalized by GT std (dimensionless) — averaged
    across PCs. 0 = identical mode distributions, 1 = shift of one GT
    standard deviation. Captures whether GT and GNN visit the same
    collective modes regardless of timing.
    """
    from skimage.metrics import structural_similarity
    from scipy.stats import wasserstein_distance

    # Per-frame R²
    n_frames = gt.shape[1]
    r2_list = []
    for t in range(n_frames):
        gt_col = gt[:, t]
        pred_col = pred[:, t]
        ss_tot = np.sum((gt_col - np.mean(gt_col)) ** 2)
        if ss_tot > 0:
            ss_res = np.sum((gt_col - pred_col) ** 2)
            r2_list.append(1 - ss_res / ss_tot)
        else:
            r2_list.append(0.0)
    r2_mean = np.mean(r2_list)

    data_range = max(np.abs(gt).max(), np.abs(pred).max()) * 2
    if data_range == 0:
        data_range = 1.0
    ssim_val = structural_similarity(gt, pred, data_range=data_range)

    # Time-unaligned mode Wasserstein via PCA projection
    gt_T = gt.T  # [n_frames, n_metabolites]
    pred_T = pred.T

    gt_centered = gt_T - gt_T.mean(axis=0)
    U, S, Vt = np.linalg.svd(gt_centered, full_matrices=False)

    cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
    n_pcs = max(1, int(np.searchsorted(cumvar, 0.99) + 1))
    n_pcs = min(n_pcs, 20)

    basis = Vt[:n_pcs]  # [n_pcs, n_metabolites]
    gt_proj = (gt_T - gt_T.mean(axis=0)) @ basis.T
    pred_proj = (pred_T - gt_T.mean(axis=0)) @ basis.T

    wd_per_pc = []
    for k in range(n_pcs):
        wd = wasserstein_distance(gt_proj[:, k], pred_proj[:, k])
        std_k = gt_proj[:, k].std()
        wd_per_pc.append(wd / std_k if std_k > 0 else 0.0)
    mean_wd = np.mean(wd_per_pc)

    return {'r2': r2_mean, 'ssim': ssim_val, 'mean_wasserstein': mean_wd}


def save_exploration_artifacts(root_dir, exploration_dir, config, config_file_, pre_folder, iteration,
                               iter_in_block=1, block_number=1):
    """
    save exploration artifacts for Claude analysis.

    returns dict with paths to saved directories.
    """
    import glob
    import shutil
    import matplotlib.image as mpimg

    config_save_dir = f"{exploration_dir}/config"
    scatter_save_dir = f"{exploration_dir}/stoichiometry_scatter"
    matrix_save_dir = f"{exploration_dir}/stoichiometry_matrix"
    concentrations_save_dir = f"{exploration_dir}/concentrations"
    mlp_save_dir = f"{exploration_dir}/mlp"
    tree_save_dir = f"{exploration_dir}/exploration_tree"
    protocol_save_dir = f"{exploration_dir}/protocol"
    kinograph_save_dir = f"{exploration_dir}/kinograph"
    embedding_save_dir = f"{exploration_dir}/embedding"
    rate_constants_save_dir = f"{exploration_dir}/rate_constants"

    # create directories at start of experiment (clear only on iteration 1)
    if iteration == 1:
        if os.path.exists(exploration_dir):
            shutil.rmtree(exploration_dir)
    # always ensure directories exist (for resume support)
    os.makedirs(config_save_dir, exist_ok=True)
    os.makedirs(scatter_save_dir, exist_ok=True)
    os.makedirs(matrix_save_dir, exist_ok=True)
    os.makedirs(concentrations_save_dir, exist_ok=True)
    os.makedirs(mlp_save_dir, exist_ok=True)
    os.makedirs(tree_save_dir, exist_ok=True)
    os.makedirs(protocol_save_dir, exist_ok=True)
    os.makedirs(kinograph_save_dir, exist_ok=True)
    os.makedirs(embedding_save_dir, exist_ok=True)
    os.makedirs(rate_constants_save_dir, exist_ok=True)

    is_block_start = (iter_in_block == 1)

    # save config file only at first iteration of each block
    if is_block_start:
        src_config = f"{root_dir}/config/{pre_folder}{config_file_}.yaml"
        dst_config = f"{config_save_dir}/block_{block_number:03d}.yaml"
        if os.path.exists(src_config):
            shutil.copy2(src_config, dst_config)

    # save stoichiometry scatterplot (S learning mode)
    matrix_dir = f"{root_dir}/log/{pre_folder}{config_file_}/tmp_training/matrix"
    scatter_files = glob.glob(f"{matrix_dir}/comparison_*.png")
    if scatter_files:
        latest_scatter = max(scatter_files, key=os.path.getmtime)
        dst_scatter = f"{scatter_save_dir}/iter_{iteration:03d}.png"
        shutil.copy2(latest_scatter, dst_scatter)

    # save rate constants scatterplot (S given / freeze_stoichiometry mode)
    rate_dir = f"{root_dir}/log/{pre_folder}{config_file_}/tmp_training/rate_constants"
    rate_files = glob.glob(f"{rate_dir}/comparison_*.png")
    if rate_files:
        latest_rate = max(rate_files, key=os.path.getmtime)
        dst_rate = f"{rate_constants_save_dir}/iter_{iteration:03d}.png"
        shutil.copy2(latest_rate, dst_rate)

    # save stoichiometry matrix heatmap only at first iteration of each block
    data_folder = f"{root_dir}/graphs_data/{config.dataset}"
    if is_block_start:
        src_matrix = f"{data_folder}/connectivity_matrix.png"
        dst_matrix = f"{matrix_save_dir}/block_{block_number:03d}.png"
        if os.path.exists(src_matrix):
            shutil.copy2(src_matrix, dst_matrix)

    # save concentrations plot only at first iteration of each block
    concentrations_path = f"{data_folder}/concentrations.png"
    if is_block_start:
        dst_conc = f"{concentrations_save_dir}/block_{block_number:03d}.png"
        if os.path.exists(concentrations_path):
            shutil.copy2(concentrations_path, dst_conc)

    # save combined MLP plot (functions_combined or substrate_func + rate_func)
    results_dir = f"{root_dir}/log/{pre_folder}{config_file_}/results"
    src_combined = f"{results_dir}/functions_combined.png"
    if os.path.exists(src_combined):
        shutil.copy2(src_combined, f"{mlp_save_dir}/iter_{iteration:03d}_MLP.png")
    else:
        # fallback: combine substrate_func + rate_func
        src_mlp0 = f"{results_dir}/substrate_func.png"
        src_mlp1 = f"{results_dir}/rate_func.png"
        if os.path.exists(src_mlp0) and os.path.exists(src_mlp1):
            try:
                img0 = mpimg.imread(src_mlp0)
                img1 = mpimg.imread(src_mlp1)
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                axes[0].imshow(img0)
                axes[0].axis('off')
                axes[1].imshow(img1)
                axes[1].axis('off')
                plt.tight_layout()
                plt.savefig(f"{mlp_save_dir}/iter_{iteration:03d}_MLP.png", dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"\033[93mwarning: could not combine MLP plots: {e}\033[0m")

    # save kinograph montage
    src_montage = f"{results_dir}/kinograph_montage.png"
    if os.path.exists(src_montage):
        shutil.copy2(src_montage, f"{kinograph_save_dir}/iter_{iteration:03d}.png")

    # save derivative kinograph montage
    src_deriv = f"{results_dir}/deriv_kinograph_montage.png"
    if os.path.exists(src_deriv):
        shutil.copy2(src_deriv, f"{kinograph_save_dir}/iter_{iteration:03d}_deriv.png")

    # save embedding plot (saved at log/{dataset}/embedding/, not tmp_training/embedding/)
    embedding_dir = f"{root_dir}/log/{pre_folder}{config_file_}/embedding"
    if os.path.isdir(embedding_dir):
        embed_files = glob.glob(f"{embedding_dir}/*.png")
        if embed_files:
            latest_embed = max(embed_files, key=os.path.getmtime)
            shutil.copy2(latest_embed, f"{embedding_save_dir}/iter_{iteration:03d}.png")
            if is_block_start:
                shutil.copy2(latest_embed, f"{embedding_save_dir}/block_{block_number:03d}.png")

    return {
        'config_save_dir': config_save_dir,
        'scatter_save_dir': scatter_save_dir,
        'matrix_save_dir': matrix_save_dir,
        'concentrations_save_dir': concentrations_save_dir,
        'mlp_save_dir': mlp_save_dir,
        'tree_save_dir': tree_save_dir,
        'protocol_save_dir': protocol_save_dir,
        'kinograph_save_dir': kinograph_save_dir,
        'embedding_save_dir': embedding_save_dir,
        'rate_constants_save_dir': rate_constants_save_dir,
        'concentrations_path': concentrations_path
    }
