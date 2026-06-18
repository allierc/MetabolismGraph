from typing import Optional, Literal, Annotated, List
import yaml
from pydantic import BaseModel, ConfigDict, Field


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dimension: int = 2
    n_frames: int = 1000
    start_frame: int = 0
    seed: int = 42

    delta_t: float = 1

    # metabolism parameters
    n_metabolites: int = 100
    n_metabolite_types: int = 1  # number of metabolite types (for per-type homeostasis)
    n_reactions: int = 64
    max_metabolites_per_reaction: int = 5
    n_input_metabolites: int = 0

    # concentration initialization
    concentration_min: float = 2.5
    concentration_max: float = 7.5

    # cyclic structures for oscillatory dynamics
    cycle_fraction: float = 0.0  # fraction of reactions in cycles (0.0 to 1.0)
    cycle_length: int = 4  # number of metabolites per cycle

    # mass-action kinetics: v = k * Π(c^s) (multiplicative, needed for oscillations)
    use_mass_action: bool = False

    # reaction rate constants: k in [10^log_k_min, 10^log_k_max]
    log_k_min: float = -3.0  # default: k_min = 0.001
    log_k_max: float = -1.0  # default: k_max = 0.1

    # flux limiting: prevent negative concentrations (disable for freer oscillations)
    flux_limit: bool = True

    # kinograph visualization
    kinograph_range: float = 1.0  # range around baseline for vmin/vmax

    noise_model_level: float = 0.0
    # stoichiometrically-SOUND intrinsic noise: multiplicative fluctuation on the
    # reaction RATES k_j(t)=k_j*(1+sigma*xi) (enzyme-activity noise). Unlike
    # noise_model_level (which perturbs concentrations and violates mass conservation),
    # this only varies the flux, so every trajectory stays physical. The true mirror
    # of the flyvis circuit-parameter noise lever. Applied in the GENERATOR only.
    noise_rate_level: float = 0.0

    # homeostatic dynamics: prevents equilibration by pulling concentrations toward baseline
    # dc/dt += -homeostatic_strength * (c - c_baseline)
    homeostatic_strength: float = 0.0  # λ (0 = disabled)
    baseline_mode: Literal["initial", "fixed"] = "initial"  # "initial" uses c(t=0), "fixed" uses baseline_concentration
    baseline_concentration: float = 1.0  # c_baseline when baseline_mode="fixed"
    # per-type parameters (optional, overrides global values)
    homeostatic_lambda_per_type: Optional[List[float]] = None  # [λ_0, λ_1, ...] per type
    homeostatic_baseline_per_type: Optional[List[float]] = None  # [c_0, c_1, ...] per type

    # circadian modulation: c_baseline(t) = c_baseline * (1 + A * sin(2π*t/T))
    circadian_amplitude: float = 0.0  # A (0 = no oscillation)
    circadian_period: float = 1440.0  # T in frames (1440 = 24h if delta_t=1min)

    # external input configuration
    external_input_type: Literal["none", "visual", "modulation", "ou"] = "none"
    external_input_mode: Literal[
        "additive", "multiplicative", "multiplicative_substrate",
        "multiplicative_product", "none"
    ] = "none"
    # analytic "modulation" drive: amplitude of the per-input sinusoidal drive
    # written into x[:,4] each frame (used when external_input_type == "modulation")
    external_input_amplitude: float = 0.0
    # for external_input_type="ou": AR(1)/Ornstein-Uhlenbeck per-step correlation (smooth,
    # aperiodic, low-frequency drive matching the real metabolome's structure -- not a sinusoid)
    external_input_phi: float = 0.98

    node_value_map: Optional[str] = None

    # SBML model import: generate data from an external SBML file instead of synthetic
    sbml_file: Optional[str] = None  # path to SBML .xml file (relative to repo root)
    # impose synthetic kinetics on a REAL FBA network TOPOLOGY (no kinetics needed):
    # parse S from this SBML, keep genuine enzymatic reactions, then the generator model
    # imposes random Vmax/Km (=ground truth) -> testable k-recovery on real topology.
    topology_sbml: Optional[str] = None       # path to an FBA SBML (e.g. e_coli_core.xml)
    topology_subgraph_reactions: int = 0  # >0: extract a connected central-carbon subgraph of this size (genome-scale models)
    topology_from_dataset: Optional[str] = None  # reuse an existing dataset's stoichiometry + impose synthetic kinetics
    sbml_t_end: float = 100.0  # simulation end time (SBML time units)

    # Michaelis-Menten kinetics: v = Vmax * Π [c/(Km+c)]^|s|
    log_km_min: float = -1.0  # Km_min = 0.1 mM
    log_km_max: float = 1.0   # Km_max = 10 mM

    # External sources (stimuli): fixed-concentration external sources/sinks
    stimulus_concentrations: Optional[List[float]] = None  # [c_boundary_0, c_boundary_1, ...]
    stimulus_names: Optional[List[str]] = None  # species names for reference


class ClaudeConfig(BaseModel):
    """configuration for Claude-driven exploration experiments."""
    model_config = ConfigDict(extra="forbid")

    n_epochs: int = 1
    data_augmentation_loop: int = 100
    n_iter_block: int = 24
    ucb_c: float = 1.414
    n_parallel: int = 4
    node_name: str = "a100"


class GraphModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model_name: str = ""
    prediction: Literal["first_derivative"] = "first_derivative"

    aggr_type: str = "add"
    embedding_dim: int = 2  # dimension of metabolite embeddings a_i

    field_type: str = ""

    # MLP_sub (substrate_func): (c_k, |s_kj|) -> substrate contribution
    output_size_sub: int = 1
    hidden_dim_sub: int = 64
    n_layers_sub: int = 3
    # substrate-function parameterisation: 'mlp' (default, MLP on (c,|s|)),
    # 'logspace' (g = exp(MLP(log c, |s|)) -> power law c^s is linear in log-c,
    # so curvature/dynamic-range are easy to learn), or 'powerlaw'
    # (g = c^{a(|s|)} with a tiny learnable exponent map -- exact for mass-action).
    substrate_func_type: str = "mlp"

    # MLP_node (rate_func / node_func): homeostasis function
    hidden_dim_node: int = 64
    n_layers_node: int = 3

    input_size_update: int = 3
    n_layers_update: int = 3
    hidden_dim_update: int = 64

    update_type: str = "none"

    # INR type for external input learning
    inr_type: Literal["siren_t", "siren_id", "siren_x", "lowrank"] = "siren_t"

    # SIREN parameters
    input_size_nnr_f: int = 3
    n_layers_nnr_f: int = 5
    hidden_dim_nnr_f: int = 128
    output_size_nnr_f: int = 1
    outermost_linear_nnr_f: bool = True
    omega_f: float = 80.0

    nnr_f_xy_period: float = 1.0
    nnr_f_T_period: float = 1.0

    # lowrank parameters
    lowrank_rank: int = 64
    lowrank_svd_init: bool = True


class PlottingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    colormap: str = "tab20"


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    device: Annotated[str, Field(pattern=r"^(auto|cpu|cuda:\d+)$")] = "auto"

    n_epochs: int = 20
    n_epochs_init: int = 99999
    batch_size: int = 1
    small_init_batch_size: bool = True

    n_runs: int = 2
    seed: int = 42
    time_step: int = 1

    # recurrent training: multi-step rollout during training
    recurrent_training: bool = False
    noise_recurrent_level: float = 0.0

    # variance-weighted sampling: prefer timepoints with high target variance
    variance_weighted_sampling: bool = False

    sparsity: Literal["none"] = "none"

    learning_rate_start: float = 0.001
    learning_rate_embedding_start: float = 0.001
    training_single_type: bool = False  # if True, fix embeddings to single type (no a_i learning)
    learning_rate_NNR_f: float = 0.0001

    # per-component learning rates (0 = use learning_rate_start)
    learning_rate_k: float = 0.0
    learning_rate_node: float = 0.0
    learning_rate_sub: float = 0.0
    learning_rate_embedding: float = 0.0

    # stoichiometry learning rate and regularization
    learning_rate_S_start: float = 0.0
    freeze_stoichiometry: bool = False  # if True, S is fixed (not learned)
    coeff_S_L1: float = 0.0
    coeff_S_integer: float = 0.0
    coeff_mass_conservation: float = 0.0

    # MLP_sub monotonicity regularization (c^s should be increasing)
    coeff_MLP_sub_diff: float = 100.0  # penalize decreasing MLP_sub output

    # MLP_node L1 regularization: penalize large homeostasis output
    # keeps MLP_node values small relative to the reaction terms
    coeff_MLP_node_L1: float = 0.0

    # MLP_sub normalization: penalize substrate_func(c=1, |s|=1) deviating from 1
    # breaks scale ambiguity between k and MLP_sub at the source
    coeff_MLP_sub_norm: float = 0.0

    # k floor: penalize log_k values below threshold (prevents outlier reactions)
    coeff_k_floor: float = 0.0
    k_floor_threshold: float = -3.0

    # phase-1 regularization
    first_coeff_L1: float = 0.0

    measurement_noise_level: float = 0

    # external input learning
    learn_external_input: bool = False

    cluster_distance_threshold: float = 0.1  # DBSCAN eps for embedding clustering

    data_augmentation_loop: int = 40

    # Phase 2: homeostasis training (recurrent, reaction frozen)
    homeostasis_training: bool = False
    skip_phase1: bool = False  # if True, skip Phase 1 and go straight to Phase 2
    homeostasis_time_step: int = 32  # recurrent rollout steps for Phase 2
    learning_rate_node_homeostasis: float = 0.0  # 0 = use learning_rate_node
    learning_rate_embedding_homeostasis: float = 0.0  # 0 = use learning_rate_node_homeostasis

    # ---- autoregressive rollout curriculum (opt-in; active only when
    # n_steps_schedule is non-empty). Ported from connectome-cx: ramp the
    # supervised rollout horizon per epoch, co-ramp LR down, soft tail-loss
    # weighting beyond the horizon, gradient clipping. Leaves the single-step
    # k-recovery path untouched when n_steps_schedule is empty. ----
    n_steps_schedule: list[int] = []        # per-epoch rollout horizon, e.g. [10,50,100,200,...]
    lr_schedule: list[float] = []           # per-epoch lr for 'k'+'MLP_sub' groups; [] = keep base

    # ---- hybrid structured-MM curriculum (substrate_func_type='mm'): freeze the kinetic
    # SHAPE (per-reaction Km, the MLP_sub group) at lr=0 while the SCALE (log_k / Vmax)
    # converges, then slowly ramp the Km lr to watch the joint shape x scale solution
    # evolve. Independent of lr_schedule (which co-ramps k+MLP_sub for the AR curriculum).
    lr_sub_schedule: list[float] = []       # per-epoch lr for the 'MLP_sub' group ONLY; [] = keep base
    pretrain_substrate_steps: int = 0       # >0: warm-start MLP_sub (Km) + log_k on one-step loss before the main scheme
    pretrain_substrate_lr: float = 1e-3     # lr for the pretraining warm-start
    init_km_from_gt: bool = False           # structured-MM oracle: init log_km from GT (upper-bound baseline)
    coeff_tail_loss: float = 0.0            # weight for frames in [T_epoch, T_eff); 0 = hard cutoff
    ar_max_roll: int = 0                    # cap on T_eff (0 = 2*T_epoch when tail>0 else T_epoch)
    grad_clip: float = 0.0                  # max grad norm (0 = no clipping)
    # anchor k: after the single-step warmup (epoch 0), freeze log_k (lr->0) for the
    # rest of the curriculum so the multi-step phase cannot scramble the recovered
    # rate constants. Tests whether stability can be bought WITHOUT losing identifiability.
    anchor_k_after_warmup: bool = False

    # torch.compile the per-step dx/dt core to fuse the many tiny kernels of the
    # autoregressive rollout (the loop is CPU-launch-bound on the small graph).
    # Numerically identical to the eager path (same ops, just fused). cuda only.
    compile_rollout: bool = False

    # hard wall-clock budget (hours). When > 0, training saves a final checkpoint
    # and stops as soon as the elapsed time exceeds it (guarantees a bounded run
    # regardless of the rollout-horizon cost). 0 = no limit.
    max_train_hours: float = 0.0


class MetabolismGraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: Optional[str] = "MetabolismGraph"
    dataset: str
    config_file: str = "none"

    simulation: SimulationConfig
    graph_model: GraphModelConfig
    claude: Optional[ClaudeConfig] = None
    plotting: PlottingConfig = PlottingConfig()
    training: TrainingConfig

    @staticmethod
    def from_yaml(file_name: str):
        with open(file_name, "r") as file:
            raw_config = yaml.safe_load(file)
        return MetabolismGraphConfig(**raw_config)

    def pretty(self):
        return yaml.dump(self, default_flow_style=False, sort_keys=False, indent=4)
