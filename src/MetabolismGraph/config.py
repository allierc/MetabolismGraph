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
    external_input_type: Literal["none", "visual", "modulation"] = "none"
    external_input_mode: Literal[
        "additive", "multiplicative", "multiplicative_substrate",
        "multiplicative_product", "none"
    ] = "none"

    node_value_map: Optional[str] = None


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

    input_size: int = 2
    output_size: int = 16
    hidden_dim: int = 32
    n_layers: int = 3

    # MLP_sub (substrate_func): (c_k, |s_kj|) -> substrate contribution
    hidden_dim_sub: int = 64
    n_layers_sub: int = 3

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

    # k center regularization: penalize mean(log_k) deviating from GT range center
    # breaks scale ambiguity between k and MLP_sub
    coeff_k_center: float = 0.0

    # phase-1 regularization
    first_coeff_L1: float = 0.0

    measurement_noise_level: float = 0

    # external input learning
    learn_external_input: bool = False

    data_augmentation_loop: int = 40


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
