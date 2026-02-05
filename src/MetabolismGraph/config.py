from typing import Optional, Literal, Annotated
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
    n_reactions: int = 64
    max_metabolites_per_reaction: int = 5
    n_input_metabolites: int = 0

    noise_model_level: float = 0.0

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

    field_type: str = ""

    input_size: int = 2
    output_size: int = 16
    hidden_dim: int = 32
    n_layers: int = 3

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

    sparsity: Literal["none"] = "none"

    learning_rate_start: float = 0.001
    learning_rate_embedding_start: float = 0.001
    learning_rate_NNR_f: float = 0.0001

    # stoichiometry learning rate and regularization
    learning_rate_S_start: float = 0.0
    coeff_S_L1: float = 0.0
    coeff_S_integer: float = 0.0
    coeff_mass_conservation: float = 0.0

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
