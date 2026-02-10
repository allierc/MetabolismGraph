import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse
import os

from MetabolismGraph.config import MetabolismGraphConfig
from MetabolismGraph.generators.data_generator import data_generate
from MetabolismGraph.models.graph_trainer import data_train, data_test
from MetabolismGraph.utils import set_device

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="MetabolismGraph")
    parser.add_argument(
        "-o", "--option", nargs="+", help="option that takes multiple values"
    )

    print()
    device = []
    args = parser.parse_args()

    if args.option:
        print(f"Options: {args.option}")
    if args.option is not None:
        task = args.option[0]
        # Support multiple configs: -o task config1,config2,config3 [best_model]
        # or: -o task config1 config2 config3 [best_model]
        if len(args.option) > 1:
            # Check if configs are comma-separated
            if ',' in args.option[1]:
                config_list = [c.strip() for c in args.option[1].split(',')]
                best_model = args.option[2] if len(args.option) > 2 else None
            else:
                # Multiple space-separated configs, last one might be best_model
                potential_configs = args.option[1:]
                # If last arg looks like a model name (not a config), treat it as best_model
                # Recognized patterns: 'best', 'last', 'phase2', or numeric like '1_342000'
                def _is_model_name(s):
                    return s in ['best', 'last', 'phase2'] or s.replace('_', '').isdigit()
                if len(potential_configs) > 1 and _is_model_name(potential_configs[-1]):
                    config_list = potential_configs[:-1]
                    best_model = potential_configs[-1]
                else:
                    config_list = potential_configs
                    best_model = None
        else:
            config_list = ['metabolism_1']
            best_model = None
    else:
        best_model = '1_342000'
        task = 'train'
        # Default: generate simulation configs
        config_list = [
            'iter_096'
        ]

    print(f"Task: {task}")
    print(f"Config list: {config_list}")
    print()

    for config_file_ in config_list:
        print(" ")
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        config_file = config_file_

        # load config
        config = MetabolismGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
        config.config_file = config_file_

        if device == []:
            device = set_device(config.training.device)

        if "generate" in task:
            data_generate(
                config,
                device=device,
                visualize=False,
                bSave=True,
            )

        if "train" in task:
            data_train(
                config=config,
                erase=False,
                best_model=best_model,
                device=device,
            )

        if "test" in task:
            config.simulation.noise_model_level = 0.0

            data_test(
                config=config,
                visualize=False,
                verbose=False,
                best_model='best',
                run=0,
                device=device,
            )
