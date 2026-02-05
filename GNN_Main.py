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
        config_list = [args.option[1]]
        if len(args.option) > 2:
            best_model = args.option[2]
        else:
            best_model = None
    else:
        best_model = ''
        task = 'generate'
        config_list = ['metabolism_1']

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
                erase=True,
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
