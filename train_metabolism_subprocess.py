#!/usr/bin/env python3
"""
standalone metabolism training script for subprocess execution.

this script is called by GNN_LLM.py as a subprocess to ensure that any code
modifications to trainer.py are reloaded for each iteration.

Usage:
    python train_metabolism_subprocess.py --config CONFIG_PATH --device DEVICE [--erase] [--log_file LOG_PATH]
"""

import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports

import argparse
import sys
import os
import traceback

# add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from MetabolismGraph.config import MetabolismGraphConfig
from MetabolismGraph.models.graph_trainer import data_train
from MetabolismGraph.utils import set_device


def main():
    parser = argparse.ArgumentParser(description='train GNN on metabolism data')
    parser.add_argument('--config', type=str, required=True, help='path to config YAML file')
    parser.add_argument('--device', type=str, default='auto', help='device to use')
    parser.add_argument('--erase', action='store_true', help='erase existing log files')
    parser.add_argument('--log_file', type=str, default=None, help='path to analysis log file')
    parser.add_argument('--config_file', type=str, default=None, help='config file name for log directory')
    parser.add_argument('--error_log', type=str, default=None, help='path to error log file')
    parser.add_argument('--best_model', type=str, default=None, help='best model path')

    args = parser.parse_args()

    # open error log file if specified
    error_log = None
    if args.error_log:
        try:
            error_log = open(args.error_log, 'w')
        except Exception as e:
            print(f"warning: could not open error log file: {e}", file=sys.stderr)

    try:
        # load config
        config = MetabolismGraphConfig.from_yaml(args.config)

        # set config_file if provided (needed for proper log directory path)
        if args.config_file:
            config.config_file = args.config_file
            config.dataset = args.config_file

        # set device
        device = set_device(args.device)

        # open log file if specified
        log_file = None
        if args.log_file:
            log_file = open(args.log_file, 'w')

        try:
            # run training - this will reload any modified code
            data_train(
                config=config,
                erase='True',
                best_model='',
                device=device,
                log_file=log_file
            )
        finally:
            if log_file:
                log_file.close()

    except Exception as e:
        error_msg = f"\n{'='*80}\n"
        error_msg += "TRAINING SUBPROCESS ERROR\n"
        error_msg += f"{'='*80}\n\n"
        error_msg += f"Error Type: {type(e).__name__}\n"
        error_msg += f"Error Message: {str(e)}\n\n"
        error_msg += "Full Traceback:\n"
        error_msg += traceback.format_exc()
        error_msg += f"\n{'='*80}\n"

        print(error_msg, file=sys.stderr, flush=True)

        if error_log:
            error_log.write(error_msg)
            error_log.flush()

        sys.exit(1)

    finally:
        if error_log:
            error_log.close()


if __name__ == '__main__':
    main()
