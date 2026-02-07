import gc
import glob
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter


def sort_key(filename):
    if filename.split('_')[-2] == 'graphs':
        return 0
    else:
        return 1E7 * int(filename.split('_')[-2]) + int(filename.split('_')[-1][:-3])


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def set_device(device: str = 'auto'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)

    if device == 'auto':
        if torch.cuda.is_available():
            try:
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                    encoding='utf-8'
                )
                free_mem_list = []
                for line in result.strip().split('\n'):
                    index_str, mem_str = line.strip().split(',')
                    index = int(index_str)
                    free_mem = float(mem_str) * 1024 * 1024
                    free_mem_list.append((index, free_mem))
                num_gpus = torch.cuda.device_count()
                if num_gpus != len(free_mem_list):
                    print(f"mismatch in GPU count between PyTorch ({num_gpus}) and nvidia-smi ({len(free_mem_list)})")
                    device = 'cpu'
                    print(f"using device: {device}")
                else:
                    max_free_memory = -1
                    best_device_id = -1
                    for index, free_mem in free_mem_list:
                        if free_mem > max_free_memory:
                            max_free_memory = free_mem
                            best_device_id = index
                    if best_device_id == -1:
                        raise ValueError("could not determine the GPU with the most free memory.")
                    device = f'cuda:{best_device_id}'
                    torch.cuda.set_device(best_device_id)
                    total_memory_gb = torch.cuda.get_device_properties(best_device_id).total_memory / 1024 ** 3
                    free_memory_gb = max_free_memory / 1024 ** 3
                    print(
                        f"using device: {device}, name: {torch.cuda.get_device_name(best_device_id)}, "
                        f"total memory: {total_memory_gb:.2f} GB, free memory: {free_memory_gb:.2f} GB")
            except Exception as e:
                print(f"failed to get GPU information: {e}")
                device = 'cpu'
                print(f"using device: {device}")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print(f"using device: {device}")
        else:
            device = 'cpu'
            print(f"using device: {device}")
    return device


def create_log_dir(config=[], erase=True):
    log_dir = os.path.join('.', 'log', config.config_file)
    print('log_dir: {}'.format(log_dir))

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/external_input'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/matrix'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/function'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/function/substrate_func'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/function/rate_func'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/embedding'), exist_ok=True)

    if erase:
        files = glob.glob(f"{log_dir}/models/*")
        for f in files:
            os.remove(f)
        results_dir = os.path.join(log_dir, 'results')
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir, ignore_errors=True)
        os.makedirs(results_dir, exist_ok=True)
        tmp_training_dir = os.path.join(log_dir, 'tmp_training')
        if os.path.exists(tmp_training_dir):
            shutil.rmtree(tmp_training_dir, ignore_errors=True)
        os.makedirs(os.path.join(log_dir, 'tmp_training/external_input'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'tmp_training/matrix'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'tmp_training/function'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'tmp_training/function/substrate_func'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'tmp_training/function/rate_func'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'tmp_training/embedding'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'tmp_training/rate_constants'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_recons'), exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_dir, 'training.log'),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    return log_dir, logger


def fig_init(fontsize=12, formatx='%.2f', formaty='%.2f'):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks([])
    plt.yticks([])
    ax.tick_params(axis='both', which='major', pad=15)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_formatter(FormatStrFormatter(formatx))
    ax.yaxis.set_major_formatter(FormatStrFormatter(formaty))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    for spine in ax.spines.values():
        spine.set_alpha(0.75)
    return fig, ax


def get_equidistant_points(n_points=1024):
    indices = np.arange(0, n_points, dtype=float) + 0.5
    r = np.sqrt(indices / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y = r * np.cos(theta), r * np.sin(theta)
    return x, y


def check_and_clear_memory(
        device: str = None,
        iteration_number: int = None,
        every_n_iterations: int = 100,
        memory_percentage_threshold: float = 0.6
):
    if device and 'cuda' in device:
        logging.getLogger(__name__)
        if (iteration_number % every_n_iterations == 0):
            torch.cuda.memory_allocated(device)
            gc.collect()
            torch.cuda.empty_cache()
        if torch.cuda.memory_allocated(device) > memory_percentage_threshold * torch.cuda.get_device_properties(device).total_memory:
            print("memory usage is high. calling garbage collector and clearing cache.")
            gc.collect()
            torch.cuda.empty_cache()


def add_pre_folder(config_file_):
    """return (config_file, pre_folder) for MetabolismGraph.

    MetabolismGraph configs are flat (no subdirectories), so this always
    returns ('', config_file_).
    """
    return config_file_, ''


def linear_model(x, a, b):
    return a * x + b


# --- data I/O ---

def detect_format(path: str | Path) -> Literal['npy', 'none']:
    path = Path(path)
    base_path = path.with_suffix('') if path.suffix in ('.npy',) else path
    npy_path = Path(str(base_path) + '.npy')
    if npy_path.exists():
        return 'npy'
    return 'none'


def load_simulation_data(path: str | Path) -> np.ndarray:
    path = Path(path)
    base_path = path.with_suffix('') if path.suffix in ('.npy',) else path
    fmt = detect_format(path)
    if fmt == 'none':
        raise FileNotFoundError(f"no .npy found at {base_path}")
    npy_path = Path(str(base_path) + '.npy')
    return np.load(npy_path)
