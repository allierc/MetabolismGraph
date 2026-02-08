"""
minimal plotting stub for metabolism exploration.

the full data_plot function is not needed for the LLM loop.
this stub provides a compatible interface that saves basic result figures.
"""

import matplotlib
matplotlib.use('Agg')
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from MetabolismGraph.utils import create_log_dir, sort_key


def data_plot(config, config_file, epoch_list, style, extended, device, apply_weight_correction=False, log_file=None):
    """plot metabolism results (minimal version).

    copies key result figures to the results directory for the LLM loop.
    """

    if 'black' in style:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    log_dir, logger = create_log_dir(config=config, erase=False)
    os.makedirs(os.path.join(log_dir, 'results'), exist_ok=True)

    if epoch_list == ['best']:
        files = glob.glob(f"{log_dir}/models/*")
        if not files:
            print(f"warning: no model files found in {log_dir}/models/")
            return
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]
        epoch_list = [filename]

    # save scatter/matrix plots from tmp_training to results
    matrix_dir = os.path.join(log_dir, 'tmp_training', 'matrix')
    results_dir = os.path.join(log_dir, 'results')

    if os.path.isdir(matrix_dir):
        scatter_files = glob.glob(f"{matrix_dir}/comparison_*.png")
        if scatter_files:
            latest = max(scatter_files, key=os.path.getmtime)
            import shutil
            shutil.copy2(latest, os.path.join(results_dir, 'stoichiometry_scatter.png'))

    # save function plots from tmp_training/function to results
    func_dir = os.path.join(log_dir, 'tmp_training', 'function')
    if os.path.isdir(func_dir):
        for subdir in ['substrate_func', 'rate_func']:
            src_dir = os.path.join(func_dir, subdir)
            if os.path.isdir(src_dir):
                png_files = glob.glob(f"{src_dir}/*.png")
                if png_files:
                    latest = max(png_files, key=os.path.getmtime)
                    import shutil
                    shutil.copy2(latest, os.path.join(results_dir, f'{subdir}.png'))

    # combine function plots if both exist
    src_mlp0 = os.path.join(results_dir, 'substrate_func.png')
    src_mlp1 = os.path.join(results_dir, 'rate_func.png')
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
            plt.savefig(os.path.join(results_dir, 'functions_combined.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"warning: could not combine function plots: {e}")

    if log_file:
        log_file.write(f"plot_complete=True\n")
        log_file.flush()

    print(f"plots saved to {results_dir}")
