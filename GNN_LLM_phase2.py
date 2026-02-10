import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse
import os
import re
import shutil
import subprocess
import time
import yaml

# redirect PyTorch JIT cache to /scratch instead of /tmp (per IT request)
if os.path.isdir('/scratch'):
    os.environ['TMPDIR'] = '/scratch/allierc'
    os.makedirs('/scratch/allierc', exist_ok=True)

import sys

from MetabolismGraph.config import MetabolismGraphConfig
from MetabolismGraph.models.graph_trainer import data_train, data_test
from MetabolismGraph.models.exploration_tree import compute_ucb_scores
from MetabolismGraph.models.plot_exploration_tree import parse_ucb_scores, plot_ucb_tree
from MetabolismGraph.models.utils import save_exploration_artifacts
from MetabolismGraph.utils import set_device, add_pre_folder
from GNN_PlotFigure import data_plot

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

N_PARALLEL = 4
FIXED_TIME_STEPS = {0: 4, 1: 16, 2: 32, 3: 64}
BATCHES_PER_BLOCK = 2
ITERS_PER_BLOCK = BATCHES_PER_BLOCK * N_PARALLEL  # 8


# ---------------------------------------------------------------------------
# resume helpers
# ---------------------------------------------------------------------------

def detect_last_iteration(analysis_path, config_save_dir, n_parallel):
    """detect the last fully completed batch from saved artifacts.

    scans two sources:
      1. analysis.md for ``## Iter N:`` entries (written by Claude after training)
      2. config save dir for ``iter_NNN_slot_SS.yaml`` files (saved after test+plot)

    returns the start_iteration for the next batch (1-indexed), or 1 if nothing found.
    """
    found_iters = set()

    # source 1: analysis.md — most reliable, written by Claude
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            for line in f:
                match = re.match(r'^##+ Iter (\d+):', line)
                if match:
                    found_iters.add(int(match.group(1)))

    # source 2: saved config snapshots
    if os.path.isdir(config_save_dir):
        for fname in os.listdir(config_save_dir):
            match = re.match(r'iter_(\d+)_slot_\d+\.yaml', fname)
            if match:
                found_iters.add(int(match.group(1)))

    if not found_iters:
        return 1

    last_iter = max(found_iters)

    # find the batch that contains last_iter
    batch_start = ((last_iter - 1) // n_parallel) * n_parallel + 1
    batch_iters = set(range(batch_start, batch_start + n_parallel))

    # check if the full batch completed
    if batch_iters.issubset(found_iters):
        # full batch done — resume from next batch
        resume_at = batch_start + n_parallel
    else:
        # partial batch — redo this batch
        resume_at = batch_start

    return resume_at


# ---------------------------------------------------------------------------
# cluster helpers
# ---------------------------------------------------------------------------

CLUSTER_HOME = "/groups/saalfeld/home/allierc"
CLUSTER_ROOT_DIR = f"{CLUSTER_HOME}/Graph/MetabolismGraph"


def submit_cluster_job(slot, config_path, analysis_log_path, config_file_field,
                       log_dir, root_dir, erase=True, node_name='a100',
                       best_model=None):
    """submit a single training job to the cluster WITHOUT -K (non-blocking).

    returns the LSF job ID string, or None if submission failed.
    """
    cluster_script_path = f"{log_dir}/cluster_train_{slot:02d}.sh"
    error_details_path = f"{log_dir}/training_error_{slot:02d}.log"

    # build cluster-side paths
    cluster_config_path = config_path.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_analysis_log = analysis_log_path.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_error_log = error_details_path.replace(root_dir, CLUSTER_ROOT_DIR)

    cluster_train_cmd = f"python train_metabolism_subprocess.py --config '{cluster_config_path}' --device cuda"
    cluster_train_cmd += f" --log_file '{cluster_analysis_log}'"
    cluster_train_cmd += f" --config_file '{config_file_field}'"
    cluster_train_cmd += f" --error_log '{cluster_error_log}'"
    if best_model:
        cluster_train_cmd += f" --best_model '{best_model}'"
    if erase:
        cluster_train_cmd += " --erase"

    with open(cluster_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {CLUSTER_ROOT_DIR}\n")
        f.write(f"conda run -n neural-graph {cluster_train_cmd}\n")
    os.chmod(cluster_script_path, 0o755)

    cluster_script = cluster_script_path.replace(root_dir, CLUSTER_ROOT_DIR)

    # cluster-side log paths for capturing stdout/stderr
    cluster_log_dir = log_dir.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_stdout = f"{cluster_log_dir}/cluster_train_{slot:02d}.out"
    cluster_stderr = f"{cluster_log_dir}/cluster_train_{slot:02d}.err"

    # submit WITHOUT -K so it returns immediately; capture stdout/stderr to files
    ssh_cmd = (
        f"ssh allierc@login1 \"cd {CLUSTER_ROOT_DIR} && "
        f"bsub -n 8 -gpu 'num=1' -q gpu_{node_name} -W 6000 "
        f"-o '{cluster_stdout}' -e '{cluster_stderr}' "
        f"'bash {cluster_script}'\""
    )
    print(f"\033[96m  slot {slot} (ts={FIXED_TIME_STEPS[slot]}): submitting via SSH\033[0m")
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

    match = re.search(r'Job <(\d+)>', result.stdout)
    if match:
        job_id = match.group(1)
        print(f"\033[92m  slot {slot}: job {job_id} submitted\033[0m")
        return job_id
    else:
        print(f"\033[91m  slot {slot}: submission FAILED\033[0m")
        print(f"    stdout: {result.stdout.strip()}")
        print(f"    stderr: {result.stderr.strip()}")
        return None


def wait_for_cluster_jobs(job_ids, log_dir=None, poll_interval=60):
    """poll bjobs via SSH until all jobs finish.

    args:
        job_ids: dict {slot: job_id_string}
        log_dir: local directory where cluster_train_XX.err files are written
        poll_interval: seconds between polls

    returns:
        dict {slot: bool} — True if DONE, False if EXIT/failed
    """
    pending = dict(job_ids)  # {slot: job_id}
    results = {}

    while pending:
        ids_str = ' '.join(pending.values())
        ssh_cmd = f'ssh allierc@login1 "bjobs {ids_str} 2>/dev/null"'
        out = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

        for slot, jid in list(pending.items()):
            for line in out.stdout.splitlines():
                if jid in line:
                    if 'DONE' in line:
                        results[slot] = True
                        del pending[slot]
                        print(f"\033[92m  slot {slot} (job {jid}): DONE\033[0m")
                    elif 'EXIT' in line:
                        results[slot] = False
                        del pending[slot]
                        print(f"\033[91m  slot {slot} (job {jid}): FAILED (EXIT)\033[0m")
                        # try to read error log for diagnosis
                        if log_dir:
                            err_file = f"{log_dir}/cluster_train_{slot:02d}.err"
                            if os.path.exists(err_file):
                                try:
                                    with open(err_file, 'r') as ef:
                                        err_content = ef.read().strip()
                                    if err_content:
                                        print(f"\033[91m  --- slot {slot} error log ---\033[0m")
                                        for eline in err_content.splitlines()[-30:]:
                                            print(f"\033[91m    {eline}\033[0m")
                                        print(f"\033[91m  --- end error log ---\033[0m")
                                except Exception:
                                    pass
                    # else: PEND or RUN — still waiting

            # if job not found in bjobs output, it may have finished and been cleaned up
            if slot in pending and jid not in out.stdout:
                # bjobs doesn't list completed jobs after a while — check if log exists
                results[slot] = True  # assume done if disappeared from queue
                del pending[slot]
                print(f"\033[93m  slot {slot} (job {jid}): no longer in queue (assuming DONE)\033[0m")

        if pending:
            statuses = [f"slot {s}" for s in pending]
            print(f"\033[90m  ... waiting for {', '.join(statuses)} ({poll_interval}s)\033[0m")
            time.sleep(poll_interval)

    return results


def is_git_repo(path):
    """check if path is inside a git repository."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--is-inside-work-tree'],
            cwd=path, capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def get_modified_code_files(root_dir, code_files):
    """return list of code_files that have uncommitted changes (staged or unstaged)."""
    modified = []
    try:
        result = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD'],
            cwd=root_dir, capture_output=True, text=True, timeout=10
        )
        changed = set(result.stdout.strip().splitlines())
        # also check staged changes
        result2 = subprocess.run(
            ['git', 'diff', '--name-only', '--cached'],
            cwd=root_dir, capture_output=True, text=True, timeout=10
        )
        changed.update(result2.stdout.strip().splitlines())
        for f in code_files:
            if f in changed:
                modified.append(f)
    except Exception:
        pass
    return modified


def run_claude_cli(prompt, root_dir, max_turns=500, allow_code_edit=False):
    """run Claude CLI with real-time output streaming. returns output text.

    When allow_code_edit=True (between blocks), Claude can also use Bash for git diff.
    """
    tools = ['Read', 'Edit', 'Write']
    if allow_code_edit:
        tools.append('Bash')

    claude_cmd = [
        'claude',
        '-p', prompt,
        '--output-format', 'text',
        '--max-turns', str(max_turns),
        '--allowedTools',
        *tools
    ]

    output_lines = []
    process = subprocess.Popen(
        claude_cmd,
        cwd=root_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='', flush=True)
        output_lines.append(line)

    process.wait()
    return ''.join(output_lines)


# ---------------------------------------------------------------------------
# Phase 2 specific helpers
# ---------------------------------------------------------------------------

def compute_phase2_score(log_path):
    """compute composite Phase 2 score from analysis.log.

    Returns dict with:
      - phase2_score: composite metric in [0, 1]
      - slope_accuracy: 1 - mean(|learned - gt| / |gt|) clamped to [0, 1]
      - embedding_cluster_acc: from analysis.log
      - rate_constants_R2: from analysis.log (sanity check)
    """
    if not os.path.exists(log_path):
        return {'phase2_score': 0.0, 'slope_accuracy': 0.0,
                'embedding_cluster_acc': 0.0, 'rate_constants_R2': 0.0}

    with open(log_path, 'r') as f:
        content = f.read()

    # parse slopes
    slopes = {}
    gt_slopes = {}
    for t in range(10):  # up to 10 types
        m = re.search(rf'MLP_node_slope_{t}:\s*([-\d.eE+]+)', content)
        g = re.search(rf'MLP_node_gt_slope_{t}:\s*([-\d.eE+]+)', content)
        if m and g:
            slopes[t] = float(m.group(1))
            gt_slopes[t] = float(g.group(1))

    cluster_m = re.search(r'embedding_cluster_acc:\s*([\d.]+)', content)
    cluster_acc = float(cluster_m.group(1)) if cluster_m else 0.0

    r2_m = re.search(r'rate_constants_R2:\s*([\d.]+)', content)
    r2 = float(r2_m.group(1)) if r2_m else 0.0

    # compute slope accuracy
    if slopes and gt_slopes:
        errors = []
        for t in slopes:
            if t in gt_slopes and abs(gt_slopes[t]) > 1e-8:
                errors.append(abs(slopes[t] - gt_slopes[t]) / abs(gt_slopes[t]))
        slope_accuracy = max(0.0, min(1.0, 1.0 - (sum(errors) / len(errors)))) if errors else 0.0
    else:
        slope_accuracy = 0.0

    r2_preserved = 1.0 if r2 > 0.5 else 0.0

    phase2_score = 0.5 * slope_accuracy + 0.4 * cluster_acc + 0.1 * r2_preserved

    return {
        'phase2_score': phase2_score,
        'slope_accuracy': slope_accuracy,
        'embedding_cluster_acc': cluster_acc,
        'rate_constants_R2': r2,
    }


def setup_phase2_data(phase1_dataset_dir, slot_dataset_names, root_dir):
    """create symlinks so Phase 2 slots can find Phase 1 data."""
    graphs_dir = os.path.join(root_dir, 'graphs_data')
    for slot_name in slot_dataset_names:
        link_path = os.path.join(graphs_dir, slot_name)
        if os.path.exists(link_path):
            continue
        if os.path.islink(link_path):
            os.unlink(link_path)
        os.symlink(phase1_dataset_dir, link_path)
        print(f"\033[90m  symlink: {link_path} -> {phase1_dataset_dir}\033[0m")


def setup_cluster_data_symlinks(phase1_dataset, slot_dataset_names):
    """create symlinks on the cluster for Phase 2 data directories."""
    for slot_name in slot_dataset_names:
        cluster_link = f"{CLUSTER_ROOT_DIR}/graphs_data/{slot_name}"
        cluster_target = f"{CLUSTER_ROOT_DIR}/graphs_data/{phase1_dataset}"
        ssh_cmd = (
            f"ssh allierc@login1 "
            f"\"test -e '{cluster_link}' || ln -s '{cluster_target}' '{cluster_link}'\""
        )
        result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"\033[90m  cluster symlink: {slot_name} -> {phase1_dataset}\033[0m")
        else:
            print(f"\033[91m  cluster symlink failed for {slot_name}: {result.stderr.strip()}\033[0m")


def setup_cluster_checkpoint(phase1_model_path, slot_config_file, root_dir):
    """ensure Phase 1 checkpoint is available in slot's model dir on cluster.

    Args:
        phase1_model_path: absolute path to the .pt model file
        slot_config_file: config file name for this Phase 2 slot
        root_dir: project root directory

    Copies the model file to the slot's model dir on the cluster.
    Returns the best_model label string (e.g. '1_342000') or None if failed.
    """
    if not os.path.isfile(phase1_model_path):
        print(f"\033[91m  Phase 1 model not found: {phase1_model_path}\033[0m")
        return None

    filename = os.path.basename(phase1_model_path)

    # extract label (e.g. "best_model_with_0_graphs_1_342000.pt" -> "1_342000")
    parts = filename.split('graphs_')
    if len(parts) < 2:
        print(f"\033[91m  cannot parse model filename: {filename}\033[0m")
        return None
    best_model_label = parts[1].replace('.pt', '')

    # copy to cluster slot model dir
    cluster_slot_models = f"{CLUSTER_ROOT_DIR}/log/{slot_config_file}/models"
    cluster_model_path = f"{cluster_slot_models}/{filename}"

    ssh_cmd = (
        f"ssh allierc@login1 \"mkdir -p '{cluster_slot_models}'\""
    )
    subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

    rsync_cmd = f"rsync -az '{phase1_model_path}' allierc@login1:'{cluster_model_path}'"
    result = subprocess.run(rsync_cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"\033[90m  checkpoint synced: {filename} -> {cluster_slot_models}/\033[0m")
    else:
        print(f"\033[91m  checkpoint sync failed: {result.stderr.strip()}\033[0m")
        return None

    return best_model_label


def setup_local_checkpoint(phase1_model_path, slot_config_file, root_dir):
    """copy Phase 1 checkpoint into slot's local model dir.

    Args:
        phase1_model_path: absolute path to the .pt model file
        slot_config_file: config file name for this Phase 2 slot
        root_dir: project root directory

    Creates log/{slot_config_file}/models/ and copies the model file there.
    Returns the best_model label string (e.g. '1_342000') or None if failed.
    """
    if not os.path.isfile(phase1_model_path):
        print(f"\033[91m  Phase 1 model not found: {phase1_model_path}\033[0m")
        return None

    filename = os.path.basename(phase1_model_path)

    # extract label (e.g. "best_model_with_0_graphs_1_342000.pt" -> "1_342000")
    parts = filename.split('graphs_')
    if len(parts) < 2:
        print(f"\033[91m  cannot parse model filename: {filename}\033[0m")
        return None
    best_model_label = parts[1].replace('.pt', '')

    # create local slot model dir and copy checkpoint
    local_slot_models = os.path.join(root_dir, 'log', slot_config_file, 'models')
    os.makedirs(local_slot_models, exist_ok=True)
    dst_path = os.path.join(local_slot_models, filename)
    if not os.path.exists(dst_path):
        shutil.copy2(phase1_model_path, dst_path)
        print(f"\033[90m  local checkpoint: {filename} -> {local_slot_models}/\033[0m")
    else:
        print(f"\033[90m  local checkpoint already exists: {dst_path}\033[0m")

    return best_model_label


def sync_code_to_cluster(root_dir):
    """sync modified graph_trainer.py to cluster via rsync."""
    src = os.path.join(root_dir, 'src/MetabolismGraph/models/graph_trainer.py')
    dst = f"allierc@login1:{CLUSTER_ROOT_DIR}/src/MetabolismGraph/models/graph_trainer.py"

    rsync_cmd = f"rsync -az '{src}' '{dst}'"
    result = subprocess.run(rsync_cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"\033[92m  synced graph_trainer.py to cluster\033[0m")
    else:
        print(f"\033[91m  sync failed: {result.stderr.strip()}\033[0m")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="MetabolismGraph — Phase 2 Homeostasis Exploration")
    parser.add_argument(
        "-o", "--option", nargs="+", help="option that takes multiple values"
    )
    parser.add_argument(
        "--fresh", action="store_true", default=True, help="start from iteration 1 (ignore auto-resume)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="auto-resume from last completed batch"
    )
    parser.add_argument(
        "--phase1-checkpoint", type=str, required=True,
        help="path to Phase 1 best model .pt file (e.g., log/iter_096/models/best_model_with_0_graphs_1_342000.pt)"
    )
    parser.add_argument(
        "--phase1-dataset", type=str, default=None,
        help="Phase 1 dataset name in graphs_data/ (default: auto-detect from checkpoint path)"
    )

    print()
    device = []
    args = parser.parse_args()

    if args.option:
        print(f"options: {args.option}")
    if args.option is not None:
        task = args.option[0]
        config_list = [args.option[1]]
        task_params = {}
        for arg in args.option[2:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                task_params[key] = int(value) if value.isdigit() else value
    else:
        task = 'train_test_plot_Claude_cluster'
        config_list = ['phase2_homeostasis']
        task_params = {'iterations': 64}

    n_iterations = task_params.get('iterations', 64)
    base_config_name = config_list[0] if config_list else 'phase2_homeostasis'
    instruction_name = task_params.get('instruction', 'instruction_phase2_homeostasis')
    llm_task_name = task_params.get('llm_task', f'{base_config_name}_Claude')

    # -----------------------------------------------------------------------
    # setup
    # -----------------------------------------------------------------------
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_root = root_dir + "/config"

    # find the source config — use Phase 1 config as base, or iter_096 default
    source_config_name = task_params.get('source_config', 'iter_096')
    source_config = f"{config_root}/{source_config_name}.yaml"
    if not os.path.exists(source_config):
        print(f"\033[91merror: source config not found: {source_config}\033[0m")
        sys.exit(1)

    # read source config to extract claude params
    with open(source_config, 'r') as f:
        source_data = yaml.safe_load(f)
    claude_cfg = source_data.get('claude', {})
    claude_n_epochs = claude_cfg.get('n_epochs', 1)
    claude_data_augmentation_loop = claude_cfg.get('data_augmentation_loop', 1000)
    claude_n_iter_block = claude_cfg.get('n_iter_block', ITERS_PER_BLOCK)
    claude_ucb_c = claude_cfg.get('ucb_c', 1.414)
    claude_node_name = claude_cfg.get('node_name', 'a100')
    n_iter_block = ITERS_PER_BLOCK  # always 8 for Phase 2 (2 batches x 4 slots)

    # resolve Phase 1 checkpoint to absolute path
    phase1_checkpoint = args.phase1_checkpoint
    if not os.path.isabs(phase1_checkpoint):
        phase1_checkpoint = os.path.join(root_dir, phase1_checkpoint)
    if not os.path.isfile(phase1_checkpoint):
        print(f"\033[91merror: Phase 1 checkpoint not found: {phase1_checkpoint}\033[0m")
        print(f"\033[93m  provide the path to a .pt model file, e.g.:\033[0m")
        print(f"\033[93m    log/iter_096/models/best_model_with_0_graphs_1_342000.pt\033[0m")
        sys.exit(1)

    print(f"\033[94mcluster node: gpu_{claude_node_name}\033[0m")
    print(f"\033[94mPhase 1 checkpoint: {phase1_checkpoint}\033[0m")
    print(f"\033[94mfixed time_steps: {FIXED_TIME_STEPS}\033[0m")

    # Phase 1 dataset detection — extract from log dir name
    # e.g. log/iter_096/models/best_model.pt -> dataset = "iter_096"
    # e.g. log/simulation_oscillatory_rank_50_Claude_03/models/best_model.pt -> dataset = "simulation_oscillatory_rank_50_Claude_03"
    phase1_dataset = args.phase1_dataset
    if not phase1_dataset:
        # walk up from models/ dir to get the log dir name
        models_dir = os.path.dirname(phase1_checkpoint)
        log_entry_dir = os.path.dirname(models_dir)
        phase1_dataset = os.path.basename(log_entry_dir)
        print(f"\033[90m  auto-detected Phase 1 dataset: {phase1_dataset}\033[0m")

    # -----------------------------------------------------------------------
    # resume detection
    # -----------------------------------------------------------------------
    if args.resume:
        analysis_path_probe = f"{root_dir}/{llm_task_name}_analysis.md"
        config_save_dir_probe = f"{root_dir}/log/Claude_exploration/{instruction_name}_parallel/config"
        start_iteration = detect_last_iteration(analysis_path_probe, config_save_dir_probe, N_PARALLEL)
        if start_iteration > 1:
            print(f"\033[93mauto-resume: resuming from batch starting at {start_iteration}\033[0m")
        else:
            print(f"\033[93mfresh start (no previous iterations found)\033[0m")
    else:
        start_iteration = 1
        _analysis_check = f"{root_dir}/{llm_task_name}_analysis.md"
        if os.path.exists(_analysis_check):
            print(f"\033[91mWARNING: fresh start will erase existing results in:\033[0m")
            print(f"\033[91m  {_analysis_check}\033[0m")
            print(f"\033[91m  {root_dir}/{llm_task_name}_memory.md\033[0m")
            answer = input("\033[91mcontinue? (y/n): \033[0m").strip().lower()
            if answer != 'y':
                print("aborted.")
                sys.exit(0)
        print(f"\033[93mfresh start\033[0m")

    # -----------------------------------------------------------------------
    # initialize 4 slot configs
    # -----------------------------------------------------------------------
    config_paths = {}
    analysis_log_paths = {}
    slot_names = {}
    pre_folder = ''  # Phase 2 configs don't use pre_folder

    for slot in range(N_PARALLEL):
        slot_name = f"{llm_task_name}_{slot:02d}"
        slot_names[slot] = slot_name
        target = f"{config_root}/{slot_name}.yaml"
        config_paths[slot] = target
        analysis_log_paths[slot] = f"{root_dir}/{slot_name}_analysis.log"

        if start_iteration == 1 and not args.resume:
            # fresh start: copy source config, set Phase 2 params per slot
            shutil.copy2(source_config, target)
            with open(target, 'r') as f:
                config_data = yaml.safe_load(f)
            config_data['dataset'] = slot_name
            config_data['description'] = f'Phase 2 homeostasis (time_step={FIXED_TIME_STEPS[slot]})'
            # Phase 2 training settings
            config_data['training']['homeostasis_training'] = True
            config_data['training']['skip_phase1'] = True
            config_data['training']['homeostasis_time_step'] = FIXED_TIME_STEPS[slot]
            config_data['training']['n_epochs'] = claude_n_epochs
            config_data['training']['data_augmentation_loop'] = claude_data_augmentation_loop
            config_data['claude'] = {
                'n_epochs': claude_n_epochs,
                'data_augmentation_loop': claude_data_augmentation_loop,
                'n_iter_block': n_iter_block,
                'ucb_c': claude_ucb_c,
                'node_name': claude_node_name
            }
            with open(target, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            print(f"\033[93m  slot {slot}: created {target} (ts={FIXED_TIME_STEPS[slot]}, dataset='{slot_name}')\033[0m")
        else:
            print(f"\033[93m  slot {slot}: preserving {target} (resuming)\033[0m")

    # -----------------------------------------------------------------------
    # shared files
    # -----------------------------------------------------------------------
    config_file = llm_task_name + '_00'
    analysis_path = f"{root_dir}/{llm_task_name}_analysis.md"
    memory_path = f"{root_dir}/{llm_task_name}_memory.md"
    ucb_path = f"{root_dir}/{llm_task_name}_ucb_scores.txt"
    instruction_path = f"{root_dir}/{instruction_name}.md"
    parallel_instruction_path = f"{root_dir}/instruction_phase2_parallel.md"
    reasoning_log_path = f"{root_dir}/{llm_task_name}_reasoning.log"

    exploration_dir = f"{root_dir}/log/Claude_exploration/{instruction_name}_parallel"
    log_dir = exploration_dir
    os.makedirs(log_dir, exist_ok=True)

    cluster_enabled = 'cluster' in task

    # check instruction files exist
    if not os.path.exists(instruction_path):
        print(f"\033[91merror: instruction file not found: {instruction_path}\033[0m")
        sys.exit(1)
    if not os.path.exists(parallel_instruction_path):
        print(f"\033[93mwarning: parallel instruction file not found: {parallel_instruction_path}\033[0m")
        parallel_instruction_path = None

    # initialize shared files on fresh start
    if start_iteration == 1 and not args.resume:
        with open(analysis_path, 'w') as f:
            f.write(f"# Phase 2 Experiment Log: {base_config_name} (parallel)\n\n")
        print(f"\033[93mcleared {analysis_path}\033[0m")
        open(reasoning_log_path, 'w').close()
        with open(memory_path, 'w') as f:
            f.write(f"# Phase 2 Working Memory: {base_config_name}\n\n")
            f.write("## Knowledge Base\n\n")
            f.write("### Time Step Comparison\n")
            f.write("| Block | ts=4 score | ts=16 score | ts=32 score | ts=64 score | Best strategy | Key finding |\n")
            f.write("| ----- | ---------- | ----------- | ----------- | ----------- | ------------- | ----------- |\n\n")
            f.write("### Established Principles\n\n")
            f.write("### Open Questions\n\n")
            f.write("---\n\n")
            f.write("## Previous Block Summary\n\n")
            f.write("---\n\n")
            f.write("## Current Block (Block 1)\n\n")
            f.write("### Block Info\n\n")
            f.write("### Strategy Under Test\n\n")
            f.write("### Iterations This Block\n\n")
            f.write("### Emerging Observations\n\n")
        print(f"\033[93mcleared {memory_path}\033[0m")
        if os.path.exists(ucb_path):
            os.remove(ucb_path)

    print(f"\033[93mPhase 2 PARALLEL (N={N_PARALLEL}, {n_iterations} iterations, starting at {start_iteration})\033[0m")

    # -----------------------------------------------------------------------
    # setup Phase 2 data symlinks and checkpoints
    # -----------------------------------------------------------------------
    print(f"\n\033[93mSetting up Phase 2 data and checkpoints\033[0m")

    # local data symlinks
    slot_dataset_full = [slot_names[s] for s in range(N_PARALLEL)]
    phase1_dataset_dir = os.path.join(root_dir, 'graphs_data', phase1_dataset)
    if os.path.isdir(phase1_dataset_dir):
        setup_phase2_data(phase1_dataset_dir, slot_dataset_full, root_dir)
    else:
        print(f"\033[93m  Phase 1 data dir not found locally: {phase1_dataset_dir}\033[0m")
        print(f"\033[93m  (will rely on cluster-side data)\033[0m")

    # cluster data symlinks
    if cluster_enabled:
        setup_cluster_data_symlinks(phase1_dataset, slot_dataset_full)

    # setup Phase 1 checkpoints for each slot (local + cluster)
    best_model_labels = {}
    for slot in range(N_PARALLEL):
        # local: always needed (for test+plot even in cluster mode)
        label = setup_local_checkpoint(
            phase1_checkpoint, slot_names[slot], root_dir
        )
        best_model_labels[slot] = label

        # cluster: copy model to cluster slot dir
        if cluster_enabled:
            label_cluster = setup_cluster_checkpoint(
                phase1_checkpoint, slot_names[slot], root_dir
            )
            if not label:
                best_model_labels[slot] = label_cluster

        if best_model_labels[slot]:
            print(f"\033[92m  slot {slot}: checkpoint label = '{best_model_labels[slot]}'\033[0m")
        else:
            print(f"\033[91m  slot {slot}: no checkpoint found!\033[0m")

    # -----------------------------------------------------------------------
    # BATCH 0: Claude "start" call — initialize 4 config variations
    # -----------------------------------------------------------------------
    if start_iteration == 1 and not args.resume:
        print(f"\n\033[94m{'='*60}\033[0m")
        print(f"\033[94mBATCH 0: Claude initializing {N_PARALLEL} Phase 2 config variations\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        slot_list = "\n".join(
            f"  Slot {s} (time_step={FIXED_TIME_STEPS[s]}): {config_paths[s]}"
            for s in range(N_PARALLEL)
        )

        parallel_ref = f"\nParallel instructions: {parallel_instruction_path}" if parallel_instruction_path else ""

        start_prompt = f"""PARALLEL START: Initialize {N_PARALLEL} Phase 2 config variations for the first batch.

Instructions (follow all instructions): {instruction_path}{parallel_ref}
Working memory: {memory_path}
Full log (append only): {analysis_path}

Config files to edit (all {N_PARALLEL}):
{slot_list}

Fixed time_steps per slot: {FIXED_TIME_STEPS}
Phase 1 checkpoint: {phase1_checkpoint}

Read the instructions and the base config. Each slot already has a unique dataset name
and fixed homeostasis_time_step — do NOT change these fields.

For the first batch, use the SAME training parameters across all 4 slots to establish
a baseline comparison of the 4 different time_steps. Only vary learning rates or other
Phase 2 training parameters.

Write the planned initial config to the working memory file."""

        print("\033[93mClaude start call...\033[0m")
        output_text = run_claude_cli(start_prompt, root_dir, max_turns=100)

        if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
            print(f"\n\033[91mOAuth token expired during start call\033[0m")
            print("\033[93m  1. run: claude /login\033[0m")
            print(f"\033[93m  2. then re-run this script\033[0m")
            sys.exit(1)

        if output_text.strip():
            with open(reasoning_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"=== BATCH 0 (start call) ===\n")
                f.write(f"{'='*60}\n")
                f.write(output_text.strip())
                f.write("\n\n")

    # -----------------------------------------------------------------------
    # main batch loop
    # -----------------------------------------------------------------------
    for batch_start in range(start_iteration, n_iterations + 1, N_PARALLEL):
        iterations = [batch_start + s for s in range(N_PARALLEL)
                      if batch_start + s <= n_iterations]

        batch_first = iterations[0]
        batch_last = iterations[-1]
        n_slots = len(iterations)

        block_number = (batch_first - 1) // n_iter_block + 1
        iter_in_block_first = (batch_first - 1) % n_iter_block + 1
        iter_in_block_last = (batch_last - 1) % n_iter_block + 1
        is_block_end = any((it - 1) % n_iter_block + 1 == n_iter_block for it in iterations)
        is_block_start = iter_in_block_first == 1
        batch_in_block = 1 if iter_in_block_first <= N_PARALLEL else 2

        # block boundary: erase UCB at start of new block
        if batch_first > 1 and (batch_first - 1) % n_iter_block == 0:
            if os.path.exists(ucb_path):
                os.remove(ucb_path)
                print(f"\033[93mblock boundary: deleted {ucb_path}\033[0m")

        print(f"\n\n\033[94m{'='*60}\033[0m")
        print(f"\033[94mBATCH: iters {batch_first}-{batch_last} / {n_iterations}  "
              f"(block {block_number}, batch {batch_in_block}/{BATCHES_PER_BLOCK})\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        # -----------------------------------------------------------
        # BETWEEN-BLOCK CODE REVIEW (at start of blocks > 1)
        # -----------------------------------------------------------
        if is_block_start and block_number > 1:
            print(f"\n\033[95m>>> BLOCK BOUNDARY: Claude code review <<<\033[0m")

            code_review_prompt = f""">>> BLOCK END + CODE REVIEW <<<

Block {block_number - 1} is complete. Before starting block {block_number}, review results
and optionally modify the Phase 2 training code.

Instructions: {instruction_path}
{f'Parallel instructions: {parallel_instruction_path}' if parallel_instruction_path else ''}
Working memory: {memory_path}
Full log (append only): {analysis_path}

Phase 2 training code: {root_dir}/src/MetabolismGraph/models/graph_trainer.py
(Only modify code between the Phase 2 markers: `# ===== Phase 2:` and `# --- final analysis`)

Review the last block's results across all 4 time_steps. If a strategy change is warranted:
1. Explain your rationale with literature references
2. Edit the Phase 2 code block in graph_trainer.py
3. Update the working memory with the new strategy

If no code change is needed, just update configs and memory for the next block."""

            output_text = run_claude_cli(code_review_prompt, root_dir,
                                        max_turns=200, allow_code_edit=True)

            # sync modified code to cluster
            if cluster_enabled:
                print(f"\033[96msyncing code to cluster after code review\033[0m")
                sync_code_to_cluster(root_dir)

            if output_text.strip():
                with open(reasoning_log_path, 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"=== Block {block_number - 1} -> {block_number} code review ===\n")
                    f.write(f"{'='*60}\n")
                    f.write(output_text.strip())
                    f.write("\n\n")

        # -----------------------------------------------------------
        # PHASE 1: no data generation for Phase 2 (reuses Phase 1 data)
        # -----------------------------------------------------------
        print(f"\n\033[93mPHASE 1: loading configs (no data generation)\033[0m")

        configs = {}
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            config = MetabolismGraphConfig.from_yaml(config_paths[slot])
            config.config_file = slot_names[slot]
            configs[slot] = config

            if device == []:
                device = set_device(config.training.device)

        # -----------------------------------------------------------
        # PHASE 2: submit 4 training jobs
        # -----------------------------------------------------------
        job_results = {}

        if "train" in task:
            if cluster_enabled:
                print(f"\n\033[93mPHASE 2: submitting {n_slots} Phase 2 training jobs to cluster\033[0m")

                job_ids = {}
                for slot_idx, iteration in enumerate(iterations):
                    slot = slot_idx
                    config = configs[slot]
                    jid = submit_cluster_job(
                        slot=slot,
                        config_path=config_paths[slot],
                        analysis_log_path=analysis_log_paths[slot],
                        config_file_field=config.config_file,
                        log_dir=log_dir,
                        root_dir=root_dir,
                        erase=True,
                        node_name=claude_node_name,
                        best_model=best_model_labels.get(slot)
                    )
                    if jid:
                        job_ids[slot] = jid
                    else:
                        job_results[slot] = False

                # wait for all submitted jobs
                if job_ids:
                    print(f"\n\033[93mPHASE 3: waiting for {len(job_ids)} cluster jobs\033[0m")
                    cluster_results = wait_for_cluster_jobs(job_ids, log_dir=log_dir, poll_interval=60)
                    job_results.update(cluster_results)

                # auto-repair for training errors
                for slot_idx in range(n_slots):
                    if job_results.get(slot_idx) == False:
                        err_content = None
                        err_file = f"{log_dir}/training_error_{slot_idx:02d}.log"
                        lsf_err_file = f"{log_dir}/cluster_train_{slot_idx:02d}.err"

                        for ef_path in [err_file, lsf_err_file]:
                            if os.path.exists(ef_path):
                                try:
                                    with open(ef_path, 'r') as ef:
                                        content = ef.read()
                                    if 'TRAINING SUBPROCESS ERROR' in content or 'Traceback' in content:
                                        err_content = content
                                        break
                                except Exception:
                                    pass

                        if not err_content:
                            continue

                        print(f"\033[91m  slot {slot_idx}: TRAINING ERROR — attempting auto-repair\033[0m")

                        code_files = [
                            'src/MetabolismGraph/models/graph_trainer.py',
                        ]
                        modified_code = get_modified_code_files(root_dir, code_files) if is_git_repo(root_dir) else code_files

                        max_repair_attempts = 3
                        repaired = False
                        for attempt in range(max_repair_attempts):
                            print(f"\033[93m  slot {slot_idx}: repair attempt {attempt + 1}/{max_repair_attempts}\033[0m")
                            repair_prompt = f"""TRAINING CRASHED - Please fix the code error.

Error traceback:
```
{err_content[-3000:]}
```

Modified files: {chr(10).join(f'- {root_dir}/{f}' for f in modified_code)}

Fix the bug in the Phase 2 training code. Do NOT make other changes."""

                            repair_cmd = [
                                'claude', '-p', repair_prompt,
                                '--output-format', 'text', '--max-turns', '10',
                                '--allowedTools', 'Read', 'Edit', 'Write'
                            ]
                            repair_result = subprocess.run(repair_cmd, cwd=root_dir, capture_output=True, text=True)
                            if 'CANNOT_FIX' in repair_result.stdout:
                                break

                            # sync fix to cluster
                            sync_code_to_cluster(root_dir)

                            # resubmit
                            config = configs[slot_idx]
                            jid = submit_cluster_job(
                                slot=slot_idx,
                                config_path=config_paths[slot_idx],
                                analysis_log_path=analysis_log_paths[slot_idx],
                                config_file_field=config.config_file,
                                log_dir=log_dir,
                                root_dir=root_dir,
                                erase=True,
                                node_name=claude_node_name,
                                best_model=best_model_labels.get(slot_idx)
                            )
                            if jid:
                                retry_results = wait_for_cluster_jobs(
                                    {slot_idx: jid}, log_dir=log_dir, poll_interval=60
                                )
                                if retry_results.get(slot_idx):
                                    job_results[slot_idx] = True
                                    repaired = True
                                    print(f"\033[92m  slot {slot_idx}: repair successful!\033[0m")
                                    break
                                for ef_path in [err_file, lsf_err_file]:
                                    if os.path.exists(ef_path):
                                        try:
                                            with open(ef_path, 'r') as ef:
                                                err_content = ef.read()
                                            break
                                        except Exception:
                                            pass

                        if not repaired:
                            print(f"\033[91m  slot {slot_idx}: repair failed — reverting code\033[0m")
                            if is_git_repo(root_dir):
                                for fp in code_files:
                                    try:
                                        subprocess.run(['git', 'checkout', 'HEAD', '--', fp],
                                                      cwd=root_dir, capture_output=True, timeout=10)
                                    except Exception:
                                        pass
                                sync_code_to_cluster(root_dir)

            else:
                # local execution (no cluster)
                print(f"\n\033[93mPHASE 2: training {n_slots} models locally\033[0m")

                for slot_idx, iteration in enumerate(iterations):
                    slot = slot_idx
                    config = configs[slot]
                    print(f"\033[90m  slot {slot} (iter {iteration}, ts={FIXED_TIME_STEPS[slot]}): training...\033[0m")

                    log_file = open(analysis_log_paths[slot], 'w')
                    try:
                        data_train(
                            config=config,
                            erase=True,
                            best_model=best_model_labels.get(slot, ''),
                            device=device,
                            log_file=log_file
                        )
                        job_results[slot] = True
                    except Exception as e:
                        print(f"\033[91m  slot {slot}: training failed: {e}\033[0m")
                        job_results[slot] = False
                    finally:
                        log_file.close()

        else:
            for slot in range(n_slots):
                job_results[slot] = True

        # -----------------------------------------------------------
        # PHASE 4: test + plot for successful slots
        # -----------------------------------------------------------
        print(f"\n\033[93mPHASE 4: test + plot\033[0m")

        activity_paths = {}
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            if not job_results.get(slot, False):
                print(f"\033[90m  slot {slot} (iter {iteration}): skipping (failed)\033[0m")
                continue

            config = configs[slot]
            log_file = open(analysis_log_paths[slot], 'a')

            if "test" in task:
                config.simulation.noise_model_level = 0.0
                data_test(
                    config=config,
                    best_model='best',
                    device=device,
                    log_file=log_file,
                )

            if 'plot' in task:
                slot_config_file = slot_names[slot]
                folder_name = f'./log/{slot_names[slot]}/tmp_results/'
                os.makedirs(folder_name, exist_ok=True)
                data_plot(
                    config=config,
                    config_file=slot_config_file,
                    epoch_list=['best'],
                    style='color',
                    extended='plots',
                    device=device,
                    apply_weight_correction=True,
                    log_file=log_file
                )

            log_file.close()

            # save exploration artifacts
            iter_in_block = (iteration - 1) % n_iter_block + 1
            artifact_paths = save_exploration_artifacts(
                root_dir, exploration_dir, config, slot_names[slot],
                '', iteration,
                iter_in_block=iter_in_block, block_number=block_number
            )
            activity_paths[slot] = artifact_paths.get('concentrations_path', 'N/A')

            # save config snapshot
            config_save_dir = f"{exploration_dir}/config"
            os.makedirs(config_save_dir, exist_ok=True)
            dst_config = f"{config_save_dir}/iter_{iteration:03d}_slot_{slot:02d}.yaml"
            shutil.copy2(config_paths[slot], dst_config)

        # -----------------------------------------------------------
        # PHASE 5: compute Phase 2 scores + UCB
        # -----------------------------------------------------------
        print(f"\n\033[93mPHASE 5: computing Phase 2 scores\033[0m")

        phase2_scores = {}
        for slot_idx, iteration in enumerate(iterations):
            if not job_results.get(slot_idx, False):
                continue
            scores = compute_phase2_score(analysis_log_paths[slot_idx])
            phase2_scores[slot_idx] = scores
            print(f"  slot {slot_idx} (ts={FIXED_TIME_STEPS[slot_idx]}): "
                  f"phase2_score={scores['phase2_score']:.4f} "
                  f"(slope_acc={scores['slope_accuracy']:.4f}, "
                  f"cluster={scores['embedding_cluster_acc']:.4f}, "
                  f"R2={scores['rate_constants_R2']:.4f})")

        # UCB scoring with phase2_score as primary metric
        ucb_c = claude_ucb_c
        existing_content = ""
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r') as f:
                existing_content = f.read()

        stub_entries = ""
        for slot_idx, iteration in enumerate(iterations):
            if not job_results.get(slot_idx, False):
                continue
            scores = phase2_scores.get(slot_idx, {})
            p2_score = scores.get('phase2_score', 0.0)
            if f'## Iter {iteration}:' not in existing_content:
                stub_entries += (
                    f"\n## Iter {iteration}: pending\n"
                    f"Node: id={iteration}, parent=root\n"
                    f"Metrics: phase2_score={p2_score:.4f}\n"
                )

        tmp_analysis = analysis_path + '.tmp_ucb'
        with open(tmp_analysis, 'w') as f:
            f.write(existing_content + stub_entries)

        compute_ucb_scores(
            tmp_analysis, ucb_path, c=ucb_c,
            current_log_path=None,
            current_iteration=batch_last,
            block_size=n_iter_block,
            config_file=config_file,
            primary_metric='phase2_score'
        )
        os.remove(tmp_analysis)
        print(f"\033[92mUCB scores computed: {ucb_path}\033[0m")

        # -----------------------------------------------------------
        # PHASE 6: Claude analysis + mutations
        # -----------------------------------------------------------
        print(f"\n\033[93mPHASE 6: Claude analysis + next mutations\033[0m")

        slot_info_lines = []
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            status = "COMPLETED" if job_results.get(slot, False) else "FAILED"
            act_path = activity_paths.get(slot, "N/A")
            scores = phase2_scores.get(slot, {})
            score_str = (
                f"phase2_score={scores.get('phase2_score', 0):.4f}, "
                f"slope_acc={scores.get('slope_accuracy', 0):.4f}, "
                f"cluster_acc={scores.get('embedding_cluster_acc', 0):.4f}, "
                f"R2={scores.get('rate_constants_R2', 0):.4f}"
            ) if scores else "N/A"
            slot_info_lines.append(
                f"Slot {slot} (iteration {iteration}, time_step={FIXED_TIME_STEPS[slot]}) [{status}]:\n"
                f"  Phase 2 scores: {score_str}\n"
                f"  Metrics log: {analysis_log_paths[slot]}\n"
                f"  Concentrations: {act_path}\n"
                f"  Config: {config_paths[slot]}"
            )
        slot_info = "\n\n".join(slot_info_lines)

        block_end_marker = ""
        if is_block_end:
            block_end_marker = "\n>>> BLOCK END <<<"

        parallel_ref = f"\nParallel instructions: {parallel_instruction_path}" if parallel_instruction_path else ""

        claude_prompt = f"""Batch iterations {batch_first}-{batch_last} / {n_iterations}
Block info: block {block_number}, batch {batch_in_block}/{BATCHES_PER_BLOCK}, iterations {iter_in_block_first}-{iter_in_block_last}/{n_iter_block} within block{block_end_marker}

PARALLEL MODE: Analyze {n_slots} Phase 2 results, then propose next {N_PARALLEL} mutations.

Instructions (follow all instructions): {instruction_path}{parallel_ref}
Working memory: {memory_path}
Full log (append only): {analysis_path}
UCB scores: {ucb_path}

{slot_info}

Analyze all {n_slots} results. For each successful slot, write a separate iteration entry
(## Iter N: ...) to the full log and memory file. Then edit all {N_PARALLEL} config files
to set up the next batch.

IMPORTANT:
- Do NOT change 'dataset' or 'homeostasis_time_step' in any config
- Do NOT change simulation parameters
- Only modify Phase 2 training parameters (learning rates, augmentation, batch size)"""

        print("\033[93mClaude analysis...\033[0m")
        output_text = run_claude_cli(claude_prompt, root_dir)

        if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
            print(f"\n\033[91mOAuth token expired at batch {batch_first}-{batch_last}\033[0m")
            print("\033[93m  1. run: claude /login\033[0m")
            print(f"\033[93m  2. re-run with --resume\033[0m")
            sys.exit(1)

        if output_text.strip():
            with open(reasoning_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"=== Batch {batch_first}-{batch_last} ===\n")
                f.write(f"{'='*60}\n")
                f.write(output_text.strip())
                f.write("\n\n")

        # recompute UCB after Claude writes entries
        compute_ucb_scores(analysis_path, ucb_path, c=ucb_c,
                           current_log_path=None,
                           current_iteration=batch_last,
                           block_size=n_iter_block,
                           config_file=config_file,
                           primary_metric='phase2_score')

        # UCB tree visualization
        should_save_tree = is_block_start or is_block_end
        if should_save_tree and os.path.exists(ucb_path):
            tree_save_dir = f"{exploration_dir}/exploration_tree"
            os.makedirs(tree_save_dir, exist_ok=True)
            ucb_tree_path = f"{tree_save_dir}/ucb_tree_iter_{batch_last:03d}.png"
            nodes = parse_ucb_scores(ucb_path)
            if nodes:
                config = configs[0]
                sim_info = f"Phase 2 homeostasis (ts={FIXED_TIME_STEPS})"
                plot_ucb_tree(nodes, ucb_tree_path,
                              title=f"Phase 2 UCB - Batch {batch_first}-{batch_last}",
                              simulation_info=sim_info)

        # save instruction file at first iteration of each block
        protocol_save_dir = f"{exploration_dir}/protocol"
        os.makedirs(protocol_save_dir, exist_ok=True)
        if is_block_start:
            dst_instruction = f"{protocol_save_dir}/block_{block_number:03d}.md"
            if os.path.exists(instruction_path):
                shutil.copy2(instruction_path, dst_instruction)

        # save memory at end of block
        if is_block_end:
            memory_save_dir = f"{exploration_dir}/memory"
            os.makedirs(memory_save_dir, exist_ok=True)
            dst_memory = f"{memory_save_dir}/block_{block_number:03d}_memory.md"
            if os.path.exists(memory_path):
                shutil.copy2(memory_path, dst_memory)
                print(f"\033[92msaved memory snapshot: {dst_memory}\033[0m")

        # batch summary
        n_success = sum(1 for v in job_results.values() if v)
        n_failed = sum(1 for v in job_results.values() if not v)
        print(f"\n\033[92mbatch {batch_first}-{batch_last} complete: "
              f"{n_success} succeeded, {n_failed} failed\033[0m")


# python GNN_LLM_phase2.py --phase1-checkpoint log/iter_096/models/best_model_with_0_graphs_1_342000.pt -o train_test_plot_Claude_cluster phase2_homeostasis iterations=128
# python GNN_LLM_phase2.py --phase1-checkpoint log/iter_096/models/best_model_with_0_graphs_1_342000.pt -o train_test_plot_Claude_cluster phase2_homeostasis iterations=128 --resume
