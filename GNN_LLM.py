import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse
import os
import shutil
import subprocess
import time
import re
import yaml

# redirect PyTorch JIT cache to /scratch instead of /tmp (per IT request)
if os.path.isdir('/scratch'):
    os.environ['TMPDIR'] = '/scratch/allierc'
    os.makedirs('/scratch/allierc', exist_ok=True)


import sys

from MetabolismGraph.config import MetabolismGraphConfig
from MetabolismGraph.generators.data_generator import data_generate
from MetabolismGraph.models.graph_trainer import data_train, data_test
from MetabolismGraph.models.exploration_tree import compute_ucb_scores
from MetabolismGraph.models.plot_exploration_tree import parse_ucb_scores, plot_ucb_tree
from MetabolismGraph.models.utils import save_exploration_artifacts
from MetabolismGraph.utils import set_device, add_pre_folder
from MetabolismGraph.git_code_tracker import track_code_modifications, is_git_repo, get_modified_code_files
from GNN_PlotFigure import data_plot

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

def detect_last_iteration(analysis_path):
    """detect the last completed iteration from analysis.md.

    scans for '## Iter N:' entries written by Claude after each iteration.
    returns the next iteration to run (1-indexed), or 1 if nothing found.
    """
    found_iters = set()
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            for line in f:
                match = re.match(r'^##+ Iter (\d+):', line)
                if match:
                    found_iters.add(int(match.group(1)))
    if not found_iters:
        return 1
    return max(found_iters) + 1


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="MetabolismGraph")
    parser.add_argument(
        "-o", "--option", nargs="+", help="option that takes multiple values"
    )
    parser.add_argument(
        "--fresh", action="store_true", default=True, help="start from iteration 1 (default)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="auto-resume from last completed iteration"
    )



    print()
    device=[]
    args = parser.parse_args()

    if args.option:
        print(f"options: {args.option}")
    if args.option != None:
        task = args.option[0]
        config_list = [args.option[1]]
        if len(args.option) > 2:
            best_model = args.option[2]
        else:
            best_model = None
        task_params = {}
        for arg in args.option[2:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                task_params[key] = int(value) if value.isdigit() else value
    else:
        best_model = ''
        task = 'test_Claude'  # 'train', 'test', 'generate', 'plot', 'Claude', 'code', 'cluster'
        config_list = ['metabolism']
        task_params = {'iterations': 64}



    n_iterations = task_params.get('iterations', 5)
    base_config_name = config_list[0] if config_list else 'metabolism'
    instruction_name = task_params.get('instruction', f'instruction_{base_config_name}')
    llm_task_name = task_params.get('llm_task', f'{base_config_name}_Claude')

    # auto-resume or fresh start
    _root = os.path.dirname(os.path.abspath(__file__))
    if args.resume:
        start_iteration = detect_last_iteration(f"{_root}/{llm_task_name}_analysis.md")
        if start_iteration > 1:
            print(f"\033[93mresuming from iteration {start_iteration}\033[0m")
        else:
            print(f"\033[93mno previous iterations found, starting fresh\033[0m")
    else:
        start_iteration = 1
        _analysis_check = f"{_root}/{llm_task_name}_analysis.md"
        if os.path.exists(_analysis_check):
            print(f"\033[91mwarning: fresh start will erase existing results in:\033[0m")
            print(f"\033[91m  {_analysis_check}\033[0m")
            print(f"\033[91m  {_root}/{llm_task_name}_memory.md\033[0m")
            answer = input("\033[91mcontinue? (y/n): \033[0m").strip().lower()
            if answer != 'y':
                print("aborted.")
                sys.exit(0)
        print(f"\033[93mfresh start\033[0m")

    if 'Claude' in task:
        iteration_range = range(start_iteration, n_iterations + 1)


        root_dir = os.path.dirname(os.path.abspath(__file__))
        config_root = root_dir + "/config"

        if start_iteration > 1:
            print(f"\033[93mresuming from iteration {start_iteration}\033[0m")


        for cfg in config_list:
            cfg_file, pre = add_pre_folder(cfg)
            source_config = f"{config_root}/{pre}{cfg}.yaml"
            target_config = f"{config_root}/{pre}{llm_task_name}.yaml"

            # only copy and initialize config on fresh start (not when resuming)
            if start_iteration == 1 and not args.resume:
                # erase config from new/processing/done directories
                for subdir in ['new', 'processing', 'done']:
                    cleanup_path = f"{config_root}/{subdir}/{llm_task_name}.yaml"
                    if os.path.exists(cleanup_path):
                        os.remove(cleanup_path)
                        print(f"\033[93mdeleted {cleanup_path}\033[0m")

                if os.path.exists(source_config):
                    shutil.copy2(source_config, target_config)
                    print(f"\033[93mcopied {source_config} -> {target_config}\033[0m")
                    with open(target_config, 'r') as f:
                        config_data = yaml.safe_load(f)
                    claude_cfg = config_data.get('claude', {})
                    claude_n_epochs = claude_cfg.get('n_epochs', 1)
                    claude_data_augmentation_loop = claude_cfg.get('data_augmentation_loop', 100)
                    claude_n_iter_block = claude_cfg.get('n_iter_block', 24)
                    claude_ucb_c = claude_cfg.get('ucb_c', 1.414)
                    claude_node_name = claude_cfg.get('node_name', 'a100')
                    config_data['dataset'] = llm_task_name
                    config_data['training']['n_epochs'] = claude_n_epochs
                    config_data['training']['data_augmentation_loop'] = claude_data_augmentation_loop
                    config_data['description'] = 'designed by Claude'
                    config_data['claude'] = {
                        'n_epochs': claude_n_epochs,
                        'data_augmentation_loop': claude_data_augmentation_loop,
                        'n_iter_block': claude_n_iter_block,
                        'ucb_c': claude_ucb_c,
                        'node_name': claude_node_name
                    }
                    with open(target_config, 'w') as f:
                        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                    print(f"\033[93mmodified {target_config}: dataset='{llm_task_name}', n_epochs={claude_n_epochs}, data_augmentation_loop={claude_data_augmentation_loop}, n_iter_block={claude_n_iter_block}, ucb_c={claude_ucb_c}, node_name={claude_node_name}\033[0m")
            else:
                print(f"\033[93mpreserving {target_config} (resuming from iter {start_iteration})\033[0m")
                # load existing config to get claude parameters
                with open(target_config, 'r') as f:
                    config_data = yaml.safe_load(f)
                claude_cfg = config_data.get('claude', {})
                claude_n_epochs = claude_cfg.get('n_epochs', 1)
                claude_data_augmentation_loop = claude_cfg.get('data_augmentation_loop', 100)
                claude_n_iter_block = claude_cfg.get('n_iter_block', 24)
                claude_ucb_c = claude_cfg.get('ucb_c', 1.414)
                claude_node_name = claude_cfg.get('node_name', 'a100')

        n_iter_block = claude_n_iter_block

        print(f"\033[94mcluster node: gpu_{claude_node_name}\033[0m")

        ucb_file = f"{root_dir}/{llm_task_name}_ucb_scores.txt"
        if start_iteration == 1 and not args.resume:
            # only delete UCB file when starting fresh (not resuming)
            if os.path.exists(ucb_file):
                os.remove(ucb_file)
                print(f"\033[93mdeleted {ucb_file}\033[0m")
        else:
            print(f"\033[93mpreserving {ucb_file} (resuming from iter {start_iteration})\033[0m")

        config_list = [llm_task_name]

        # track if code was modified by Claude (starts False, set True after Claude modifies code)
        code_modified_by_claude = False
        # check if code modifications are enabled (task contains 'code')
        code_changes_enabled = 'code' in task
        # check if cluster execution is enabled (task contains 'cluster')
        cluster_enabled = 'cluster' in task
        if code_changes_enabled:
            print("\033[93mcode modifications enabled (task contains 'code')\033[0m")
            if cluster_enabled:
                print("\033[93mcluster execution enabled (task contains 'cluster')\033[0m")
            else:
                print("\033[90mcluster execution disabled (add 'cluster' to task to enable)\033[0m")
        else:
            print("\033[90mcode modifications disabled (add 'code' to task to enable)\033[0m")
    else:
        iteration_range = range(1, 2)
        code_modified_by_claude = False
        code_changes_enabled = False
        cluster_enabled = False




    for config_file_ in config_list:
        print(" ")
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        config_file, pre_folder = add_pre_folder(config_file_)



        if 'Claude' in task:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            instruction_path = f"{root_dir}/{instruction_name}.md"
            analysis_path = f"{root_dir}/{llm_task_name}_analysis.md"
            memory_path = f"{root_dir}/{llm_task_name}_memory.md"

            # check instruction file exists
            if not os.path.exists(instruction_path):
                print(f"\033[91merror: instruction file not found: {instruction_path}\033[0m")
                print("\033[93mavailable instruction files:\033[0m")
                for f in os.listdir(root_dir):
                    if f.endswith('.md') and not f.startswith('analysis_') and not f.startswith('README'):
                        print(f"  - {f[:-3]}")
                continue

            # clear analysis and memory files at start (only if not resuming)
            if start_iteration == 1 and not args.resume:
                with open(analysis_path, 'w') as f:
                    f.write(f"# Experiment Log: {config_file_}\n\n")
                print(f"\033[93mcleared {analysis_path}\033[0m")
                # clear reasoning.log for Claude tasks
                reasoning_path = analysis_path.replace('_analysis.md', '_reasoning.log')
                open(reasoning_path, 'w').close()
                print(f"\033[93mcleared {reasoning_path}\033[0m")
                # initialize working memory file
                with open(memory_path, 'w') as f:
                    f.write(f"# Working Memory: {config_file_}\n\n")
                    f.write("## Knowledge Base (accumulated across all blocks)\n\n")
                    f.write("### Regime Comparison Table\n")
                    f.write("| Block | n_metabolites | n_reactions | n_frames | noise | eff_rank | Best R² | Optimal lr_S | Optimal L1 | Key finding |\n")
                    f.write("| ----- | ------------- | ----------- | -------- | ----- | -------- | ------- | ------------ | ---------- | ----------- |\n\n")
                    f.write("### Established Principles\n\n")
                    f.write("### Open Questions\n\n")
                    f.write("---\n\n")
                    f.write("## Previous Block Summary\n\n")
                    f.write("---\n\n")
                    f.write("## Current Block (Block 1)\n\n")
                    f.write("### Block Info\n\n")
                    f.write("### Hypothesis\n\n")
                    f.write("### Iterations This Block\n\n")
                    f.write("### Emerging Observations\n\n")
                print(f"\033[93mcleared {memory_path}\033[0m")
            else:
                print(f"\033[93mpreserving {analysis_path} (resuming from iter {start_iteration})\033[0m")
                print(f"\033[93mpreserving {memory_path} (resuming from iter {start_iteration})\033[0m")
                reasoning_path = analysis_path.replace('_analysis.md', '_reasoning.log')
                print(f"\033[93mpreserving {reasoning_path} (resuming from iter {start_iteration})\033[0m")
            print(f"\033[93m{instruction_name} ({n_iterations} iterations, starting at {start_iteration})\033[0m")

        root_dir = os.path.dirname(os.path.abspath(__file__))
        analysis_log_path = f"{root_dir}/{llm_task_name}_analysis.log"



        for iteration in iteration_range:


            if 'Claude' in task:
                print(f"\n\n\n\033[94miteration {iteration}/{n_iterations}: {config_file_} ===\033[0m")
                # block boundary: erase UCB at start of each n_iter_block-iteration block (except iter 1, already handled)
                if iteration > 1 and (iteration - 1) % n_iter_block == 0:
                    ucb_file = f"{root_dir}/{llm_task_name}_ucb_scores.txt"
                    if os.path.exists(ucb_file):
                        os.remove(ucb_file)
                        print(f"\033[93msimulation block boundary: deleted {ucb_file} (new simulation block)\\033[0m")

            # reload config to pick up any changes from previous iteration
            config = MetabolismGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_

            if device==[]:
                device = set_device(config.training.device)

            # open analysis.log for this iteration (append mode for test/plot to add metrics)
            log_file = open(analysis_log_path, 'w')

            if 'daemon' in task:
                # daemon mode: submit job to cluster via config/new/ directory
                # copy current config to config/new/, wait for cluster to process it
                # use basename to flatten the path (config/signal/foo.yaml -> config/new/foo.yaml)
                config_filename = f"{os.path.basename(config_file)}.yaml"
                new_path = f"{config_root}/new/{config_filename}"
                processing_path = f"{config_root}/processing/{config_filename}"
                done_path = f"{config_root}/done/{config_filename}"

                # ensure directories exist
                os.makedirs(f"{config_root}/new", exist_ok=True)
                os.makedirs(f"{config_root}/processing", exist_ok=True)
                os.makedirs(f"{config_root}/done", exist_ok=True)

                # copy config to new/ to submit job
                source_config = f"{config_root}/{config_file}.yaml"
                shutil.copy2(source_config, new_path)
                submit_time = time.time()
                print(f"\033[93msubmitted to daemon: {new_path}\033[0m")

                # wait for job to complete (file appears in done/)
                print(f"\033[93mwaiting for {config_filename} to be copied into config/done/ ...\033[0m")
                check_interval = 1 * 60  # 1 minute in seconds
                while True:
                    if os.path.exists(done_path):
                        print(f"\033[92mconfig file {config_filename} copied into config/done/\033[0m")
                        time.sleep(5)  # wait 5s before deleting
                        os.remove(done_path)
                        print(f"\033[93mremoved from done/: {config_filename}\033[0m")
                        break
                    # also check if still in processing (job running)
                    if os.path.exists(processing_path):
                        print("  ... job still running (in processing/)")
                    elif os.path.exists(new_path):
                        print("  ... job queued (in new/)")
                    else:
                        print("  ... waiting for daemon to pick up job")
                    time.sleep(check_interval)
                print("\033[92mdaemon job completed, proceeding with Claude analysis...\033[0m")

                # close log file (metrics will be read from cluster output)
                log_file.close()

                # read analysis.log generated by daemon (if exists)
                # the daemon writes to the same log path, so metrics should be available
                if os.path.exists(analysis_log_path):
                    print(f"\033[92mmetrics available in: {analysis_log_path}\033[0m")

            else:
                # local execution mode
                if "generate" in task:
                    data_generate(
                        config,
                        visualize=False,
                        device=device,
                        bSave=True
                    )

                if "train" in task:
                    # training execution: cluster > subprocess (code modified) > direct
                    use_subprocess = 'Claude' in task and code_changes_enabled and code_modified_by_claude

                    if cluster_enabled or use_subprocess:
                        # paths for subprocess/cluster execution
                        config_path = f"{config_root}/{config_file}.yaml"
                        log_dir = f"{root_dir}/log/Claude_exploration/{instruction_name}"
                        os.makedirs(log_dir, exist_ok=True)
                        error_log_path = f"{log_dir}/training_output_latest.log"
                        error_details_path = f"{log_dir}/training_error_latest.log"

                        if cluster_enabled:
                            # submit training job to cluster via SSH + bsub
                            if use_subprocess:
                                print("\033[93mcode modified by Claude - submitting training to cluster...\033[0m")
                            else:
                                print("\033[93msubmitting training to cluster....\033[0m")

                            # build the python command
                            train_cmd = f"python train_metabolism_subprocess.py --config '{config_path}' --device cuda"
                            train_cmd += f" --log_file '{analysis_log_path}'"
                            train_cmd += f" --config_file '{config.config_file}'"
                            train_cmd += f" --error_log '{error_details_path}'"
                            if 'Claude' in task:
                                train_cmd += " --erase"

                            # create temporary bash script that activates conda and runs training
                            cluster_script_path = f"{log_dir}/cluster_train.sh"
                            # use absolute path for cluster (home directory on cluster)
                            cluster_home = "/groups/saalfeld/home/allierc"
                            cluster_root_dir = f"{cluster_home}/Graph/MetabolismGraph"
                            conda_path = f"{cluster_home}/miniforge3/etc/profile.d/conda.sh"

                            # update paths in train_cmd to use cluster paths
                            cluster_config_path = config_path.replace(root_dir, cluster_root_dir)
                            cluster_analysis_log = analysis_log_path.replace(root_dir, cluster_root_dir)
                            cluster_error_log = error_details_path.replace(root_dir, cluster_root_dir)

                            cluster_train_cmd = f"python train_metabolism_subprocess.py --config '{cluster_config_path}' --device cuda"
                            cluster_train_cmd += f" --log_file '{cluster_analysis_log}'"
                            cluster_train_cmd += f" --config_file '{config.config_file}'"
                            cluster_train_cmd += f" --error_log '{cluster_error_log}'"
                            if 'Claude' in task:
                                cluster_train_cmd += " --erase"

                            with open(cluster_script_path, 'w') as f:
                                f.write("#!/bin/bash\n")
                                f.write(f"cd {cluster_root_dir}\n")
                                f.write(f"conda run -n metabolism-graph {cluster_train_cmd}\n")
                            os.chmod(cluster_script_path, 0o755)

                            # path to script on cluster
                            cluster_script = cluster_script_path.replace(root_dir, cluster_root_dir)

                            # submit job to cluster via SSH to login1
                            # -W 6000 = 100 hours max wall time, -K makes bsub wait for job completion
                            ssh_cmd = f"ssh allierc@login1 \"cd {cluster_root_dir} && bsub -n 8 -gpu 'num=1' -q gpu_{claude_node_name} -W 6000 -K 'bash {cluster_script}'\""

                            print(f"\033[96msubmitting via SSH: {ssh_cmd}\033[0m")

                            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

                            if result.returncode != 0:
                                print("\033[91mcluster training failed:\033[0m")
                                print(f"stdout: {result.stdout}")
                                print(f"stderr: {result.stderr}")
                                if os.path.exists(error_details_path):
                                    with open(error_details_path, 'r') as f:
                                        print(f.read())
                                raise RuntimeError(f"cluster training failed at iteration {iteration}")

                            print("\033[92mcluster training completed successfully\033[0m")
                            print(result.stdout)

                        else:
                            # run training locally in subprocess (code was modified)
                            print("\033[93mcode modified by Claude - running training in subprocess...\033[0m")

                            train_script = os.path.join(root_dir, 'train_metabolism_subprocess.py')
                            train_cmd = [
                                sys.executable,  # use same Python interpreter
                                '-u',  # force unbuffered output for real-time streaming
                                train_script,
                                '--config', config_path,
                                '--device', str(device),
                                '--log_file', analysis_log_path,
                                '--config_file', config.config_file,
                                '--error_log', error_details_path,
                                '--erase'
                            ]

                            # run training subprocess with repair loop
                            env = os.environ.copy()
                            env['PYTHONUNBUFFERED'] = '1'
                            env['TQDM_DISABLE'] = '1'  # disable tqdm in subprocess (doesn't stream well)

                            # code files that Claude might modify
                            code_files = [
                                'src/MetabolismGraph/models/trainer.py',
                                'src/MetabolismGraph/generators/data_generator.py',
                            ]

                            max_repair_attempts = 10
                            training_success = False
                            error_traceback = None

                            for repair_attempt in range(max_repair_attempts + 1):
                                process = subprocess.Popen(
                                    train_cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True,
                                    bufsize=1,
                                    env=env
                                )

                                # capture all output for logging while also streaming to console
                                output_lines = []
                                with open(error_log_path, 'w') as output_file:
                                    for line in process.stdout:
                                        output_file.write(line)
                                        output_file.flush()
                                        output_lines.append(line.rstrip())
                                        # filter: skip tqdm-like lines (progress bars)
                                        if '|' in line and '%' in line and 'it/s' in line:
                                            continue
                                        print(line, end='', flush=True)

                                process.wait()

                                if process.returncode == 0:
                                    training_success = True
                                    break

                                # training failed - capture error info
                                error_traceback = '\n'.join(output_lines[-50:])  # last 50 lines

                                if repair_attempt == 0:
                                    print(f"\033[91m\ntraining subprocess failed with code {process.returncode}\033[0m")
                                    print("\033[93mthis may indicate a code modification error.\033[0m\n")

                                    # show last 20 lines of output for context
                                    print("\033[93mlast 20 lines of output:\033[0m")
                                    print("-" * 80)
                                    for line in output_lines[-20:]:
                                        print(line)
                                    print("-" * 80)

                                    # show paths to log files
                                    print(f"\nfull output logged to: {error_log_path}")
                                    if os.path.exists(error_details_path):
                                        print(f"error details logged to: {error_details_path}")
                                        try:
                                            with open(error_details_path, 'r') as f:
                                                error_details = f.read()
                                            if error_details.strip():
                                                print("\n\033[91mdetailed error information:\033[0m")
                                                print(error_details)
                                                error_traceback = error_details + '\n' + error_traceback
                                        except Exception as e:
                                            print(f"could not read error details: {e}")

                                # check if code was modified (only attempt repair for code errors)
                                modified_code = get_modified_code_files(root_dir, code_files) if is_git_repo(root_dir) else []

                                if not modified_code and repair_attempt == 0:
                                    print("\033[93mno code modifications detected - skipping repair attempts\033[0m")
                                    break

                                # attempt repair only if code was modified
                                if repair_attempt < max_repair_attempts and modified_code:
                                    print(f"\033[93mrepair attempt {repair_attempt + 1}/{max_repair_attempts}: asking Claude to fix the code error...\033[0m")

                                    repair_prompt = f"""TRAINING CRASHED - Please fix the code error.

Attempt {repair_attempt + 1}/{max_repair_attempts}

Error traceback:
```
{error_traceback[-3000:] if error_traceback else 'No traceback available'}
```

Modified code files that may contain the bug:
{chr(10).join(f'- {root_dir}/{f}' for f in modified_code)}

Instructions:
1. Read the error traceback carefully
2. Identify the bug in the modified code
3. Fix the bug using the Edit tool
4. Do NOT make other changes, only fix the crash

If you cannot fix it, say "CANNOT_FIX" and explain why."""

                                    repair_cmd = [
                                        'claude',
                                        '-p', repair_prompt,
                                        '--output-format', 'text',
                                        '--max-turns', '10',
                                        '--allowedTools', 'Read', 'Edit', 'Write'
                                    ]

                                    repair_result = subprocess.run(repair_cmd, cwd=root_dir, capture_output=True, text=True)
                                    repair_output = repair_result.stdout

                                    if 'CANNOT_FIX' in repair_output:
                                        print("\033[91mClaude cannot fix the error\033[0m")
                                        break

                                    print(f"\033[92mrepair attempt {repair_attempt + 1} complete, retrying training...\033[0m")

                        # if still failing after all attempts, rollback and skip iteration
                        if not training_success:
                            print("\033[91mall repair attempts failed - rolling back code changes\033[0m")

                            # rollback modified files using git
                            if is_git_repo(root_dir):
                                for file_path in code_files:
                                    try:
                                        subprocess.run(['git', 'checkout', 'HEAD', '--', file_path],
                                                      cwd=root_dir, capture_output=True, timeout=10)
                                    except:
                                        pass
                                print("\033[93mrolled back code to last working state\033[0m")

                            # log failed modification to memory
                            if os.path.exists(memory_path):
                                with open(memory_path, 'a') as f:
                                    f.write(f"\n### Failed Code Modification (Iter {iteration})\n")
                                    f.write(f"Error: {error_traceback[-500:] if error_traceback else 'Unknown'}\n")
                                    f.write("**DO NOT retry this modification**\n\n")

                            continue  # skip to next iteration

                        print("\033[92mtraining subprocess completed successfully\033[0m")
                    else:
                        # no cluster and no code modifications - run training directly (fastest)
                        data_train(
                            config=config,
                            erase='Claude' in task,  # erase old models when iterating with Claude
                            best_model=best_model,
                            style = 'color',
                            device=device,
                            log_file=log_file
                        )

                if "test" in task:

                    config.simulation.noise_model_level = 0.0

                    data_test(
                        config=config,
                        best_model='best',
                        device=device,
                        log_file=log_file,
                    )

                if 'plot' in task:
                    folder_name = './log/' + pre_folder + '/tmp_results/'
                    os.makedirs(folder_name, exist_ok=True)
                    data_plot(config=config, config_file=config_file, epoch_list=['best'], style='color', extended='plots', device=device, apply_weight_correction=True, log_file=log_file)

                log_file.close()

            if 'Claude' in task:

                block_number = (iteration - 1) // n_iter_block + 1
                iter_in_block = (iteration - 1) % n_iter_block + 1
                is_block_end = iter_in_block == n_iter_block

                exploration_dir = f"{root_dir}/log/Claude_exploration/{instruction_name}"
                artifact_paths = save_exploration_artifacts(
                    root_dir, exploration_dir, config, config_file_, pre_folder, iteration,
                    iter_in_block=iter_in_block, block_number=block_number
                )
                tree_save_dir = artifact_paths['tree_save_dir']
                protocol_save_dir = artifact_paths['protocol_save_dir']
                activity_path = artifact_paths['activity_path']

                # claude analysis: reads activity.png and analysis.log, updates config per experiment protocol
                config_path = f"{root_dir}/config/{pre_folder}{config_file_}.yaml"
                ucb_path = f"{root_dir}/{llm_task_name}_ucb_scores.txt"

                # read ucb_c from config
                with open(config_path, 'r') as f:
                    raw_config = yaml.safe_load(f)
                ucb_c = raw_config.get('claude', {}).get('ucb_c', 1.414)

                # compute UCB scores for Claude to read
                compute_ucb_scores(analysis_path, ucb_path, c=ucb_c,
                                   current_log_path=analysis_log_path,
                                   current_iteration=iteration,
                                   block_size=n_iter_block,
                                   config_file=config_file)
                print(f"\033[92mUCB scores computed (c={ucb_c}): {ucb_path}\033[0m")

                # check files are ready (generated by data_generate above)
                time.sleep(2)  # pause to ensure files are written
                if not os.path.exists(activity_path):
                    print(f"\033[91merror: activity.png not found at {activity_path}\033[0m")
                    continue
                if not os.path.exists(analysis_log_path):
                    print(f"\033[91merror: analysis.log not found at {analysis_log_path}\033[0m")
                    continue
                if not os.path.exists(ucb_path):
                    print(f"\033[91merror: ucb_scores.txt not found at {ucb_path}\033[0m")
                    continue
                print("\033[92mfiles ready: activity.png, analysis.log, ucb_scores.txt\033[0m")

                # call Claude CLI for analysis
                print("\033[93mClaude analysis...\033[0m")

                claude_prompt = f"""Iteration {iteration}/{n_iterations}
Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block
{">>> BLOCK END <<<" if is_block_end else ""}

Instructions (follow all instructions): {instruction_path}
Working memory: {memory_path}
Full log (append only): {analysis_path}
Activity image: {activity_path}
Metrics log: {analysis_log_path}
UCB scores: {ucb_path}
Current config: {config_path}"""

                claude_cmd = [
                    'claude',
                    '-p', claude_prompt,
                    '--output-format', 'text',
                    '--max-turns', '500',
                    '--allowedTools',
                    'Read', 'Edit', 'Write'
                ]

                # run with real-time output streaming and token expiry detection
                output_lines = []
                process = subprocess.Popen(
                    claude_cmd,
                    cwd=root_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                # stream output line by line
                for line in process.stdout:
                    print(line, end='', flush=True)
                    output_lines.append(line)

                process.wait()

                # check for OAuth token expiration error
                output_text = ''.join(output_lines)
                if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
                    print(f"\n\033[91m{'='*60}\033[0m")
                    print(f"\033[91mOAuth token expired at iteration {iteration}\033[0m")
                    print("\033[93mto resume:\033[0m")
                    print("\033[93m  1. run: claude /login\033[0m")
                    print(f"\033[93m  2. then: python GNN_Main.py -o {task} {config_file_} start={iteration}\033[0m")
                    print(f"\033[91m{'='*60}\033[0m")
                    raise SystemExit(1)

                # save Claude's terminal output to reasoning log (separate from analysis.md)
                reasoning_log_path = analysis_path.replace('_analysis.md', '_reasoning.log')
                if output_text.strip():
                    with open(reasoning_log_path, 'a') as f:
                        f.write(f"\n{'='*60}\n")
                        f.write(f"=== Iteration {iteration} ===\n")
                        f.write(f"{'='*60}\n")
                        f.write(output_text.strip())
                        f.write("\n\n")

                # git tracking: commit any code modifications made by Claude (only if code changes enabled)
                if code_changes_enabled:
                    if is_git_repo(root_dir):
                        print("\n\033[96mchecking for code modifications to commit\033[0m")
                        git_results = track_code_modifications(
                            root_dir=root_dir,
                            iteration=iteration,
                            analysis_path=analysis_path,
                            reasoning_path=reasoning_log_path
                        )

                        if git_results:
                            for file_path, success, message in git_results:
                                if success:
                                    print(f"\033[92m  git: {message}\033[0m")
                                    # set flag so next iteration uses subprocess
                                    code_modified_by_claude = True
                                else:
                                    print(f"\033[93m  git: {message}\033[0m")
                        else:
                            print("\033[90m  no code modifications detected\033[0m")
                    else:
                        # not a git repo - check for code modifications directly
                        tracked_code_files = ['src/MetabolismGraph/models/trainer.py']
                        modified_files = get_modified_code_files(root_dir, tracked_code_files)
                        if modified_files:
                            code_modified_by_claude = True
                            print(f"\033[93m  code modified (no git): {modified_files}\033[0m")
                        if iteration == 1:
                            print("\033[90m  not a git repository - code modifications will not be version controlled\033[0m")

                # save instruction file at first iteration of each block
                if iter_in_block == 1:
                    dst_instruction = f"{protocol_save_dir}/block_{block_number:03d}.md"
                    if os.path.exists(instruction_path):
                        shutil.copy2(instruction_path, dst_instruction)

                # save memory file at end of each block (after Claude updates it)
                if is_block_end:
                    memory_save_dir = f"{exploration_dir}/memory"
                    os.makedirs(memory_save_dir, exist_ok=True)
                    dst_memory = f"{memory_save_dir}/block_{block_number:03d}_memory.md"
                    if os.path.exists(memory_path):
                        shutil.copy2(memory_path, dst_memory)
                        print(f"\033[92msaved memory snapshot: {dst_memory}\033[0m")

                # recompute UCB scores after Claude to pick up mutations from analysis markdown
                compute_ucb_scores(analysis_path, ucb_path, c=ucb_c,
                                   current_log_path=analysis_log_path,
                                   current_iteration=iteration,
                                   block_size=n_iter_block,
                                   config_file=config_file)

                # generate UCB tree visualization from ucb_scores.txt
                # for block 1: save every iteration; for block 2+: save only at block end
                should_save_tree = (block_number == 1) or is_block_end
                if should_save_tree:
                    ucb_tree_path = f"{tree_save_dir}/ucb_tree_iter_{iteration:03d}.png"
                    nodes = parse_ucb_scores(ucb_path)
                else:
                    nodes = None  # skip tree generation for intermediate iterations in blocks > 0
                if nodes:
                    # get simulation info from config for tree annotation
                    sim_info = f"n_metabolites={config.simulation.n_metabolites}, n_reactions={config.simulation.n_reactions}"
                    sim_info += f", n_frames={config.simulation.n_frames}"
                    sim_info += f", max_per_rxn={config.simulation.max_metabolites_per_reaction}"

                    plot_ucb_tree(nodes, ucb_tree_path,
                                  title=f"UCB Tree - Iter {iteration}",
                                  simulation_info=sim_info)
