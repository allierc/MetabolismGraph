#!/usr/bin/env python3
"""
git tracking for code modifications in Claude experiments.

automatically commits code changes made by Claude during experimentation,
providing version control and rollback capability.
"""

import subprocess
import os
import re
from typing import Optional, List, Tuple


def is_git_repo(root_dir: str) -> bool:
    """check if directory is a git repository."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_modified_code_files(root_dir: str, tracked_files: List[str]) -> List[str]:
    """
    check which tracked code files have been modified.

    Args:
        root_dir: root directory of the repository
        tracked_files: list of file paths to check (relative to root_dir)

    Returns:
        list of modified file paths
    """
    modified = []

    for file_path in tracked_files:
        try:
            result = subprocess.run(
                ['git', 'diff', '--quiet', file_path],
                cwd=root_dir,
                capture_output=True,
                timeout=5
            )

            if result.returncode != 0:
                modified.append(file_path)

        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            pass

    return modified


def extract_code_modification_description(
    analysis_path: str,
    reasoning_path: str,
    iteration: int
) -> Optional[str]:
    """
    extract code modification description from analysis/reasoning logs.
    """
    # try reasoning log first
    try:
        with open(reasoning_path, 'r') as f:
            content = f.read()

            pattern = r'CODE MODIFICATION:\s*\n\s*File:\s*([^\n]+)\s*\n\s*Function:\s*([^\n]+)\s*\n\s*Change:\s*([^\n]+)\s*\n\s*Hypothesis:\s*([^\n]+)'
            matches = list(re.finditer(pattern, content, re.MULTILINE))

            if matches:
                match = matches[-1]
                file_name = match.group(1).strip()
                function = match.group(2).strip()
                change = match.group(3).strip()
                hypothesis = match.group(4).strip()

                return f"{change} in {function} ({file_name})\nHypothesis: {hypothesis}"
    except (FileNotFoundError, IOError):
        pass

    # fallback to analysis.md
    try:
        with open(analysis_path, 'r') as f:
            content = f.read()

            iter_pattern = rf'## Iter {iteration}:.*?CODE MODIFICATION:.*?Change:\s*([^\n]+).*?Hypothesis:\s*([^\n]+)'
            match = re.search(iter_pattern, content, re.DOTALL)

            if match:
                change = match.group(1).strip()
                hypothesis = match.group(2).strip()
                return f"{change}\nHypothesis: {hypothesis}"
    except (FileNotFoundError, IOError):
        pass

    return None


def get_file_diff_summary(root_dir: str, file_path: str) -> str:
    """get a summary of changes to a file."""
    try:
        result = subprocess.run(
            ['git', 'diff', '--stat', file_path],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return "Changes detected"


def commit_code_modification(
    root_dir: str,
    file_path: str,
    iteration: int,
    description: Optional[str] = None,
    analysis_path: Optional[str] = None,
    reasoning_path: Optional[str] = None
) -> Tuple[bool, str]:
    """
    commit a code modification to git.
    """
    if not is_git_repo(root_dir):
        return False, "Not a git repository"

    if description is None and analysis_path and reasoning_path:
        description = extract_code_modification_description(
            analysis_path, reasoning_path, iteration
        )

    if description is None:
        description = f"Code modification in {os.path.basename(file_path)}"

    try:
        stage_result = subprocess.run(
            ['git', 'add', file_path],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=10
        )

        if stage_result.returncode != 0:
            return False, f"Failed to stage file: {stage_result.stderr}"

        diff_summary = get_file_diff_summary(root_dir, file_path)

        commit_msg = f"""[Iter {iteration}] {description}

File: {file_path}
Changes: {diff_summary}

[Automated commit by Claude Code Modification System]"""

        commit_result = subprocess.run(
            ['git', 'commit', '-m', commit_msg],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=10
        )

        if commit_result.returncode != 0:
            if 'nothing to commit' in commit_result.stdout.lower():
                return True, "No changes to commit"
            return False, f"Commit failed: {commit_result.stderr}"

        hash_result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=5
        )

        commit_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else "unknown"

        return True, f"Committed as {commit_hash}: {os.path.basename(file_path)}"

    except subprocess.TimeoutExpired:
        return False, "Git command timeout"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def track_code_modifications(
    root_dir: str,
    iteration: int,
    analysis_path: str,
    reasoning_path: str,
    code_files: Optional[List[str]] = None
) -> List[Tuple[str, bool, str]]:
    """
    check for and commit any code modifications.
    """
    if not is_git_repo(root_dir):
        return []

    # default tracked files for MetabolismGraph
    if code_files is None:
        code_files = [
            'src/MetabolismGraph/models/trainer.py',
        ]

    modified_files = get_modified_code_files(root_dir, code_files)

    results = []
    for file_path in modified_files:
        success, message = commit_code_modification(
            root_dir=root_dir,
            file_path=file_path,
            iteration=iteration,
            analysis_path=analysis_path,
            reasoning_path=reasoning_path
        )
        results.append((file_path, success, message))

    return results


def git_push(root_dir: str, remote: str = 'origin', branch: Optional[str] = None) -> Tuple[bool, str]:
    """push commits to remote repository."""
    if not is_git_repo(root_dir):
        return False, "Not a git repository"

    try:
        if branch is None:
            branch_result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=root_dir,
                capture_output=True,
                text=True,
                timeout=5
            )

            if branch_result.returncode != 0:
                return False, "Could not determine current branch"

            branch = branch_result.stdout.strip()

        push_result = subprocess.run(
            ['git', 'push', remote, branch],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=30
        )

        if push_result.returncode != 0:
            return False, f"Push failed: {push_result.stderr}"

        return True, f"Pushed to {remote}/{branch}"

    except subprocess.TimeoutExpired:
        return False, "Push timeout (30s)"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python git_code_tracker.py <root_dir>")
        sys.exit(1)

    root_dir = sys.argv[1]

    print(f"testing git tracker in: {root_dir}")
    print(f"is git repo: {is_git_repo(root_dir)}")

    if is_git_repo(root_dir):
        tracked_files = [
            'src/MetabolismGraph/models/trainer.py',
        ]

        modified = get_modified_code_files(root_dir, tracked_files)
        print(f"modified files: {modified}")
