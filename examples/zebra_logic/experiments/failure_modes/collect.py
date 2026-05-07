"""
Utility to collect config folders for failed experiments into a failure_modes directory.

Given a CSV file with experiment results and a directory containing configs,
this script copies config folders for experiments where "correct" is False
to a new failure_modes directory.
"""

import csv
import shutil
from pathlib import Path


def collect_failure_modes(
    csv_file: str, experiment_dir: str, output_dir: str = None
):
    """
    Copy config folders for failed experiments to a failure_modes directory.

    Args:
        csv_file: Path to the CSV file with experiment results
        experiment_dir: Path to the directory containing the configs folder
        output_dir: Path to the output directory. If None, creates failure_modes in experiment_dir
    """
    exp_path = Path(__file__).resolve().parent.parent / experiment_dir
    csv_path = exp_path / csv_file
    configs_path = exp_path / "configs"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    if not configs_path.exists():
        raise FileNotFoundError(f"Configs directory not found: {configs_path}")

    # Default output directory
    if output_dir is None:
        output_dir = exp_path / "failure_modes" / csv_path.stem
    else:
        output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Read CSV and collect failure hashes
    failure_hashes = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check if 'correct' field exists and is False
            if "correct" in row and row["correct"].lower() == "false":
                if "config_hash" in row:
                    failure_hashes.append(row["config_hash"])

    # Copy config folders for failures
    copied_count = 0
    skipped_count = 0

    for hash_value in failure_hashes:
        source_path = configs_path / hash_value
        dest_path = output_dir / hash_value

        if source_path.exists():
            if dest_path.exists():
                print(
                    f"Skipping {hash_value} - already exists in failure_modes"
                )
                skipped_count += 1
            else:
                shutil.copytree(source_path, dest_path)
                print(f"Copied {hash_value}")
                copied_count += 1
        else:
            print(f"Warning: Config folder not found for {hash_value}")

    print("\nSummary:")
    print(f"  Total failures found: {len(failure_hashes)}")
    print(f"  Copied: {copied_count}")
    print(f"  Skipped (already exist): {skipped_count}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    csv_file = "merged_results_all_normal_reflect_majority_vote_always.csv"
    experiment_dir = "output_5may/aggregate_experiment/"
    output_dir = None

    collect_failure_modes(csv_file, experiment_dir, output_dir)
