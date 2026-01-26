from itertools import islice
from pathlib import Path
from typing import Any

import folio_experiments as fe
import pandas as pd
import yaml

from delphyne.stdlib.experiments.experiment_launcher import (
    CONFIGS_SUBDIR,
    EXPERIMENT_STATE_FILE,
    RESULT_FILE,
    RESULTS_SUMMARY,
)

ONEHOT = "output_20jan/oneshot_experiment"
NAIVE = "output_20jan/iterative_naive_experiment"
BLACKLIST = "output_20jan/iterative_blacklist_experiment"


def process_results(
    experiment_dir: str, previous_merged_df_path: Path | None = None
):
    results: list[dict[str, Any]] = []
    path_prefix = Path(__file__).resolve().parent / experiment_dir
    results_summary = pd.read_csv(path_prefix / RESULTS_SUMMARY)  # type: ignore
    with open(path_prefix / EXPERIMENT_STATE_FILE, "r") as f:
        experiment_yaml = yaml.safe_load(f)  # type: ignore
    configs_dir = path_prefix / CONFIGS_SUBDIR
    benchs = fe.load_folio_benchmark()
    configs = experiment_yaml.get("configs")

    previous_merged_df = (
        pd.read_csv(previous_merged_df_path)  # type: ignore
        if previous_merged_df_path
        else None
    )

    for hsh, cfg in configs.items():
        if (
            previous_merged_df is not None
            and hsh in previous_merged_df["config_hash"].values
        ):
            continue

        bench_id = cfg.get("params").get("bench_id")
        with open(configs_dir / hsh / RESULT_FILE, "r") as f:
            head = "".join(islice(f, 100))  # first 100 lines
            result_yaml = yaml.safe_load(head)

        result = result_yaml.get("outcome").get("result").get("values")
        result = result[0] if result else None
        ground_truth = benchs[bench_id][1]
        results.append(
            {
                "bench_id": bench_id,
                "result": result,
                "ground_truth": ground_truth,
                "correct": result == ground_truth,
                "config_hash": hsh,
            }
        )
    results_df = pd.DataFrame(results)
    merged_df = results_summary.merge(
        results_df, left_index=True, right_index=True
    )
    if previous_merged_df is not None:
        merged_df = pd.concat([previous_merged_df, merged_df])
    merged_df.to_csv(path_prefix / "merged_results.csv", index=False)  # type: ignore


def get_correctness_ratio(
    merged_df_path: Path, effort: str
) -> tuple[int, int]:
    merged_df = pd.read_csv(merged_df_path)  # type: ignore
    df_effort = merged_df[merged_df["reasoning_effort"] == effort]
    correct = df_effort["correct"].sum()
    total = len(df_effort)
    return int(correct), int(total)


def merged_df_path(experiment_dir: str) -> Path:
    path_prefix = Path(__file__).resolve().parent / experiment_dir
    return path_prefix / "merged_results.csv"


if __name__ == "__main__":
    process_results(ONEHOT)
    process_results(NAIVE)
    process_results(BLACKLIST)

    for effort_level in ["low", "medium"]:
        ratio_onehot = get_correctness_ratio(
            merged_df_path(ONEHOT), effort_level
        )
        ratio_naive = get_correctness_ratio(
            merged_df_path(NAIVE), effort_level
        )
        ratio_blacklist = get_correctness_ratio(
            merged_df_path(BLACKLIST), effort_level
        )
        print(
            f"Effort: {effort_level} | "
            f"One-shot: {ratio_onehot[0]}/{ratio_onehot[1]} | "
            f"Iterative Naive: {ratio_naive[0]}/{ratio_naive[1]} | "
            f"Iterative Blacklist: {ratio_blacklist[0]}/{ratio_blacklist[1]}"
        )
