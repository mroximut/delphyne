from pathlib import Path
from typing import Any, Literal, Sequence

import folio_experiments as fe
import pandas as pd
import yaml

from delphyne.stdlib.experiments.experiment_launcher import (
    CONFIGS_SUBDIR,
    EXPERIMENT_STATE_FILE,
    RESULT_FILE,
    RESULTS_SUMMARY,
)

# ONESHOT = "output_27jan/oneshot_experiment"
# NAIVE = "output_20jan/iterative_naive_experiment"
# BLACKLIST = "output_27jan/iterative_blacklist_experiment"
# ONESHOT_REFLECT = "output_27jan_reflect/oneshot_experiment"
BLACKLIST_REFLECT = "output_9feb/iterative_blacklist_experiment"
AGGREGATE = "output_9feb/aggregate_experiment"

ONLY_ASK = "output_3mar/only_ask_experiment"
FORMALIZATION_AGENT = "output_3mar/formalization_agent_experiment"
Z3_AGENT = "output_3mar/z3_agent_experiment"


def process_output_aggregate(
    values: tuple[
        bool | None,
        Sequence[tuple[list[str], tuple[bool | None, bool | None]]],
    ],
    sequence_type: Literal["mixed", "all_normal_reflect"],
    aggregation_type: Literal["majority_vote", "favor_unsat"],
    reflection_type: Literal[
        "mixed", "always", "never", "only_if_sat", "only_if_unsat"
    ],
) -> bool | None:
    types = (aggregation_type, sequence_type, reflection_type)
    if types in [
        ("majority_vote", "mixed", "mixed"),
        ("majority_vote", "all_normal_reflect", "always"),
    ]:
        return values[0]
    if types in [
        ("favor_unsat", "mixed", "mixed"),
        ("favor_unsat", "all_normal_reflect", "always"),
    ]:
        return (
            any(sol for _, (_, sol) in values[1])
            if len(values[1]) > 0
            else None
        )
    if types in [
        ("majority_vote", "mixed", "never"),
        ("majority_vote", "all_normal_reflect", "never"),
    ]:
        results = [sol for _, (sol, _) in values[1] if sol is not None]
        return (
            sum(1 for r in results if r) >= len(results) / 2
            if len(results) > 0
            else None
        )
    if types in [
        ("favor_unsat", "mixed", "never"),
        ("favor_unsat", "all_normal_reflect", "never"),
    ]:
        return (
            any(sol for _, (sol, _) in values[1] if sol is not None)
            if len(values[1]) > 0
            else None
        )
    if reflection_type == "only_if_sat":
        reflects = [
            reflect
            for _, (sol, reflect) in values[1]
            if sol is False and reflect is not None
        ]
        prelims = [sol for _, (sol, _) in values[1] if sol is True]
        results = prelims + reflects
        if types in [
            ("majority_vote", "mixed", "only_if_sat"),
            ("majority_vote", "all_normal_reflect", "only_if_sat"),
        ]:
            return (
                sum(1 for r in results if r) >= len(results) / 2
                if len(results) > 0
                else None
            )
        if types in [
            ("favor_unsat", "mixed", "only_if_sat"),
            ("favor_unsat", "all_normal_reflect", "only_if_sat"),
        ]:
            return any(r for r in results) if len(results) > 0 else None


def process_output_for_oneshot(
    values: tuple[
        bool | None,
        Sequence[tuple[list[str], tuple[bool | None, bool | None]]],
    ],
    model_type: Literal["literal", "normal", "implicitly"],
    reflection_type: Literal["always", "never", "only_if_sat"],
) -> bool | None:
    match (model_type, reflection_type):
        case ("literal", "always"):
            res = [sol for li, (_, sol) in values[1] if li[1] == "literally"]
            return res[0] if len(res) == 1 else None
        case ("literal", "never"):
            res = [sol for li, (sol, _) in values[1] if li[1] == "literally"]
            return res[0] if len(res) == 1 else None
        case ("implicitly", "always"):
            res = [sol for li, (_, sol) in values[1] if li[1] == "implicitly"]
            return res[0] if len(res) == 1 else None
        case ("implicitly", "never"):
            res = [sol for li, (sol, _) in values[1] if li[1] == "implicitly"]
            return res[0] if len(res) == 1 else None
        case ("normal", "always"):
            res = [sol for li, (_, sol) in values[1] if li[1] == "normal"]
            return res[0] if len(res) >= 1 else None
        case ("normal", "only_if_sat"):
            res = [
                (sol1, sol2)
                for li, (sol1, sol2) in values[1]
                if li[1] == "normal"
            ]
            sol1 = res[0][0] if len(res) >= 1 else None
            sol2 = res[0][1] if len(res) >= 1 else None
            return sol1 if sol1 is True else sol2 if sol2 is not None else None
        case ("normal", "never"):
            res = [sol for li, (sol, _) in values[1] if li[1] == "normal"]
            return res[0] if len(res) >= 1 else None
        case _:
            pass


def process_results(
    experiment_dir: str,
    strategy_type: Literal[
        "aggregate",
        "blacklist",
        "oneshot",
        "formalization_agent",
        "z3_agent",
        "only_ask",
    ],
    save_name: str = "merged_results.csv",
    sequence_type: str = "",
    aggregation_type: str = "",
    reflection_type: str = "",
    oneshot_model_type: str = "",
    oneshot_reflect_type: str = "",
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    path_prefix = Path(__file__).resolve().parent / experiment_dir
    results_summary = pd.read_csv(path_prefix / RESULTS_SUMMARY)  # type: ignore
    with open(path_prefix / EXPERIMENT_STATE_FILE, "r") as f:
        experiment_yaml = yaml.safe_load(f)  # type: ignore
    configs_dir = path_prefix / CONFIGS_SUBDIR
    benchs = fe.load_folio_benchmark()
    configs = experiment_yaml.get("configs")

    for hsh, cfg in configs.items():
        if cfg.get("status") != "done":
            continue

        bench_id = cfg.get("params").get("bench_id")
        with open(configs_dir / hsh / RESULT_FILE, "r") as f:
            lines: list[str] = []
            for line in f:
                if line.strip().startswith("spent_budget:"):
                    break
                lines.append(line)
            head = "".join(lines)
            result_yaml = yaml.safe_load(head)

        result = result_yaml.get("outcome").get("result").get("values")
        if len(result) == 0:
            result = None
        else:
            result = result[0]

            if strategy_type == "aggregate":
                result = process_output_aggregate(
                    values=result,
                    sequence_type=sequence_type,  # type: ignore
                    aggregation_type=aggregation_type,  # type: ignore
                    reflection_type=reflection_type,  # type: ignore
                )
            elif strategy_type == "oneshot":
                result = process_output_for_oneshot(
                    values=result,
                    model_type=oneshot_model_type,  # type: ignore
                    reflection_type=oneshot_reflect_type,  # type: ignore
                )
            else:
                result = result

        ground_truth = benchs[bench_id][1]
        results.append(
            {
                "bench_id": bench_id,
                "result": result,
                "ground_truth": ground_truth,
                "correct": None
                if ground_truth not in [True, False]
                else result == ground_truth,
                "config_hash": hsh,
            }
        )
    results_df = pd.DataFrame(results)
    merged_df = results_summary.merge(
        results_df, left_index=True, right_index=True
    )

    if strategy_type == "aggregate" or strategy_type == "oneshot":
        merged_df = merged_df[
            (
                merged_df["sequence_type"].isnull()
                if sequence_type == "mixed"
                else (merged_df["sequence_type"] == sequence_type)
            )
        ]
    correct = merged_df["correct"].sum()
    total = len(merged_df["ground_truth"].dropna())
    print(
        f"Correct: {correct}/{total} ({correct / total:.2%}) for {experiment_dir}, "
        f"strategy: {strategy_type}"
        + (
            f", sequence_type: {sequence_type}, aggregation_type: {aggregation_type}, reflection_type: {reflection_type}"
            if strategy_type == "aggregate"
            else f", model_type: {oneshot_model_type}, reflect: {oneshot_reflect_type}"  # type:ignore
            if strategy_type == "oneshot"
            else ""
        )
    )

    merged_df.to_csv(path_prefix / save_name, index=False)  # type: ignore

    return {
        "correct": correct,
        "total": total,
        "strategy_type": strategy_type,
        "sequence_type": sequence_type,
        "aggregation_type": aggregation_type,
        "reflection_type": reflection_type,
        "oneshot_model_type": oneshot_model_type,
        "oneshot_reflect_type": oneshot_reflect_type,
    }


def get_correctness_ratio(
    merged_df_path: Path, effort: str, reflect_if_sat: bool = False
) -> tuple[int, int]:
    merged_df = pd.read_csv(merged_df_path)  # type: ignore
    df_effort = merged_df[merged_df["reasoning_effort"] == effort]
    if "reflect_if_sat" in df_effort.columns:
        df_effort = df_effort[df_effort["reflect_if_sat"] == reflect_if_sat]
    if df_effort.empty and not reflect_if_sat:
        df_effort = merged_df[merged_df["reasoning_effort"] == effort]
        df_effort = df_effort[df_effort["reflect_if_sat"].isnull()]
    correct = df_effort["correct"].sum()
    total = len(df_effort["ground_truth"].dropna())
    return int(correct), int(total)


def merged_df_path(experiment_dir: str) -> Path:
    path_prefix = Path(__file__).resolve().parent / experiment_dir
    return path_prefix / "merged_results.csv"


def main_aggregate():
    oneshot_dicts = [
        process_results(
            BLACKLIST_REFLECT,
            "blacklist",
            oneshot_model_type="iterative_blacklist",
            oneshot_reflect_type="only_if_sat",
        )
    ]

    aggregate_dicts = [
        process_results(
            AGGREGATE,
            "aggregate",
            sequence_type=sequence_type,
            aggregation_type=aggregation_type,
            reflection_type=reflection_type,
            save_name=f"merged_results_{sequence_type}_{aggregation_type}_{reflection_type}.csv",
        )
        for sequence_type in ["mixed", "all_normal_reflect"]
        for aggregation_type in ["majority_vote", "favor_unsat"]
        for reflection_type in [
            ("mixed" if sequence_type == "mixed" else "always"),
            "never",
            "only_if_sat",
        ]
    ]

    oneshot_dicts += [
        process_results(
            AGGREGATE,
            "oneshot",
            sequence_type="mixed",
            reflection_type="mixed",
            oneshot_model_type=model_type,
            oneshot_reflect_type=reflect,
            save_name=f"merged_results_oneshot_{model_type}_{reflect}.csv",
        )
        for model_type in ["literal", "implicitly"]
        for reflect in ["always", "never"]
    ]

    oneshot_dicts += [
        process_results(
            AGGREGATE,
            "oneshot",
            sequence_type="all_normal_reflect",
            reflection_type="always",
            oneshot_model_type="normal",
            oneshot_reflect_type=reflect,
            save_name=f"merged_results_oneshot_normal_{reflect}.csv",
        )
        for reflect in ["always", "only_if_sat", "never"]
    ]

    pd.DataFrame(aggregate_dicts).to_csv(
        Path(__file__).resolve().parent
        / "output_9feb"
        / "aggregate_summary.csv",
        index=False,
    )
    pd.DataFrame(oneshot_dicts).to_csv(
        Path(__file__).resolve().parent
        / "output_9feb"
        / "oneshot_summary.csv",
        index=False,
    )


def main_agents():
    agent_dicts = [
        process_results(
            ONLY_ASK,
            "only_ask",
            save_name="merged_results_only_ask.csv",
        ),
        process_results(
            FORMALIZATION_AGENT,
            "formalization_agent",
            save_name="merged_results_formalization_agent.csv",
        ),
        process_results(
            Z3_AGENT,
            "z3_agent",
            save_name="merged_results_z3_agent.csv",
        ),
    ]

    pd.DataFrame(agent_dicts).to_csv(
        Path(__file__).resolve().parent / "output_3mar" / "agents_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    # main_aggregate()
    main_agents()
