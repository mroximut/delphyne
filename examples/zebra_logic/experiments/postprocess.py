from pathlib import Path
from typing import Any, Literal, Sequence

import folio_experiments as fe
import pandas as pd
import yaml
from pydantic import BaseModel

from delphyne.stdlib.experiments.experiment_launcher import (
    CONFIGS_SUBDIR,
    EXPERIMENT_STATE_FILE,
    RESULT_FILE,
    RESULTS_SUMMARY,
)


class StrFormalization(BaseModel):
    predicates: list[str] | None = None
    constants: list[str] | None = None
    constraints: list[str] | None = None
    conclusion: list[str] | None = None


class RefinedFormalization(BaseModel):
    str_formalization: StrFormalization
    reason: str | None


class Verdict(BaseModel):
    reflection_flag: Literal["always", "never", "only_if_sat", "only_if_unsat"]
    style_flag: Literal["normal", "literally", "implicitly"]
    formalizations: list[StrFormalization]
    refined_formalizations: list[RefinedFormalization] | None
    first_solution: bool | None
    final_solution: bool | None
    judgement_solution: bool | None = None


def _any(
    verdicts: Sequence[Verdict], sol: Literal["final", "first"]
) -> bool | None:
    if sol == "final":
        return (
            any(v.final_solution for v in verdicts)
            if len(verdicts) > 0
            else None
        )
    elif sol == "first":
        return (
            any(v.first_solution for v in verdicts)
            if len(verdicts) > 0
            else None
        )


def _maj(
    verdicts: Sequence[Verdict], sol: Literal["final", "first"]
) -> bool | None:
    if sol == "final":
        return (
            sum(1 for v in verdicts if v.final_solution) >= len(verdicts) / 2
            if len(verdicts) > 0
            else None
        )
    elif sol == "first":
        return (
            sum(1 for v in verdicts if v.first_solution) >= len(verdicts) / 2
            if len(verdicts) > 0
            else None
        )


def process_output_aggregate(
    returned: tuple[bool | None, Sequence[Verdict]],
    sequence_type: Literal["mixed", "all_normal", "all_normal_reflect"],
    aggregation_type: Literal["majority_vote", "favor_unsat", "judge"],
    reflection_type: Literal[
        "mixed", "always", "never", "only_if_sat", "only_if_unsat"
    ],
) -> bool | None:
    strategy_result, verdicts = returned
    types = (aggregation_type, sequence_type, reflection_type)
    if aggregation_type == "judge" and reflection_type == "always":
        return strategy_result
    if types in [
        ("majority_vote", "mixed", "mixed"),
        ("majority_vote", "all_normal", "always"),
        ("majority_vote", "all_normal_reflect", "always"),
    ]:
        return _maj(verdicts, "final")  # res
    if types in [
        ("favor_unsat", "mixed", "mixed"),
        ("favor_unsat", "all_normal", "always"),
        ("favor_unsat", "all_normal_reflect", "always"),
    ]:
        return _any(verdicts, "final")
    if types in [
        ("majority_vote", "mixed", "never"),
        ("majority_vote", "all_normal", "never"),
        ("majority_vote", "all_normal_reflect", "never"),
    ]:
        return _maj(verdicts, "first")
    if types in [
        ("favor_unsat", "mixed", "never"),
        ("favor_unsat", "all_normal", "never"),
        ("favor_unsat", "all_normal_reflect", "never"),
    ]:
        return _any(verdicts, "first")
    if reflection_type == "only_if_sat":
        reflects = [
            reflect
            for (sol, reflect) in [
                (v.first_solution, v.final_solution) for v in verdicts
            ]
            if sol is False and reflect is not None
        ]
        prelims = [
            v.first_solution for v in verdicts if v.first_solution is True
        ]
        results = prelims + reflects
        if types in [
            ("majority_vote", "mixed", "only_if_sat"),
            ("majority_vote", "all_normal", "only_if_sat"),
            ("majority_vote", "all_normal_reflect", "only_if_sat"),
        ]:
            return (
                sum(1 for r in results if r) >= len(results) / 2
                if len(results) > 0
                else None
            )
        if types in [
            ("favor_unsat", "mixed", "only_if_sat"),
            ("favor_unsat", "all_normal", "only_if_sat"),
            ("favor_unsat", "all_normal_reflect", "only_if_sat"),
        ]:
            return any(r for r in results) if len(results) > 0 else None
    raise ValueError(
        "Invalid combination: "
        + f"{sequence_type}, {aggregation_type}, {reflection_type}"
    )


def process_output_for_oneshot(
    returned: tuple[bool | None, Sequence[Verdict]],
    model_type: Literal["literal", "normal", "implicitly"],
    reflection_type: Literal["always", "never", "only_if_sat"],
) -> bool | None:
    _, vs = returned
    match (model_type, reflection_type):
        case ("literal", "always"):
            res = [v.final_solution for v in vs if v.style_flag == "literally"]
            return res[0] if len(res) == 1 else None
        case ("literal", "never"):
            res = [v.first_solution for v in vs if v.style_flag == "literally"]
            return res[0] if len(res) == 1 else None
        case ("implicitly", "always"):
            res = [
                v.final_solution for v in vs if v.style_flag == "implicitly"
            ]
            return res[0] if len(res) == 1 else None
        case ("implicitly", "never"):
            res = [
                v.first_solution for v in vs if v.style_flag == "implicitly"
            ]
            return res[0] if len(res) == 1 else None
        case ("normal", "always"):
            res = [v.final_solution for v in vs if v.style_flag == "normal"]
            return res[0] if len(res) >= 1 else None
        case ("normal", "only_if_sat"):
            res = [
                (v.first_solution, v.final_solution)
                for v in vs
                if v.style_flag == "normal"
            ]
            sol1 = res[0][0] if len(res) >= 1 else None
            sol2 = res[0][1] if len(res) >= 1 else None
            return sol1 if sol1 is True else sol2 if sol2 is not None else None
        case ("normal", "never"):
            res = [v.first_solution for v in vs if v.style_flag == "normal"]
            return res[0] if len(res) >= 1 else None
        case _:
            raise ValueError(
                "Invalid combination of model_type and "
                + f"reflection_type: {model_type}, {reflection_type}"
            )


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
    sequence_type: str | None = None,
    aggregation_type: str | None = None,
    reflection_type: str | None = None,
    oneshot_model_type: str | None = None,
    oneshot_reflect_type: str | None = None,
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
                    returned=(result[0], [Verdict(**v) for v in result[1]]),
                    sequence_type=sequence_type,  # type: ignore
                    aggregation_type=aggregation_type,  # type: ignore
                    reflection_type=reflection_type,  # type: ignore
                )
            elif strategy_type == "oneshot":
                result = process_output_for_oneshot(
                    returned=(result[0], [Verdict(**v) for v in result[1]]),
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
    pct = f"{correct / total:.2%}" if total else "n/a"
    print(
        f"Correct: {correct}/{total} ({pct}) for "
        f"{experiment_dir}, strategy: {strategy_type}"
        + (
            f", sequence_type: {sequence_type}, aggregation_type:"
            f"{aggregation_type}, reflection_type: {reflection_type}"
            if strategy_type == "aggregate"
            else f", model_type: {oneshot_model_type}, "
            f" reflect: {oneshot_reflect_type}"
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


def main_aggregate(experiment_dir: str):
    oneshot_dicts: list[dict[str, Any]] = []
    aggregate = experiment_dir + "/aggregate_experiment"
    blacklist = experiment_dir + "/iterative_experiment"

    sequence_types = [
        "all_normal_reflect",
    ]

    aggregate_dicts = [
        process_results(
            aggregate,
            "aggregate",
            sequence_type=sequence_type,
            aggregation_type=aggregation_type,
            reflection_type=reflection_type,
            save_name=f"merged_results_{sequence_type}_{aggregation_type}_{reflection_type}.csv",
        )
        for sequence_type in sequence_types
        for aggregation_type in ["majority_vote", "favor_unsat", "judge"]
        for reflection_type in (
            [
                ("always" if sequence_type != "mixed" else "mixed"),
                "never",
                "only_if_sat",
            ]
            if aggregation_type != "judge"
            else ["always"]
        )
    ]

    # oneshot_dicts += [
    #     process_results(
    #         aggregate,
    #         "oneshot",
    #         sequence_type="mixed",
    #         reflection_type="mixed",
    #         oneshot_model_type=model_type,
    #         oneshot_reflect_type=reflect,
    #         save_name=f"merged_results_oneshot_{model_type}_{reflect}.csv",
    #     )
    #     for model_type in ["literal", "implicitly"]
    #     for reflect in ["always", "never"]
    # ]

    oneshot_dicts += [
        process_results(
            aggregate,
            "oneshot",
            sequence_type=sequence_type,
            reflection_type=(
                "always" if sequence_type != "mixed" else "mixed"
            ),
            oneshot_model_type="normal",
            oneshot_reflect_type=reflect,
            save_name=(
                f"merged_results_oneshot_{sequence_type}_normal_{reflect}.csv"
            ),
        )
        for sequence_type in sequence_types
        for reflect in ["always", "only_if_sat", "never"]
    ]

    oneshot_dicts += [
        process_results(
            blacklist,
            "blacklist",
            oneshot_model_type="iterative_blacklist",
            oneshot_reflect_type="only_if_sat",
        )
    ]

    pd.DataFrame(aggregate_dicts).to_csv(
        Path(__file__).resolve().parent
        / experiment_dir
        / "aggregate_summary.csv",
        index=False,
    )
    pd.DataFrame(oneshot_dicts).to_csv(
        Path(__file__).resolve().parent
        / experiment_dir
        / "oneshot_summary.csv",
        index=False,
    )


def main_agents(experiment_dir: str):
    only_ask = experiment_dir + "/only_ask_experiment"
    formalization_agent = experiment_dir + "/formalization_agent_experiment"
    z3_agent = experiment_dir + "/z3_agent_experiment"
    agent_dicts = [
        process_results(
            only_ask,
            "only_ask",
            save_name="merged_results_only_ask.csv",
        ),
        process_results(
            formalization_agent,
            "formalization_agent",
            save_name="merged_results_formalization_agent.csv",
        ),
        process_results(
            z3_agent,
            "z3_agent",
            save_name="merged_results_z3_agent.csv",
        ),
    ]

    pd.DataFrame(agent_dicts).to_csv(
        Path(__file__).resolve().parent
        / experiment_dir
        / "agents_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main_aggregate("output_16_may_low")
    # main_agents("output_11may")
    pass
