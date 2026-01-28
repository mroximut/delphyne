from typing import cast

import folio_experiments as fe

import delphyne as dp

configs_oneshot = [
    fe.OneshotConfig(
        bench_id=bench_id,
        model_name="gpt-5-nano",
        max_rounds=max_rounds,
        reasoning_effort=cast(dp.ReasoningEffort, reasoning_effort),
        reflect_if_sat=reflect,
        temperature=None,
        max_dollar_budget=0.01,
        seed=0,
    )
    for bench_id in [
        id
        for (n, id) in enumerate(fe.BENCHS.keys())
        if n in fe.sample(len(fe.BENCHS), 50, seed=42)
    ]
    for max_rounds, reasoning_effort, reflect in [
        # (10, "low", False),
        (10, "low", True),
        # (10, "medium"),
        # (10, "minimal")
    ]
]


if __name__ == "__main__":
    dp.Experiment(
        config_class=fe.OneshotConfig,
        context=dp.workspace_execution_context(__file__),
        configs=configs_oneshot,
        output_dir=f"experiments/output_27jan_reflect/{dp.path_stem(__file__)}",
    ).run_cli()
