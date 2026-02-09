from typing import cast

import folio_experiments as fe

import delphyne as dp

configs_aggregate = [
    fe.AggregateConfig(
        bench_id=bench_id,
        model_name="gpt-5-nano",
        max_rounds_each=max_rounds_each,
        reasoning_effort=cast(dp.ReasoningEffort, reasoning_effort),
        max_dollar_budget=0.01,
        seed=0,
    )
    for bench_id in [
        id
        for (n, id) in enumerate(fe.BENCHS.keys())
        if n in fe.sample(len(fe.BENCHS), 50, seed=42)
        if fe.BENCHS[id][1] is not None
    ]
    for max_rounds_each, reasoning_effort in [
        # (10, "low", False),
        (5, "low"),
        # (10, "medium"),
        # (10, "minimal")
    ]
]


if __name__ == "__main__":
    dp.Experiment(
        config_class=fe.AggregateConfig,
        context=dp.workspace_execution_context(__file__),
        configs=configs_aggregate,
        output_dir=f"experiments/output_27jan_reflect/{dp.path_stem(__file__)}",
    ).run_cli()
