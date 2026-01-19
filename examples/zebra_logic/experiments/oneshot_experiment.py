from typing import cast

import folio_experiments as fe

import delphyne as dp

configs_oneshot = [
    fe.OneshotConfig(
        bench_id=bench_id,
        model_name="gpt-5-nano",
        max_rounds=max_rounds,
        reasoning_effort=cast(dp.ReasoningEffort, reasoning_effort),
        temperature=None,
        max_dollar_budget=0.01,
        seed=0,
    )
    for bench_id in [
        id for (n, id) in enumerate(fe.BENCHS.keys()) if n > 3 and n < 15
    ]
    for max_rounds, reasoning_effort in [(5, "low"), (10, "minimal")]
]


if __name__ == "__main__":
    dp.Experiment(
        config_class=fe.OneshotConfig,
        context=dp.workspace_execution_context(__file__),
        configs=configs_oneshot,
        output_dir=f"experiments/output/{dp.path_stem(__file__)}",
    ).run_cli()
