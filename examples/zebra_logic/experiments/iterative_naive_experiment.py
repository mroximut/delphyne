from typing import cast

import folio_experiments as fe

import delphyne as dp

configs_iterative = [
    fe.IterativeNaiveConfig(
        bench_id=bench_id,
        model_name="gpt-5-nano",
        max_restarts=2,
        max_requests_per_attempt=requests,
        max_retries_per_sentence=retries,
        max_rounds_per_retry_of_sentence=rounds,
        reasoning_effort=cast(dp.ReasoningEffort, reasoning_effort),
        temperature=None,
        max_dollar_budget=0.01,
        seed=0,
    )
    for bench_id in [id for (n, id) in enumerate(fe.BENCHS.keys()) if n < 25]
    for requests, rounds, retries, reasoning_effort in [
        (20, 2, 2, "low"),
        # (20, 2, 2, "medium"),
        # (15, 3, "minimal"),
    ]
]


if __name__ == "__main__":
    dp.Experiment(
        config_class=fe.IterativeNaiveConfig,
        context=dp.workspace_execution_context(__file__),
        configs=configs_iterative,
        output_dir=f"experiments/output_20jan/{dp.path_stem(__file__)}",
    ).run_cli()
