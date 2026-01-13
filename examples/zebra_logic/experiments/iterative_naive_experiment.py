from typing import cast

import folio_experiments as fe

import delphyne as dp

configs_iterative = [
    fe.IterativeNaiveConfig(
        bench_id=bench_id,
        model_name="gpt-5-nano",
        max_restarts=2,
        max_requests_per_attempt=requests,
        max_retries_per_sentence=2,
        max_rounds_per_retry_of_sentence=rounds,
        reasoning_effort=cast(dp.ReasoningEffort, reasoning_effort),
        temperature=None,
        max_dollar_budget=0.01,
        seed=0,
    )
    for bench_id in [
        id for (n, id) in enumerate(fe.BENCHS.keys()) if n > 3 and n < 15
    ]
    for requests, rounds, reasoning_effort in [
        # (10, 2, "low"),
        (15, 3, "minimal"),
    ]
]


if __name__ == "__main__":
    dp.Experiment(
        config_class=fe.IterativeNaiveConfig,
        context=dp.workspace_execution_context(__file__),
        configs=configs_iterative,
        output_dir=f"experiments/output/{dp.path_stem(__file__)}",
    ).run_cli()
