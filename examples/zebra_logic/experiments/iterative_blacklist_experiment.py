from typing import cast

import folio_experiments as fe

import delphyne as dp

configs_iterative = [
    fe.IterativeBlacklistConfig(
        bench_id=bench_id,
        model_name="gpt-5-nano",
        max_restarts=2,
        max_requests_per_attempt=requests,
        max_retries_per_sentence=retries,
        max_depth_for_reflect=5,
        reflect_if_sat=reflect,
        reasoning_effort=cast(dp.ReasoningEffort, reasoning_effort),
        temperature=None,
        max_dollar_budget=0.01,
        seed=0,
    )
    for bench_id in [
        id
        for (n, id) in enumerate(fe.BENCHS.keys())
        if n in fe.sample(len(fe.BENCHS), 50, seed=42)
    ]
    for requests, retries, reasoning_effort, reflect in [
        # (20, 3, "minimal"),
        # (30, 3, "low", False),
        (30, 3, "low", True),
        # (20, 3, "medium"),
        # (15, 3, "minimal"),
    ]
]


if __name__ == "__main__":
    dp.Experiment(
        config_class=fe.IterativeBlacklistConfig,
        context=dp.workspace_execution_context(__file__),
        configs=configs_iterative,
        output_dir=f"experiments/output_27jan_reflect/{dp.path_stem(__file__)}",
    ).run_cli()
