from typing import cast

import folio_experiments as fe

import delphyne as dp

configs_iterative = [
    fe.IterativeConfig(
        bench_id=bench_id,
        model_name="gpt-5-nano",
        max_restarts=2,
        max_requests_per_attempt=requests,
        max_retries_per_chunk=retries,
        number_of_chunks=2,
        majority_vote_size=3,
        max_depth_for_formalize=5,
        reflect_if_sat=reflect,
        reasoning_effort=cast(dp.ReasoningEffort, reasoning_effort),
        temperature=None,
        max_dollar_budget=budget,
        seed=seed,
        api_type="chat_completions",
    )
    for bench_id in fe.VALIDATION_IDS
    for requests, retries, reasoning_effort, budget, reflect in [
        (30, 3, "low", 0.01, True),
        # (30, 3, "medium", 0.02, True),
    ]
    for seed in range(3)
]


if __name__ == "__main__":
    dp.Experiment(
        config_class=fe.IterativeConfig,
        context=dp.workspace_execution_context(__file__),
        configs=configs_iterative,
        output_dir=f"experiments/output_custom/{dp.path_stem(__file__)}",
    ).run_cli()
