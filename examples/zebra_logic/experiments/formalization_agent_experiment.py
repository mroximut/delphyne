from typing import cast

import folio_experiments as fe

import delphyne as dp

configs_formalization_agent = [
    fe.FormalizationAgentConfig(
        bench_id=bench_id,
        model_name="gpt-5-nano",
        reasoning_effort=cast(dp.ReasoningEffort, reasoning_effort),
        temperature=None,
        max_dollar_budget=budget,
        seed=seed,
        num_requests=reqs,
        api_type="responses",
    )
    for bench_id in fe.VALIDATION_IDS
    for reasoning_effort, reqs, budget in [
        # ("low", 10, 0.01),
        ("medium", 10, 0.02),
    ]
    for seed in range(1)
]


if __name__ == "__main__":
    dp.Experiment(
        config_class=fe.FormalizationAgentConfig,
        context=dp.workspace_execution_context(__file__),
        configs=configs_formalization_agent,
        output_dir=f"experiments/output_16_may/{dp.path_stem(__file__)}",
    ).run_cli()
