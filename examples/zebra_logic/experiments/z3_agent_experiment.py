from typing import cast

import folio_experiments as fe

import delphyne as dp

configs_z3_agent = [
    fe.Z3AgentConfig(
        bench_id=bench_id,
        model_name="gpt-5-nano",
        reasoning_effort=cast(dp.ReasoningEffort, reasoning_effort),
        temperature=None,
        max_dollar_budget=0.01,
        seed=0,
        num_requests=10,
    )
    for bench_id in [id for id in fe.SAMPLE_IDS_9feb_200]
    for reasoning_effort in [
        "low",
    ]
]


if __name__ == "__main__":
    dp.Experiment(
        config_class=fe.Z3AgentConfig,
        context=dp.workspace_execution_context(__file__),
        configs=configs_z3_agent,
        output_dir=f"experiments/output_3mar/{dp.path_stem(__file__)}",
    ).run_cli()
