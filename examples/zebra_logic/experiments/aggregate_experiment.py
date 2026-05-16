from typing import Literal, cast

import folio_experiments as fe

import delphyne as dp

configs_aggregate = [
    fe.AggregateConfig(
        bench_id=bench_id,
        model_name="gpt-5-nano",
        max_rounds_each=max_rounds_each,
        reasoning_effort=cast(dp.ReasoningEffort, reasoning_effort),
        sequence_type=cast(
            Literal["mixed", "all_normal", "all_normal_reflect"], sequence_type
        ),
        aggregation_type="judge",
        number_of_experts=3,
        max_dollar_budget=0.01,
        seed=seed,
        api_type="chat_completions",
    )
    for bench_id in fe.VALIDATION_IDS
    for max_rounds_each, reasoning_effort, sequence_type in [
        # (10, "low", False),
        # (10, "low", "mixed"),
        (5, "low", "all_normal_reflect"),
        # (5, "medium", "all_normal_reflect"),
        # (5, "low", "all_normal_reflect"),
        # (10, "medium"),
        # (10, "minimal")
    ]
    for seed in range(3)
]


if __name__ == "__main__":
    dp.Experiment(
        config_class=fe.AggregateConfig,
        context=dp.workspace_execution_context(__file__),
        configs=configs_aggregate,
        output_dir=f"experiments/output_12may_again/{dp.path_stem(__file__)}",
    ).run_cli()
