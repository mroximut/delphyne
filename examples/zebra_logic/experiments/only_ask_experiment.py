from typing import cast

import folio_experiments as fe

import delphyne as dp

configs_only_ask = [
    fe.OnlyAskConfig(
        bench_id=bench_id,
        model_name="gpt-5-nano",
        reasoning_effort=cast(dp.ReasoningEffort, reasoning_effort),
        temperature=None,
        max_dollar_budget=0.01,
        seed=0,
        num_requests=reqs,
    )
    for bench_id in [id for id in fe.SAMPLE_IDS_5_may]
    for reasoning_effort, reqs in [("low", 10)]
]


if __name__ == "__main__":
    dp.Experiment(
        config_class=fe.OnlyAskConfig,
        context=dp.workspace_execution_context(__file__),
        configs=configs_only_ask,
        output_dir=f"experiments/output_5may/{dp.path_stem(__file__)}",
    ).run_cli()
