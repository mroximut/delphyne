from dataclasses import dataclass
from typing import Any, Literal, Sequence

import folio_baseline as fb

import delphyne as dp
from delphyne import Branch, Fail, Run, Strategy, strategy
from delphyne.stdlib.search.iteration import aggregate

# from delphyne.stdlib.streams import majority_vote


@dataclass
class FolioAggregateIP:
    oneshot: dp.Policy[
        Branch | Fail | dp.Flag[fb.ReflectFlag] | dp.Flag[fb.StyleFlag],
        fb.OneShotIP,
    ]


@dataclass
class AggregationType(dp.FlagQuery[Literal["majority_vote", "favor_unsat"]]):
    """Aggregation type flag."""


@strategy
def folio_aggregate(
    puzzle: str,
) -> Strategy[
    Run | Fail | dp.Flag[AggregationType],
    FolioAggregateIP,
    tuple[bool | None, Sequence[Any]],
]:
    solutions = yield from aggregate(
        space=fb.folio_oneshot(puzzle=puzzle).using(
            lambda p: p.oneshot, FolioAggregateIP
        ),
        inner_policy_type=FolioAggregateIP,
    )
    results = [sol for _, (_, sol) in solutions if sol is not None]
    yield from dp.ensure(len(results) > 0)
    aggregation_type = yield from dp.get_flag(AggregationType)
    if aggregation_type == "favor_unsat":
        res = any(results)
    else:
        res = sum(1 for r in results if r) >= len(results) / 2

    return res, solutions


@dp.ensure_compatible(folio_aggregate)
def folio_aggregate_policy(
    model_name: dp.StandardModelName = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    max_rounds_each: int = 5,
    sequence_type: Literal[
        "mixed", "all_normal", "all_normal_reflect"
    ] = "mixed",
    aggregation_type: Literal[
        "majority_vote", "favor_unsat"
    ] = "majority_vote",
) -> dp.Policy[Run | Fail | dp.Flag[AggregationType], FolioAggregateIP]:
    def make_oneshot_policy(
        style_flag: fb.StyleFlagTag, reflect_flag: fb.ReflectFlagTag
    ):
        return fb.folio_oneshot_policy(
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            max_rounds=max_rounds_each,
            style_flag=style_flag,
            reflect_flag=reflect_flag,
        )

    oneshot_policy_literally = make_oneshot_policy(
        style_flag="literally", reflect_flag="only_if_sat"
    )
    oneshot_policy_normal = make_oneshot_policy(
        style_flag="normal", reflect_flag="never"
    )
    oneshot_policy_implicitly = make_oneshot_policy(
        style_flag="implicitly", reflect_flag="only_if_unsat"
    )
    oneshot_policy_normal_reflect = make_oneshot_policy(
        style_flag="normal", reflect_flag="always"
    )

    sequence = (
        (
            oneshot_policy_literally,
            oneshot_policy_normal,
            oneshot_policy_implicitly,
        )
        if sequence_type == "mixed"
        else (oneshot_policy_normal_reflect,) * 3
        if sequence_type == "all_normal_reflect"
        else (oneshot_policy_normal,) * 3
    )

    oneshot = (
        # majority_vote(are_equivalent=lambda x, y: x[1] == y[1])  # type: ignore
        dp.take(3) @ dp.sequence(sequence)
    )
    return dp.dfs() @ dp.elim_flag(
        AggregationType, aggregation_type
    ) & FolioAggregateIP(oneshot=oneshot)
