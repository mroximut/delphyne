from dataclasses import dataclass

import folio_baseline as fb

import delphyne as dp
from delphyne import Branch, Fail, Strategy, strategy
from delphyne.stdlib.streams import majority_vote


@dataclass
class FolioAggregateIP:
    oneshot: dp.Policy[
        Branch | Fail | dp.Flag[fb.ReflectFlag] | dp.Flag[fb.StyleFlag],
        fb.OneShotIP,
    ]


@strategy
def folio_aggregate(
    puzzle: str,
) -> Strategy[Branch | Fail, FolioAggregateIP, bool | None]:
    sentences = puzzle.strip().split("\n")
    yield from dp.ensure(len(sentences) > 0, "The puzzle is empty.")
    _, solution = yield from dp.branch(
        fb.folio_oneshot(puzzle=puzzle).using(
            lambda p: p.oneshot, FolioAggregateIP
        )
    )
    return solution


def folio_aggregate_policy(
    model_name: dp.StandardModelName = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    max_rounds_each: int = 5,
) -> dp.Policy[Branch | Fail, FolioAggregateIP]:
    oneshot_policy_literally = fb.folio_oneshot_policy(
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        max_rounds=max_rounds_each,
        style_flag="literally",
        reflect_flag="only_if_sat",
    )
    oneshot_policy_normal = fb.folio_oneshot_policy(
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        max_rounds=max_rounds_each,
        style_flag="normal",
        reflect_flag="never",
    )
    oneshot_policy_implicitly = fb.folio_oneshot_policy(
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        max_rounds=max_rounds_each,
        style_flag="implicitly",
        reflect_flag="only_if_unsat",
    )
    oneshot = (
        majority_vote(are_equivalent=lambda x, y: x[1] == y[1])  # type: ignore
        @ dp.take(3)
        @ dp.sequence(
            (
                oneshot_policy_literally,
                oneshot_policy_normal,
                oneshot_policy_implicitly,
            )
        )
    )
    return dp.dfs() & FolioAggregateIP(oneshot=oneshot)
