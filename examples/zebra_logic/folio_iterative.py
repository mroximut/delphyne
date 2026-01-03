from dataclasses import dataclass
from typing import Literal, assert_never

from folio_baseline import (
    FolioBaselineIP,
    FormalizeFOLConstraint,
    StepType,
    check_constraints,
)
from z3_tools import Z3Response

import delphyne as dp
from delphyne import Branch, Fail, Strategy, strategy
from delphyne.stdlib.policies import ensure_compatible


@dataclass
class FolioIterativeIP:
    single_sentence: dp.Policy[Branch, FolioBaselineIP]


@strategy
def formalize_single_sentence(
    sentence: str,
    previous_formalizations: list[str],
    step_type: StepType,
) -> "Strategy[Branch, FolioBaselineIP, Z3Response | str]":
    formalization_response = yield from dp.interact(
        step=lambda prefix, _: FormalizeFOLConstraint(
            sentence=sentence,
            previous_formalizations=previous_formalizations,
            step=step_type,
            prefix=prefix,
        ).using(lambda p: p.formalize, FolioBaselineIP),
        process=lambda formalization_yaml, _: check_constraints(
            "\n".join(previous_formalizations) + "\n" + formalization_yaml,
            step_type,
            add_permanently=False,
        ).using(dp.just_compute),
    )
    return formalization_response


@strategy
def folio_iterative(
    puzzle: str,
) -> "Strategy[Branch | Fail, FolioIterativeIP, bool | None]":
    sentences = puzzle.strip().split("\n")
    previous_formalizations: list[str] = []
    solution: bool | None = None

    if len(sentences) == 0:
        assert_never(
            (
                yield from dp.fail(
                    "empty_puzzle", message="The puzzle is empty."
                )
            )
        )

    for step_index, sentence in enumerate(sentences):
        step_type: StepType = (
            "Conclusion" if step_index == len(sentences) - 1 else "Constraint"
        )
        formalization_response = yield from dp.branch(
            formalize_single_sentence(
                sentence=sentence,
                previous_formalizations=previous_formalizations,
                step_type=step_type,
            ).using(lambda p: p.single_sentence, FolioIterativeIP)
        )
        if step_type == "Constraint":
            assert isinstance(formalization_response, str)
            previous_formalizations.append(formalization_response)
        if step_type == "Conclusion":
            assert isinstance(formalization_response, Z3Response)
            if formalization_response.status == "unknown":
                solution = None
            else:
                solution = formalization_response.status == "unsat"
            break

    return solution


@ensure_compatible(formalize_single_sentence)
def formalize_single_policy(
    max_rounds: int = 3,
    model_name: str = "gpt-5-nano",
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "low",
) -> dp.Policy[Branch, FolioBaselineIP]:
    model = dp.standard_model(
        model_name, {"reasoning_effort": reasoning_effort}
    )
    ip = FolioBaselineIP(formalize=dp.take(1) @ dp.few_shot(model))
    sp = dp.dfs(max_depth=max_rounds)
    return sp & ip


@ensure_compatible(folio_iterative)
def folio_iterative_policy(
    max_branching: int = 2,
    max_rounds: int = 3,
    model_name: str = "gpt-5-nano",
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "low",
) -> dp.Policy[Branch | Fail, FolioIterativeIP]:
    sp = dp.dfs(max_branching=max_branching)
    ip = FolioIterativeIP(
        single_sentence=formalize_single_policy(
            max_rounds, model_name, reasoning_effort
        )
    )
    return sp & ip
