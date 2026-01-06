from dataclasses import dataclass
from typing import Literal

from folio_baseline import (
    FolioBaselineIP,
    FormalizeFOLConstraint,
    StepType,
    check_constraints,  # type: ignore
)
from z3_tools import Z3Response

import delphyne as dp
from delphyne import Branch, Fail, Strategy, strategy
from delphyne.stdlib.policies import ensure_compatible


@dataclass
class FolioIterativeIP:
    single_sentence: dp.Policy[Branch | Fail, FolioBaselineIP]


# @strategy
# def check_constraints_mock(
#     formalization_yaml: str,
#     step_type: StepType,
#     add_permanently: bool,
#     additional_formalizations: list[str] = [],
# ) -> "Strategy[Compute, object, Z3Response | dp.Error | str]":
#     def f(x: int) -> int:
#         return x

#     x = yield from dp.compute(f)(1)
#     assert x == 1
#     if step_type == "Constraint":
#         return formalization_yaml
#     else:
#         return dp.Error(label="mock")
#         # return Z3Response(status="unsat", model=None, error=None)


@strategy
def formalize_single_sentence(
    sentence: str,
    previous_formalizations: list[str],
    step_type: StepType,
) -> "Strategy[Branch | Fail, FolioBaselineIP, Z3Response | str]":
    formalization_response = yield from dp.interact(
        step=lambda prefix, _: FormalizeFOLConstraint(
            sentence=sentence,
            previous_formalizations=previous_formalizations,
            step=step_type,
            prefix=prefix,
        ).using(lambda p: p.formalize, FolioBaselineIP),
        process=lambda formalization_yaml, _: check_constraints(
            formalization_yaml,
            step_type,
            add_permanently=False,
            additional_formalizations=previous_formalizations,
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
    yield from dp.ensure(len(sentences) > 0, "The puzzle is empty.")

    for step_index, sentence in enumerate(sentences):
        step_type: StepType = (
            "Conclusion" if step_index == len(sentences) - 1 else "Constraint"
        )
        formalization_response = yield from dp.branch(
            formalize_single_sentence(
                sentence=sentence,
                previous_formalizations=previous_formalizations,
                step_type=step_type,
            ).using(lambda p: p.single_sentence, FolioIterativeIP),
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
) -> dp.Policy[Branch | Fail, FolioBaselineIP]:
    model = dp.standard_model(
        model_name, {"reasoning_effort": reasoning_effort}
    )
    ip = FolioBaselineIP(formalize=dp.take(1) @ dp.few_shot(model))
    # ip = FolioBaselineIP(
    #     formalize=dp.take(1)
    #     @ dp.answer_with(
    #         [
    #             "```yaml\n```",
    #             "```yaml\nPredicates:\n- Human(1)\nConstants:\n- Socrates\nConstraints:\n- Human(Socrates)\n```",
    #         ],
    #     )
    # )
    sp = dp.dfs(max_depth=max_rounds)
    return sp & ip


@ensure_compatible(folio_iterative)
def folio_iterative_policy(
    max_restarts: int = 2,
    max_requests_per_attempt: int = 10,
    max_retries_per_sentence: int = 2,
    max_rounds_per_sentence: int = 2,
    model_name: str = "gpt-5-nano",
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "low",
) -> dp.Policy[Branch | Fail, FolioIterativeIP]:
    per_attempt = dp.BudgetLimit({dp.NUM_REQUESTS: max_requests_per_attempt})
    sp = dp.with_budget(per_attempt) @ dp.dfs()
    ip = FolioIterativeIP(
        single_sentence=dp.loop(max_retries_per_sentence)
        @ formalize_single_policy(
            max_rounds_per_sentence,
            model_name,
            reasoning_effort,
        )
    )
    return sp & ip
    # return dp.sequence((sp & ip for _ in range(max_restarts)))
