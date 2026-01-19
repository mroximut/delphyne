from dataclasses import dataclass
from typing import cast

from folio_baseline import (
    Blacklist,
    FormalizeFOLConstraint,
    FormalizeIP,
    StepType,
    check_constraints,  # type: ignore
)
from z3_tools import Z3Response

import delphyne as dp
from delphyne import Branch, Fail, Strategy, strategy
from delphyne.stdlib.streams import majority_vote

Z3_TIMEOUT = 5.0


@dataclass
class FolioIterativeIP:
    single_sentence: dp.Policy[Branch | Fail, FormalizeIP]
    iterate_transformer: dp.StreamTransformer


# @dataclass
# class FormalizeFOLConstraint(dp.Query[str]):
#     sentence: str
#     previous_formalizations: list[str]
#     step: StepType
#     blacklist: Blacklist | None = None

#     __parser__ = dp.last_code_block.trim


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
#         # return Z3Response(
#         #    formalizations=[], status="unsat", model=None, error=None
#         # )


@strategy
def formalize_single_sentence_blacklist(
    sentence: str,
    previous_formalizations: list[str],
    step_type: StepType,
    blacklist: Blacklist | None = None,
) -> Strategy[
    Branch | Fail, FormalizeIP, tuple[Z3Response | dp.Error, Blacklist]
]:
    if blacklist is None:
        blacklist = []

    # response = yield from dp.interact(
    #     step=lambda prefix, _: FormalizeFOLConstraint(
    #         sentence=sentence,
    #         previous_formalizations=previous_formalizations,
    #         step=step_type,
    #         prefix=prefix,
    #         blacklist=blacklist,
    #     ).using(lambda p: p.formalize, FolioBaselineIP),
    #     process=lambda formalization_yaml, _: check_constraints_mock(
    #         formalization_yaml,
    #         step_type,
    #         add_permanently=False,
    #         additional_formalizations=previous_formalizations,
    #     ).using(
    #         dp.just_compute
    #     ),  # infinite loop?, because next of iterate exhausted
    #  max_attempts parameter to interact?: return the last dp.Error after that
    # )
    formalization = yield from dp.branch(
        FormalizeFOLConstraint(
            sentence=sentence,
            previous_formalizations=previous_formalizations,
            step=step_type,
            blacklist=blacklist,
        ).using(lambda p: p.formalize, FormalizeIP)
    )
    formalization_str: str = cast(
        dp.FinalAnswer[str], formalization.parsed
    ).final

    response = yield from dp.run(
        check_constraints(
            formalization_str,
            step_type,
            add_permanently=False,
            additional_formalizations=previous_formalizations,
            blacklist=blacklist,
        ).using(lambda p: p.check, FormalizeIP)
    )
    to_blacklist: str | dp.Error = ""
    if isinstance(response, Z3Response):
        to_blacklist = response.formalizations[-1]
    else:
        assert isinstance(response, dp.Error)
        to_blacklist = response

    return response, [*blacklist, to_blacklist]


@strategy
def folio_iterative_blacklist(
    puzzle: str,
) -> Strategy[Branch | Fail, FolioIterativeIP, bool | None]:
    sentences = puzzle.strip().split("\n")
    previous_formalizations: list[str] = []
    solution: bool | None = None
    yield from dp.ensure(len(sentences) > 0, "The puzzle is empty.")

    for step_index, sentence in enumerate(sentences):
        step_type: StepType = (
            "Conclusion" if step_index == len(sentences) - 1 else "Constraint"
        )
        formalization_response = yield from dp.branch(
            dp.iterate(
                lambda prior: formalize_single_sentence_blacklist(
                    sentence=sentence,
                    previous_formalizations=previous_formalizations,
                    step_type=step_type,
                    blacklist=prior,
                ).using(lambda p: p.single_sentence, FolioIterativeIP),
                lambda p: p.iterate_transformer,
            ),
        )
        if isinstance(formalization_response, dp.Error):
            yield from dp.fail(formalization_response.label)
        assert isinstance(formalization_response, Z3Response)
        if step_type == "Constraint":
            previous_formalizations.append(
                formalization_response.formalizations[-1]
            )
        if step_type == "Conclusion":
            if formalization_response.status == "unknown":
                solution = None
            else:
                solution = formalization_response.status == "unsat"
            break

    return solution


@dp.ensure_compatible(formalize_single_sentence_blacklist)
def formalize_single_policy(
    model_name: str = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    temperature: float | None = None,
    majority_vote_size: int | None = None,
    timeout_in_seconds: float = Z3_TIMEOUT,
) -> dp.Policy[Branch | Fail, FormalizeIP]:
    model = dp.standard_model(
        model_name, {"reasoning_effort": reasoning_effort}
    )
    pp = dp.few_shot(model, temperature=temperature)
    if majority_vote_size:
        pp = majority_vote() @ dp.take(majority_vote_size) @ pp
    else:
        pp = dp.take(1) @ pp
    ip = FormalizeIP(
        formalize=pp,
        check=dp.exec @ _elim_z3_compute(timeout_in_seconds) & None,
    )
    sp = dp.dfs()
    return sp & ip


@dp.ensure_compatible(folio_iterative_blacklist)
def folio_iterative_blacklist_policy(
    max_restarts: int = 2,
    max_requests_per_attempt: int = 10,
    max_retries_per_sentence: int = 2,
    model_name: str = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    temperature: float | None = None,
    majority_vote_size: int | None = None,
) -> dp.Policy[Branch | Fail, FolioIterativeIP]:
    def make():
        per_attempt = dp.BudgetLimit(
            {dp.NUM_REQUESTS: max_requests_per_attempt}
        )
        sp = dp.with_budget(per_attempt) @ dp.dfs(
            max_branching=max_retries_per_sentence
        )
        ip = FolioIterativeIP(
            single_sentence=formalize_single_policy(
                model_name,
                reasoning_effort,
                temperature,
                majority_vote_size,
            ),
            iterate_transformer=dp.with_budget(
                dp.BudgetLimit({dp.NUM_REQUESTS: max_retries_per_sentence})
            ),
        )
        return sp & ip

    # return dp.sequence((make() for _ in range(max_restarts)))
    return make()


def _elim_z3_compute(timeout: float):
    z3_compute_args = {"timeout_in_seconds": timeout}
    return dp.elim_compute(override_args=z3_compute_args)
