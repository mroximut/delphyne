from dataclasses import dataclass
from typing import cast

from folio_baseline import (
    Z3_TIMEOUT,
    Blacklist,
    FormalizeFOLConstraint,
    FormalizeIP,
    StepType,
    check_constraints,  # type: ignore
    elim_z3_compute,
    reflect_if_sat,
)
from z3_tools import Z3Response

import delphyne as dp
from delphyne import Branch, Fail, Strategy, strategy

# from delphyne.stdlib.streams import majority_vote


@dataclass
class FolioIterativeIP:
    single_sentence: dp.Policy[Branch | Fail, FormalizeIP]
    reflect_if_sat: dp.Policy[Branch | Fail, FormalizeIP]


@strategy
def formalize_single_sentence_blacklist(
    sentence: str,
    previous_formalizations: list[str],
    step_type: StepType,
    context: list[str] | None = None,
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
            step_type=step_type,
            blacklist=blacklist,
            context=context,
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
    previous_formalizations: list[str] = [""] * len(sentences)
    solution: bool | None = None
    response: Z3Response | None = None
    yield from dp.ensure(len(sentences) > 0, "The puzzle is empty.")

    for step_index, sentence in enumerate(sentences):
        step_type: StepType = (
            "Conclusion" if step_index == len(sentences) - 1 else "Constraint"
        )
        formalization_response = yield from dp.branch(
            dp.iterate(
                lambda prior: formalize_single_sentence_blacklist(
                    sentence=sentence,
                    previous_formalizations=previous_formalizations[
                        :step_index
                    ],
                    step_type=step_type,
                    context=sentences,
                    blacklist=prior,
                ).using(lambda p: p.single_sentence, FolioIterativeIP),
            ),
        )
        if isinstance(formalization_response, dp.Error):
            yield from dp.fail(formalization_response.label)
        else:
            # assert isinstance(formalization_response, Z3Response)
            previous_formalizations[step_index] = (
                formalization_response.formalizations[-1]
            )
            if step_type == "Conclusion":
                response = formalization_response
                break

    assert response is not None
    match response.status:
        case "sat":
            if response.model is not None:
                model = response.model
                refined_response = yield from dp.branch(
                    reflect_if_sat(
                        sentences=sentences,
                        formalizations=previous_formalizations,
                        model=model,
                    ).using(lambda p: p.reflect_if_sat, FolioIterativeIP)
                )
                solution = refined_response.status == "unsat"
            else:
                solution = False
        case "unsat":
            solution = True
        case _:
            solution = None

    return solution


@dp.ensure_compatible(formalize_single_sentence_blacklist)
@dp.ensure_compatible(reflect_if_sat)
def formalize_single_policy(
    model_name: str = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    temperature: float | None = None,
    majority_vote_size: int | None = None,
    timeout_in_seconds: float = Z3_TIMEOUT,
    max_depth: int = 1,
) -> dp.Policy[Branch | Fail, FormalizeIP]:
    model = dp.standard_model(
        model_name, {"reasoning_effort": reasoning_effort}
    )
    pp = dp.few_shot(model, temperature=temperature)
    if majority_vote_size:
        pp = (  # majority_vote() @
            dp.take(majority_vote_size) @ pp
        )
    else:
        pp = dp.take(1) @ pp
    ip = FormalizeIP(
        formalize=pp,
        check=dp.exec @ elim_z3_compute(timeout_in_seconds) & None,
    )
    sp = dp.dfs(max_depth=max_depth)
    return sp & ip


@dp.ensure_compatible(folio_iterative_blacklist)
def folio_iterative_blacklist_policy(
    max_restarts: int = 2,
    max_requests_per_attempt: int = 10,
    max_retries_per_sentence: int = 2,
    model_name: str = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    reflect_if_sat: bool = False,
    temperature: float | None = None,
    majority_vote_size: int | None = None,
    max_depth_for_reflect: int = 2,
) -> dp.Policy[Branch | Fail, FolioIterativeIP]:
    def make():
        per_attempt = dp.BudgetLimit(
            {dp.NUM_REQUESTS: max_requests_per_attempt}
        )
        sp = dp.with_budget(per_attempt) @ dp.dfs(
            max_branching=max_retries_per_sentence
        )
        fallback_reflect = dp.dfs() & FormalizeIP(
            formalize=dp.answer_with(["```\nnop\n```"]),
            check=dp.exec @ elim_z3_compute(Z3_TIMEOUT) & None,
        )
        formalize = formalize_single_policy(
            model_name,
            reasoning_effort,
            temperature,
            majority_vote_size,
            max_depth=max_depth_for_reflect,
        )
        ip = FolioIterativeIP(
            single_sentence=formalize,
            reflect_if_sat=formalize.or_else(fallback_reflect)
            if reflect_if_sat
            else fallback_reflect,
        )
        return sp & ip

    # sequence does not work as intented, does max_branching count as resource?
    # return dp.sequence((make() for _ in range(max_restarts)))
    return make()
