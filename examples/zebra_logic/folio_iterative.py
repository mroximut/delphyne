from dataclasses import dataclass
from typing import Literal, Never

import fol
from folio_oneshot import (
    Z3_TIMEOUT,
    APIType,
    Blacklist,
    FormalizeIP,
    StepType,
    check_constraints,
    dummy_fallback_policy,
    elim_z3_compute,
    reflect,
)
from z3_tools import Z3Response, check_implication_in_z3

import delphyne as dp
from delphyne import Branch, Fail, Strategy, strategy
from delphyne.stdlib.streams import majority_vote


@dataclass
class NumberOfChunksFlag(dp.FlagQuery[Literal["2", "1", "3", "4", "5"]]):
    """
    Flag that indicates how many chunks to split the
    constraints into for the iterative approach.
    """


@dataclass
class FormalizeFOLConstraint(dp.Query[fol.StrFormalization]):
    sentences: list[str]
    previous_formalizations: list[fol.StrFormalization]
    step_type: StepType
    predicates: list[str] | None = None
    constants: list[str] | None = None
    context: list[str] | None = None
    blacklist: Blacklist | None = None

    __parser__ = dp.structured


@dataclass
class FixPredicatesAndConstants(
    dp.Query[dp.Response[fol.StrFormalization, Never]]
):
    context: list[str]
    prefix: dp.AnswerPrefix

    __parser__ = dp.structured.response


@dataclass
class FolioIterativeIP:
    fix: dp.PromptingPolicy
    single: dp.Policy[Branch | Fail, FormalizeIP]
    reflect_if_sat: dp.Policy[Branch | Fail, FormalizeIP]


@strategy
def formalize_single_blacklist(
    sentences: list[str],
    previous_formalizations: list[fol.StrFormalization],
    step_type: StepType,
    predicates: list[str] | None = None,
    constants: list[str] | None = None,
    context: list[str] | None = None,
    blacklist: Blacklist | None = None,
) -> Strategy[
    Branch | Fail, FormalizeIP, tuple[Z3Response | dp.Error, Blacklist]
]:
    if blacklist is None:
        blacklist = []
    previous_formalizations = (
        [sum(previous_formalizations, fol.StrFormalization())]
        if previous_formalizations
        else []
    )
    formalization_str = yield from dp.branch(
        FormalizeFOLConstraint(
            sentences=sentences,
            previous_formalizations=previous_formalizations,
            step_type=step_type,
            predicates=predicates,
            constants=constants,
            blacklist=blacklist,
            context=context,
        ).using(lambda p: p.formalize, FormalizeIP)
    )
    formalization_str += fol.StrFormalization(
        predicates=predicates, constants=constants
    )

    response = yield from dp.run(
        check_constraints(
            formalization_str,
            step_type,
            additional_formalizations=previous_formalizations,
        ).using(lambda p: p.check, FormalizeIP)
    )
    if isinstance(response, Z3Response) and response.formalizations:
        to_blacklist = response.formalizations[-1]
    elif isinstance(response, dp.Error):
        to_blacklist = response
    else:
        to_blacklist = None

    return response, (
        [*blacklist, to_blacklist] if to_blacklist is not None else blacklist
    )


@strategy
def folio_iterative_blacklist(
    puzzle: str,
) -> Strategy[
    Branch | Fail | dp.Flag[NumberOfChunksFlag], FolioIterativeIP, bool | None
]:
    sentences = puzzle.strip().split("\n")
    yield from dp.ensure(len(sentences) > 0, "The puzzle is empty.")

    def check_fix_predicates_and_constants(
        predicates_and_constants: fol.StrFormalization,
    ) -> tuple[list[str], list[str]] | dp.Error:
        predicates = predicates_and_constants.predicates or []
        constants = predicates_and_constants.constants or []
        if not predicates:
            return dp.Error(label="predicates are empty")
        try:
            fol.FormalizationParser.parse_multiple([predicates_and_constants])
        except Exception as e:
            return dp.Error(
                label="invalid predicates or constants", meta={"error": str(e)}
            )
        return predicates, constants

    predicates, constants = yield from dp.interact(
        step=lambda prefix, _: FixPredicatesAndConstants(
            context=sentences,
            prefix=prefix,
        ).using(lambda p: p.fix, FolioIterativeIP),
        process=lambda fml, _: dp.const_space(
            check_fix_predicates_and_constants(fml)
        ),
    )

    response: Z3Response | None = None
    # we split the constraints into n chunks
    constraints = sentences[:-1]
    conclusion = [sentences[-1]]
    flag = yield from dp.get_flag(NumberOfChunksFlag)
    n = int(flag)
    chunk_size = max(1, len(constraints) // n)
    chunks = [
        constraints[i : i + chunk_size]
        for i in range(0, len(constraints), chunk_size)
    ]
    empty: list[str] = []
    if len(chunks) > n:
        chunks = [*chunks[: n - 1], sum(chunks[n - 1 :], empty)]

    previous_formalizations: list[fol.StrFormalization] = [
        fol.StrFormalization()
    ] * (len(chunks) + 1)

    for step_index, chunk in enumerate(chunks + [conclusion]):
        step_type: StepType = (
            "Constraint"
            if step_index != len(chunks + [conclusion]) - 1
            else "Conclusion"
        )
        formalization_response = yield from dp.branch(
            dp.iterate(
                lambda prior: formalize_single_blacklist(
                    sentences=chunk,
                    previous_formalizations=previous_formalizations[
                        :step_index
                    ],
                    step_type=step_type,
                    predicates=predicates,
                    constants=constants,
                    context=sentences,
                    blacklist=prior,
                ).using(lambda p: p.single, FolioIterativeIP),
            )
        )
        if isinstance(formalization_response, dp.Error):
            yield from dp.fail(formalization_response.label)
        else:
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
                refined_response, _ = yield from dp.branch(
                    reflect(
                        "sat",
                        sentences=sentences,
                        formalizations=previous_formalizations,
                        model_or_unsat_core=model,
                    ).using(lambda p: p.reflect_if_sat, FolioIterativeIP)
                )
                solution = refined_response.status == "unsat"
            else:
                solution = False
        case "unsat":
            solution = True
        case _:
            yield from dp.fail(label=response.status, message=response.error)
            solution = None

    return solution


def are_equivalent(
    fml_1: fol.StrFormalization, fml_2: fol.StrFormalization
) -> bool:
    if fml_1 == fol.StrFormalization() or fml_2 == fol.StrFormalization():
        return False
    if (
        fml_1.constraints
        and fml_2.constraints
        and fml_1.constraints == fml_2.constraints
    ) or (
        not fml_1.constraints
        and not fml_2.constraints
        and fml_1.conclusion
        and fml_2.conclusion
        and fml_1.conclusion == fml_2.conclusion
    ):
        return True
    try:
        forward = check_implication_in_z3(
            [fml_1], [fml_2], timeout_in_seconds=Z3_TIMEOUT
        )
        backward = check_implication_in_z3(
            [fml_2], [fml_1], timeout_in_seconds=Z3_TIMEOUT
        )
        return forward.status == "unsat" and backward.status == "unsat"
    except Exception:
        return False


@dp.ensure_compatible(formalize_single_blacklist)
@dp.ensure_compatible(reflect)
def formalize_single_policy(
    model_name: str = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    temperature: float | None = None,
    majority_vote_size: int | None = None,
    timeout_in_seconds: float = Z3_TIMEOUT,
    max_depth: int = 5,
    api_type: APIType = "chat_completions",
) -> dp.Policy[Branch | Fail, FormalizeIP]:
    model = dp.standard_model(
        model_name,
        {"reasoning_effort": reasoning_effort},
        api_type=api_type,
    )
    pp = dp.few_shot(model, temperature=temperature)
    if majority_vote_size:
        pp = (
            majority_vote(
                are_equivalent=are_equivalent,  # type: ignore
                top_k=majority_vote_size,
            )
            @ dp.take(majority_vote_size)
            @ pp
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
    max_requests_per_attempt: int = 30,
    max_retries_per_chunk: int = 3,
    model_name: str = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    reflect_if_sat: bool = True,
    temperature: float | None = None,
    majority_vote_size: int | None = 3,
    number_of_chunks: int = 2,
    max_depth_for_formalize: int = 5,
    timeout_in_seconds: float = Z3_TIMEOUT,
    api_type: APIType = "chat_completions",
) -> dp.Policy[Branch | Fail | dp.Flag[NumberOfChunksFlag], FolioIterativeIP]:
    def make_formalize_single(majority_vote: bool):
        return formalize_single_policy(
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            majority_vote_size=majority_vote_size if majority_vote else None,
            max_depth=max_depth_for_formalize,
            timeout_in_seconds=timeout_in_seconds,
            api_type=api_type,
        )

    def make(max_retries_per_chunk: int):
        per_attempt = dp.BudgetLimit(
            {dp.NUM_REQUESTS: max_requests_per_attempt}
        )
        sp = dp.with_budget(per_attempt) @ dp.dfs(
            max_branching=max_retries_per_chunk
        )
        fallback_reflect = dummy_fallback_policy()
        formalize = make_formalize_single(
            majority_vote=(majority_vote_size is not None)
        )
        formalize_reflect = (
            make_formalize_single(majority_vote=False)  # .or_else(
            #    fallback_reflect
            # )
            if reflect_if_sat
            else fallback_reflect
        )
        fix = dp.take(max_retries_per_chunk // 2) @ dp.few_shot(
            dp.standard_model(
                model_name,
                {"reasoning_effort": reasoning_effort},
                api_type=api_type,
            ),
            temperature=temperature,
        )
        ip = FolioIterativeIP(
            fix=fix, single=formalize, reflect_if_sat=formalize_reflect
        )
        return (
            sp @ dp.elim_flag(NumberOfChunksFlag, str(number_of_chunks)) & ip
        )

    return dp.sequence(
        make(max_retries_per_chunk * (restart + 1))
        for restart in range(max_restarts)
    )
