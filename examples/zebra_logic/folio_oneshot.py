from dataclasses import dataclass
from typing import Literal, Never, Sequence

import fol
from z3_tools import Z3Response, run_fol_in_z3

import delphyne as dp
from delphyne import Branch, Compute, Fail, Strategy, strategy
from delphyne.stdlib.standard_models import APIType

type StepType = Literal["Constraint", "Conclusion", "All"]
type ReflectFlagTag = Literal[
    "only_if_sat", "never", "always", "only_if_unsat"
]
type StyleFlagTag = Literal["normal", "literally", "implicitly"]
type Blacklist = Sequence[fol.StrFormalization | dp.Error]

Z3_TIMEOUT = 5.0


@dataclass
class FormalizeIP:
    formalize: dp.PromptingPolicy
    check: dp.Policy[Compute, object] = dp.exec @ dp.elim_compute() & None


@dataclass
class OneShotIP:
    formalizeIP: FormalizeIP
    reflect: dp.Policy[Branch | Fail, FormalizeIP]


@dataclass
class RefinedFormalization:
    str_formalization: fol.StrFormalization | None
    reason: str | None


@dataclass
class ReflectIfSat(dp.Query[dp.Response[RefinedFormalization, Never]]):
    sentences: list[str]
    formalization: fol.StrFormalization
    model: str
    prefix: dp.AnswerPrefix

    __parser__ = dp.structured.response


@dataclass
class ReflectIfUnsat(dp.Query[dp.Response[RefinedFormalization, Never]]):
    sentences: list[str]
    formalization: fol.StrFormalization
    unsat_core: str
    prefix: dp.AnswerPrefix

    __parser__ = dp.structured.response


@dataclass
class ReflectFlag(dp.FlagQuery[ReflectFlagTag]):
    """
    Flag that indicates reflection behavior.
    """


@dataclass
class StyleFlag(dp.FlagQuery[StyleFlagTag]):
    """
    Flag that indicates the style of formalization.
    """


@dataclass
class FormalizeFOLOneShot(dp.Query[dp.Response[fol.StrFormalization, Never]]):
    sentences: list[str]
    prefix: dp.AnswerPrefix
    style: StyleFlagTag = "normal"

    __parser__ = dp.structured.response


@dataclass
class Verdict:
    reflection_flag: ReflectFlagTag
    style_flag: StyleFlagTag
    formalizations: list[fol.StrFormalization]
    refined_formalizations: list[RefinedFormalization] | None
    first_solution: bool | None
    final_solution: bool | None
    judgement_solution: bool | None = None


def _obviously_invalid_refinement(
    original: fol.StrFormalization,
    refined: fol.StrFormalization,
) -> dp.Error | None:
    def to_set(lst: list[str] | None) -> set[str]:
        if not lst:
            return set()
        return set([x.strip() for x in lst])

    set_refined_conclusion = to_set(refined.conclusion)
    set_refined_constraints = to_set(refined.constraints)
    mixed = (
        (set_refined_conclusion & to_set(original.constraints))
        | (set_refined_conclusion & set_refined_constraints)
        | (set_refined_constraints & to_set(original.conclusion))
    )

    if mixed:
        return dp.Error(
            label="invalid_refinement",
            meta={
                "error": "The refinement is invalid because it mixes"
                + " conclusion formulas with premise constraints.",
                "mixed_formulas": list(mixed),
            },
        )
    return None


@strategy
def reflect(
    sat_or_unsat: Literal["sat", "unsat"],
    sentences: list[str],
    formalizations: list[fol.StrFormalization],
    model_or_unsat_core: str,
) -> Strategy[Branch | Fail, FormalizeIP, tuple[Z3Response, str | None]]:
    Query = ReflectIfSat if sat_or_unsat == "sat" else ReflectIfUnsat
    formalization = sum(formalizations, fol.StrFormalization())
    refinement_reason: str | None = None

    def check(refined_formalization: RefinedFormalization):
        nonlocal refinement_reason
        if refined_formalization.str_formalization is None:
            refined_formalization.str_formalization = fol.StrFormalization()
        if (
            e := _obviously_invalid_refinement(
                formalization, refined_formalization.str_formalization
            )
        ) is not None:
            return dp.const_space(e)
        else:
            refinement_reason = refined_formalization.reason
            return check_constraints(
                refined_formalization.str_formalization, step_type="All"
            ).using(lambda p: p.check, FormalizeIP)

    response = yield from dp.interact(
        step=lambda prefix, _: Query(
            sentences,
            formalization,
            model_or_unsat_core,
            prefix,
        ).using(lambda p: p.formalize, FormalizeIP),
        process=lambda refined_formalization, _: check(refined_formalization),
    )
    return response, refinement_reason


@strategy
def folio_oneshot(
    puzzle: str,
) -> Strategy[
    Branch | Fail | dp.Flag[ReflectFlag] | dp.Flag[StyleFlag],
    OneShotIP,
    Verdict,
]:
    sentences = puzzle.strip().split("\n")
    yield from dp.ensure(len(sentences) > 0, "The puzzle is empty.")
    style_flag = yield from dp.get_flag(StyleFlag)
    response = yield from dp.interact(
        step=lambda prefix, _,: FormalizeFOLOneShot(
            sentences=sentences,
            prefix=prefix,
            style=style_flag,
        ).using(lambda p: p.formalizeIP.formalize, OneShotIP),
        process=lambda formalization_str, _: check_constraints(
            formalization_str,
            step_type="All",
        ).using(lambda p: p.formalizeIP.check, OneShotIP),
    )

    reflection_flag = yield from dp.get_flag(ReflectFlag)
    refined_response = Z3Response(
        formalizations=[], status="nop", model=None, error=None
    )
    reason: str | None = None
    match response.status:
        case "sat":
            first_solution = False
            if (
                reflection_flag in ("always", "only_if_sat")
                and response.model is not None
            ):
                model = response.model
                refined_response, reason = yield from dp.branch(
                    reflect(
                        sat_or_unsat="sat",
                        sentences=sentences,
                        formalizations=response.formalizations,
                        model_or_unsat_core=model,
                    ).using(lambda p: p.reflect, OneShotIP)
                )
                solution = refined_response.status == "unsat"
            else:
                solution = False
        case "unsat":
            first_solution = True
            if (
                reflection_flag in ("always", "only_if_unsat")
                and response.model is not None
            ):
                unsat_core = response.model
                refined_response, reason = yield from dp.branch(
                    reflect(
                        sat_or_unsat="unsat",
                        sentences=sentences,
                        formalizations=response.formalizations,
                        model_or_unsat_core=unsat_core,
                    ).using(lambda p: p.reflect, OneShotIP)
                )
                solution = refined_response.status != "sat"
            else:
                solution = True
        case _:
            yield from dp.fail(label=response.status, message=response.error)
            first_solution, solution = None, None

    return Verdict(
        reflection_flag,
        style_flag,
        response.formalizations,
        [
            RefinedFormalization(fml, reason)
            for fml in refined_response.formalizations
        ],
        first_solution,
        solution,
    )


@strategy
def check_constraints(
    formalization_str: fol.StrFormalization,
    step_type: StepType,
    check_consistency_of_premises_if_all: bool = True,
    additional_formalizations: list[fol.StrFormalization] = [],
    # blacklist: Blacklist = [],
    timeout_in_seconds: float | None = None,
) -> Strategy[Compute, object, Z3Response | dp.Error]:
    if (
        (
            step_type == "All"
            and (
                not formalization_str.constraints
                or not formalization_str.conclusion
            )
        )
        or (step_type == "Constraint" and not formalization_str.constraints)
        or (step_type == "Conclusion" and not formalization_str.conclusion)
    ):
        return Z3Response(
            status="nop",
            formalizations=[],
            model=None,
            error=None,
        )
    # if blacklist, for each item in blacklist, check if additional + item
    # implies formalization_str
    # and additional + formalization_str implies item. If so, they are
    # equivalent and we should return an error.
    # if blacklist and step_type == "Constraint":
    #     new_formalizations = additional_formalizations + [formalization_str]
    #     for black in blacklist:
    #         if isinstance(black, dp.Error):
    #             continue
    #         else:
    #             # Check for equivalence between formalization_str and item
    #             old_formalizations = additional_formalizations + [black]
    #             new_implies_old = yield from dp.compute(
    #                 check_implication_in_z3
    #             )(
    #                 new_formalizations,
    #                 [black],
    #                 timeout_in_seconds=timeout_in_seconds,
    #             )
    #             old_implies_new = yield from dp.compute(
    #                 check_implication_in_z3
    #             )(
    #                 old_formalizations,
    #                 [formalization_str],
    #                 timeout_in_seconds=timeout_in_seconds,
    #             )
    #             if (
    #                 new_implies_old.status == "unsat"
    #                 and old_implies_new.status == "unsat"
    #             ):
    #                 return dp.Error(
    #                     label="fol_equivalent_formalization",
    #                     meta={
    #                         "error": "The formalization is equivalent to "
    #                         + "a blacklisted formalization.",
    #                         "formalization_str": formalization_str,
    #                         "blacklisted_item": black,
    #                     },
    #                 )

    if check_consistency_of_premises_if_all and step_type == "All":
        consistency_response = yield from dp.compute(run_fol_in_z3)(
            additional_formalizations + [formalization_str],
            step_type="Constraint",
            timeout_in_seconds=timeout_in_seconds,
        )
        if consistency_response.status == "unsat":
            return dp.Error(
                label="fol_inconsistent_premises",
                meta={
                    "error": "The provided premises are inconsistent.",
                    "formalization_str": formalization_str,
                },
            )

    response = yield from dp.compute(run_fol_in_z3)(
        additional_formalizations + [formalization_str],
        "All" if step_type in ("Conclusion", "All") else "Constraint",
        timeout_in_seconds=timeout_in_seconds,
    )
    if response.status == "error":
        return dp.Error(
            label="fol_interpretation_error",
            meta={
                "error": response.error,
                "formalization_str": formalization_str,
            },
        )
    if response.status == "unknown":
        return dp.Error(
            label="fol_unknown_result",
            meta={
                "error": "The FOL interpretation resulted in an unknown "
                + "status in Z3.",
                "formalization_str": formalization_str,
            },
        )
    if step_type == "Constraint" and response.status == "unsat":
        return dp.Error(
            label="fol_inconsistent_constraints",
            meta={
                "error": "The provided constraints are inconsistent.",
                "formalization_str": formalization_str,
            },
        )
    return response


@dp.ensure_compatible(reflect)
def reflect_policy(
    model_name: str = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    temperature: float | None = None,
    timeout_in_seconds: float = Z3_TIMEOUT,
    max_depth: int = 5,
    api_type: APIType = "chat_completions",
) -> dp.Policy[Branch | Fail, FormalizeIP]:
    model = dp.standard_model(
        model_name,
        {"reasoning_effort": reasoning_effort},
        api_type=api_type,
    )
    pp = dp.take(1) @ dp.few_shot(model, temperature=temperature)
    ip = FormalizeIP(
        formalize=pp,
        check=dp.exec @ elim_z3_compute(timeout_in_seconds) & None,
    )
    sp = dp.dfs(max_depth=max_depth)
    return sp & ip


@dp.ensure_compatible(reflect)
def dummy_fallback_policy() -> dp.Policy[Branch | Fail, FormalizeIP]:
    return dp.dfs() & FormalizeIP(
        formalize=dp.answer_with(
            [
                dp.Structured(  # type: ignore
                    {
                        "str_formalization": None,
                        "reason": None,
                    }
                )
            ]
        ),
        check=dp.exec @ elim_z3_compute(0) & None,
    )


@dp.ensure_compatible(folio_oneshot)
def folio_oneshot_policy(
    model_name: dp.StandardModelName = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    temperature: float | None = None,
    max_rounds: int = 3,
    style_flag: StyleFlagTag = "normal",
    reflect_flag: ReflectFlagTag = "never",
    timeout_in_seconds: float = Z3_TIMEOUT,
    api_type: APIType = "chat_completions",
) -> dp.Policy[
    Branch | Fail | dp.Flag[ReflectFlag] | dp.Flag[StyleFlag], OneShotIP
]:
    model = dp.standard_model(
        model_name, {"reasoning_effort": reasoning_effort}, api_type=api_type
    )
    formalizeIP = FormalizeIP(
        formalize=dp.take(1) @ dp.few_shot(model, temperature=temperature),
        check=dp.exec @ elim_z3_compute(timeout_in_seconds) & None,
    )
    reflect = reflect_policy(
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        timeout_in_seconds=timeout_in_seconds,
        api_type=api_type,
    )  # .or_else(dummy_fallback_policy())

    return dp.dfs(max_depth=max_rounds) @ dp.elim_flag(
        ReflectFlag, reflect_flag
    ) @ dp.elim_flag(StyleFlag, style_flag) & OneShotIP(
        formalizeIP=formalizeIP,
        reflect=reflect,
    )


def elim_z3_compute(timeout: float):
    z3_compute_args = {"timeout_in_seconds": timeout}
    return dp.elim_compute(override_args=z3_compute_args)


if __name__ == "__main__":
    pass
