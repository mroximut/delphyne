from dataclasses import dataclass
from typing import Literal, Never, Sequence

import fol
from z3_tools import Z3Response, check_implication_in_z3, run_fol_in_z3

import delphyne as dp
from delphyne import Branch, Compute, Fail, Strategy, strategy
from delphyne.stdlib.standard_models import APIType

type StepType = Literal["Constraint", "Conclusion", "All"]
type ReflectFlagTag = Literal[
    "never", "always", "only_if_sat", "only_if_unsat"
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
class ReflectIfSat(dp.Query[dp.Response[fol.StrFormalization, Never]]):
    sentences: list[str]
    formalizations: list[fol.StrFormalization]
    model: str
    prefix: dp.AnswerPrefix

    __parser__ = dp.structured.response


@dataclass
class ReflectIfUnsat(dp.Query[dp.Response[fol.StrFormalization, Never]]):
    sentences: list[str]
    formalizations: list[fol.StrFormalization]
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
    refined_formalizations: list[fol.StrFormalization]
    first_solution: bool | None
    final_solution: bool | None


@strategy
def reflect(
    sat_or_unsat: Literal["sat", "unsat"],
    sentences: list[str],
    formalizations: list[fol.StrFormalization],
    model_or_unsat_core: str,
) -> Strategy[Branch | Fail, FormalizeIP, Z3Response]:
    query = ReflectIfSat if sat_or_unsat == "sat" else ReflectIfUnsat
    response = yield from dp.interact(
        step=lambda prefix, _: query(
            sentences,
            formalizations,
            model_or_unsat_core,
            prefix,
        ).using(lambda p: p.formalize, FormalizeIP),
        process=lambda refined_formalization, _: check_constraints(
            refined_formalization, step_type="All"
        ).using(lambda p: p.check, FormalizeIP),
    )
    return response


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

    assert response is not None
    reflection_flag = yield from dp.get_flag(ReflectFlag)
    refined_response = Z3Response(
        formalizations=[], status="nop", model=None, error=None
    )
    match response.status:
        case "sat":
            first_solution = False
            if (
                reflection_flag in ("always", "only_if_sat")
                and response.model is not None
            ):
                model = response.model
                refined_response = yield from dp.branch(
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
                refined_response = yield from dp.branch(
                    reflect(
                        sat_or_unsat="unsat",
                        sentences=sentences,
                        formalizations=response.formalizations,
                        model_or_unsat_core=unsat_core,
                    ).using(lambda p: p.reflect, OneShotIP)
                )
                solution = (
                    refined_response.status == "unsat"
                    or refined_response.status == "nop"
                )
            else:
                solution = True
        case _:
            first_solution = None
            solution = None

    return Verdict(
        reflection_flag,
        style_flag,
        response.formalizations,
        refined_response.formalizations,
        first_solution,
        solution,
    )


@strategy
def check_constraints(
    formalization_str: fol.StrFormalization,
    step_type: StepType,
    check_consistency_of_premises_if_all: bool = True,
    additional_formalizations: list[fol.StrFormalization] = [],
    blacklist: Blacklist = [],
    timeout_in_seconds: float | None = None,
) -> Strategy[Compute, object, Z3Response | dp.Error]:
    if not formalization_str.constraints and not formalization_str.conclusion:
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
    if blacklist and step_type == "Constraint":
        new_formalizations = additional_formalizations + [formalization_str]
        for black in blacklist:
            if isinstance(black, dp.Error):
                continue
            else:
                # Check for equivalence between formalization_str and item
                old_formalizations = additional_formalizations + [black]
                new_implies_old = yield from dp.compute(
                    check_implication_in_z3
                )(
                    new_formalizations,
                    [black],
                    timeout_in_seconds=timeout_in_seconds,
                )
                old_implies_new = yield from dp.compute(
                    check_implication_in_z3
                )(
                    old_formalizations,
                    [formalization_str],
                    timeout_in_seconds=timeout_in_seconds,
                )
                if (
                    new_implies_old.status == "unsat"
                    and old_implies_new.status == "unsat"
                ):
                    return dp.Error(
                        label="fol_equivalent_formalization",
                        meta={
                            "error": "The formalization is equivalent to "
                            + "a blacklisted formalization.",
                            "formalization_str": formalization_str,
                            "blacklisted_item": black,
                        },
                    )

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
        [formalization_str], step_type, timeout_in_seconds=timeout_in_seconds
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
    fallback_reflect = dp.dfs() & FormalizeIP(
        formalize=dp.answer_with(
            [
                dp.Structured(  # type: ignore
                    {
                        "constants": None,
                        "predicates": None,
                        "constraints": None,
                        "conclusion": None,
                    }
                )
            ]
        ),
        check=dp.exec @ elim_z3_compute(timeout_in_seconds) & None,
    )

    return dp.dfs(max_depth=max_rounds) @ dp.elim_flag(
        ReflectFlag, reflect_flag
    ) @ dp.elim_flag(StyleFlag, style_flag) & OneShotIP(
        formalizeIP=formalizeIP,
        reflect=(dp.dfs(max_depth=max_rounds) & formalizeIP).or_else(
            fallback_reflect
        ),
    )


def elim_z3_compute(timeout: float):
    z3_compute_args = {"timeout_in_seconds": timeout}
    return dp.elim_compute(override_args=z3_compute_args)


if __name__ == "__main__":
    pass
