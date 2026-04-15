from dataclasses import dataclass
from typing import Literal, Never, Sequence

from z3_tools import SessionId, Z3Response, run_fol_in_z3

import delphyne as dp
from delphyne import Branch, Compute, Fail, Strategy, strategy

type StepType = Literal["Constraint", "Conclusion", "All"]
type ReflectFlagTag = Literal[
    "never", "always", "only_if_sat", "only_if_unsat"
]
type StyleFlagTag = Literal["literally", "normal", "implicitly"]
type Blacklist = Sequence[str | dp.Error]
Z3_TIMEOUT = 5.0


@dataclass
class FormalizeFOLConstraint(dp.Query[dp.Response[str, Never]]):
    sentence: str
    previous_formalizations: list[str]
    step_type: StepType
    context: list[str] | None = None
    prefix: dp.AnswerPrefix | None = None
    blacklist: Blacklist | None = None

    __parser__ = dp.last_code_block.trim.response


@dataclass
class FormalizeIP:
    formalize: dp.PromptingPolicy
    check: dp.Policy[Compute, object] = dp.exec @ dp.elim_compute() & None


@dataclass
class OneShotIP:
    formalizeIP: FormalizeIP
    reflect: dp.Policy[Branch | Fail, FormalizeIP]


@dataclass
class FolioIterativeIP:
    single_sentence: dp.Policy[Branch | Fail, FormalizeIP]


@dataclass
class ReflectIfSat(dp.Query[dp.Response[str, Never]]):
    sentences: list[str]
    formalizations: list[str]
    model: str
    prefix: dp.AnswerPrefix

    __parser__ = dp.last_code_block.trim.response


@dataclass
class ReflectIfUnsat(dp.Query[dp.Response[str, Never]]):
    sentences: list[str]
    formalizations: list[str]
    unsat_core: str
    prefix: dp.AnswerPrefix

    __parser__ = dp.last_code_block.trim.response


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


@strategy
def reflect(
    sat_or_unsat: Literal["sat", "unsat"],
    sentences: list[str],
    formalizations: list[str],
    model_or_unsat_core: str,
) -> Strategy[Branch | Fail, FormalizeIP, Z3Response]:
    query_class = ReflectIfSat if sat_or_unsat == "sat" else ReflectIfUnsat
    response = yield from dp.interact(
        step=lambda prefix, _: query_class(
            sentences,
            formalizations,
            model_or_unsat_core,
            prefix,
        ).using(lambda p: p.formalize, FormalizeIP),
        process=lambda refined_formalization, _: check_constraints(
            refined_formalization,
            step_type="All",
            add_permanently=False,
        ).using(lambda p: p.check, FormalizeIP),
    )
    return response


@dataclass
class FormalizeFOLOneShot(dp.Query[dp.Response[str, Never]]):
    sentences: list[str]
    prefix: dp.AnswerPrefix
    style: StyleFlagTag = "normal"

    __parser__ = dp.last_code_block.trim.response


@strategy
def folio_oneshot(
    puzzle: str,
) -> Strategy[
    Branch | Fail | dp.Flag[ReflectFlag] | dp.Flag[StyleFlag],
    OneShotIP,
    tuple[list[str], tuple[bool | None, bool | None]],
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
        process=lambda formalization_yaml, _: check_constraints(
            formalization_yaml, step_type="All", add_permanently=False
        ).using(lambda p: p.formalizeIP.check, OneShotIP),
    )
    # assert isinstance(formalization_response, Z3Response)
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

    return [
        reflection_flag,
        style_flag,
        *response.formalizations,
        *refined_response.formalizations,
    ], (
        first_solution,
        solution,
    )


@strategy
def formalize_single_sentence(
    sentence: str,
    previous_formalizations: list[str],
    step_type: StepType,
) -> Strategy[Branch | Fail, FormalizeIP, Z3Response]:
    formalization_response = yield from dp.interact(
        step=lambda prefix, _: FormalizeFOLConstraint(
            sentence=sentence,
            previous_formalizations=previous_formalizations,
            step_type=step_type,
            prefix=prefix,
        ).using(lambda p: p.formalize, FormalizeIP),
        process=lambda formalization_yaml, _: check_constraints(
            formalization_yaml,
            step_type,
            add_permanently=False,
            additional_formalizations=previous_formalizations,
        ).using(lambda p: p.check, FormalizeIP),
    )
    return formalization_response


@strategy
def folio_iterative_naive(
    puzzle: str,
) -> Strategy[Branch | Fail, FolioIterativeIP, bool | None]:
    sentences = puzzle.strip().split("\n")
    previous_formalizations: list[str] = [""] * len(sentences)
    solution: bool | None = None
    yield from dp.ensure(len(sentences) > 0, "The puzzle is empty.")

    for step_index, sentence in enumerate(sentences):
        step_type: StepType = (
            "Conclusion" if step_index == len(sentences) - 1 else "Constraint"
        )
        formalization_response = yield from dp.branch(
            formalize_single_sentence(
                sentence=sentence,
                previous_formalizations=previous_formalizations[:step_index],
                step_type=step_type,
            ).using(lambda p: p.single_sentence, FolioIterativeIP),
        )
        assert isinstance(formalization_response, Z3Response)
        if step_type == "Constraint":
            previous_formalizations[step_index] = (
                formalization_response.formalizations[-1]
            )
        if step_type == "Conclusion":
            if formalization_response.status == "unknown":
                solution = None
            else:
                solution = formalization_response.status == "unsat"
            break

    return solution


@strategy
def check_constraints(
    formalization_yaml: str,
    step_type: StepType,
    add_permanently: bool,
    additional_formalizations: list[str] = [],
    check_consistency_of_premises: bool = True,
    blacklist: Blacklist = [],
    timeout_in_seconds: float | None = None,
    session_id: SessionId | None = None,
) -> Strategy[Compute, object, Z3Response | dp.Error]:
    if not formalization_yaml.strip():
        return dp.Error(
            label="fol_empty_formalization",
            meta={
                "error": "The formalization is empty. No YAML content found."
            },
        )
    if formalization_yaml.strip() == "nop":
        return Z3Response(
            status="nop",
            formalizations=[formalization_yaml],
            model=None,
            error=None,
        )
    # if blacklist, for each item in blacklist, check if additional + item implies formalization_yaml
    # and additional + formalization_yaml implies item. If so, they are equivalent and we should return an error.
    if blacklist and step_type == "Constraint":
        new_formalizations = additional_formalizations + [formalization_yaml]
        for black in blacklist:
            if isinstance(black, dp.Error):
                continue
            else:
                # Check for equivalence between formalization_yaml and item
                old_formalizations = additional_formalizations + [black]
                new_implies_old = yield from dp.compute(run_fol_in_z3)(
                    new_formalizations,
                    step_type,
                    permanently=False,
                    equivalence_target=black,
                    timeout_in_seconds=timeout_in_seconds,
                    session_id=session_id,
                )
                old_implies_new = yield from dp.compute(run_fol_in_z3)(
                    old_formalizations,
                    step_type,
                    permanently=False,
                    equivalence_target=formalization_yaml,
                    timeout_in_seconds=timeout_in_seconds,
                    session_id=session_id,
                )
                if (
                    new_implies_old.status == "unsat"
                    and old_implies_new.status == "unsat"
                ):
                    return dp.Error(
                        label="fol_equivalent_formalization",
                        meta={
                            "error": "The formalization is equivalent to a blacklisted formalization.",
                            "formalization_yaml": formalization_yaml,
                            "blacklisted_item": black,
                        },
                    )

    formalizations = additional_formalizations + [formalization_yaml]

    if check_consistency_of_premises and step_type == "All":
        consistency_response = yield from dp.compute(run_fol_in_z3)(
            additional_formalizations,
            step_type="Constraint",
            permanently=False,
            timeout_in_seconds=timeout_in_seconds,
            session_id=session_id,
        )
        if consistency_response.status == "unsat":
            return dp.Error(
                label="fol_inconsistent_premises",
                meta={
                    "error": "The provided premises are inconsistent.",
                    "formalization_yaml": formalization_yaml,
                },
            )

    response = yield from dp.compute(run_fol_in_z3)(
        formalizations,
        step_type,
        permanently=False,
        timeout_in_seconds=timeout_in_seconds,
        session_id=session_id,
    )
    if response.status == "error":
        return dp.Error(
            label="fol_interpretation_error",
            meta={
                "error": response.error,
                "formalization_yaml": formalization_yaml,
            },
        )
    if response.status == "unknown":
        return dp.Error(
            label="fol_unknown_result",
            meta={
                "error": "The FOL interpretation resulted in an unknown status in Z3.",
                "formalization_yaml": formalization_yaml,
            },
        )
    if step_type == "Constraint" and response.status == "unsat":
        return dp.Error(
            label="fol_inconsistent_constraints",
            meta={
                "error": "The provided constraints are inconsistent.",
                "formalization_yaml": formalization_yaml,
            },
        )
    if add_permanently:
        response = yield from dp.compute(run_fol_in_z3)(
            formalizations,
            step_type,
            permanently=True,
            timeout_in_seconds=timeout_in_seconds,
            session_id=session_id,
        )
    return response


@dp.ensure_compatible(formalize_single_sentence)
def formalize_single_policy(
    max_rounds: int = 3,
    model_name: str = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    temperature: float | None = None,
    timeout_in_seconds: float = Z3_TIMEOUT,
) -> dp.Policy[Branch | Fail, FormalizeIP]:
    model = dp.standard_model(
        model_name, {"reasoning_effort": reasoning_effort}
    )
    ip = FormalizeIP(
        formalize=dp.take(1) @ dp.few_shot(model, temperature=temperature),
        check=dp.exec @ elim_z3_compute(timeout_in_seconds) & None,
    )
    sp = dp.dfs(max_depth=max_rounds)
    return sp & ip


@dp.ensure_compatible(folio_oneshot)
def folio_oneshot_policy(
    model_name: dp.StandardModelName = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    temperature: float | None = None,
    max_rounds: int = 3,
    style_flag: StyleFlagTag = "normal",
    reflect_flag: ReflectFlagTag = "never",
    timeout_in_seconds: float = Z3_TIMEOUT,
) -> dp.Policy[
    Branch | Fail | dp.Flag[ReflectFlag] | dp.Flag[StyleFlag], OneShotIP
]:
    model = dp.standard_model(
        model_name, {"reasoning_effort": reasoning_effort}
    )
    formalizeIP = FormalizeIP(
        formalize=dp.take(1) @ dp.few_shot(model, temperature=temperature),
        check=dp.exec @ elim_z3_compute(timeout_in_seconds) & None,
    )
    fallback_reflect = dp.dfs() & FormalizeIP(
        formalize=dp.answer_with(["```\nnop\n```"]),
        check=dp.exec @ elim_z3_compute(Z3_TIMEOUT) & None,
    )
    return dp.dfs(max_depth=max_rounds) @ dp.elim_flag(
        ReflectFlag, reflect_flag
    ) @ dp.elim_flag(StyleFlag, style_flag) & OneShotIP(
        formalizeIP=formalizeIP,
        reflect=(dp.dfs(max_depth=max_rounds) & formalizeIP).or_else(
            fallback_reflect
        ),
    )


@dp.ensure_compatible(folio_iterative_naive)
def folio_iterative_policy(
    max_restarts: int = 2,
    max_requests_per_attempt: int = 10,
    max_retries_per_sentence: int = 2,
    max_rounds_per_retry_of_sentence: int = 2,
    model_name: str = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    temperature: float | None = None,
    timeout_in_seconds: float = Z3_TIMEOUT,
) -> dp.Policy[Branch | Fail, FolioIterativeIP]:
    per_attempt = dp.BudgetLimit({dp.NUM_REQUESTS: max_requests_per_attempt})
    sp = dp.with_budget(per_attempt) @ dp.dfs()
    ip = FolioIterativeIP(
        single_sentence=dp.loop(max_retries_per_sentence)
        @ formalize_single_policy(
            max_rounds_per_retry_of_sentence,
            model_name,
            reasoning_effort,
            temperature,
            timeout_in_seconds,
        )
    )
    # return dp.sequence((sp & ip for _ in range(max_restarts)))
    return sp & ip


def elim_z3_compute(timeout: float):
    z3_compute_args = {"timeout_in_seconds": timeout}
    return dp.elim_compute(override_args=z3_compute_args)


if __name__ == "__main__":
    pass
    # example_puzzle = """
    #     All humans are mortal.
    #     Socrates is a human.
    #     Conclusion: Socrates is mortal.
    # """

    # budget = dp.BudgetLimit({dp.NUM_REQUESTS: 4})
    # res, _ = (
    #     folio_iterative_naive(example_puzzle)
    #     .run_toplevel(
    #         dp.PolicyEnv(
    #             demonstration_files=[],
    #             prompt_dirs=[Path(__file__).parent / "prompts"],
    #         ),
    #         folio_iterative_policy(),
    #     )
    #     .collect(budget=budget, num_generated=1)
    # )
    # print(res[0].tracked.value)
