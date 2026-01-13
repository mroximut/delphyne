from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Never, Sequence

from z3_tools import Z3Response, run_fol_in_z3

import delphyne as dp
from delphyne import Branch, Compute, Fail, Strategy, strategy

type StepType = Literal["Constraint", "Conclusion", "All"]
type Blacklist = Sequence[str | dp.Error]


@dataclass
class FormalizeFOLConstraint(dp.Query[dp.Response[str, Never]]):
    sentence: str
    previous_formalizations: list[str]
    step: StepType
    prefix: dp.AnswerPrefix | None = None
    blacklist: Blacklist | None = None

    __parser__ = dp.last_code_block.trim.response


@dataclass
class FormalizeIP:
    formalize: dp.PromptingPolicy
    check: dp.Policy[Compute, object] = dp.exec @ dp.elim_compute() & None


@dataclass
class FolioIterativeIP:
    single_sentence: dp.Policy[Branch | Fail, FormalizeIP]


@dataclass
class FormalizeFOLOneShot(dp.Query[dp.Response[str, Never]]):
    """
    Given a list of `sentences` in natural language, convert them into a first-order logic
    representation using the following YAML format with sections for
    "Predicates", "Constants", "Constraints" and "Conclusion".
    The "Predicates" section in the response contains the predicate definitions
    in the form "Predicate(Arity)", the constants section contains the list of
    constants used in the formulae, and the "Constraints" and "Conclusion"
    sections contain the actual formalized FOL formulae.
    If a sentence is "Conclusion" (in that case `sentence` starts with "Conclusion:"),
    formalize it under "Conclusion" section. If it is a "Constraint" (in that case
    `sentence` does not start with "Conclusion:"), formalize it under "Constraints" section.
    Use previously defined predicates and constants as needed. Make sure to include
    all predicates and constants used in the formalization of the current sentence in the
    respective sections. If a predicate or constant is already defined in the
    previous formalizations, make sure you are consistent with the existing
    definition.
    Always respond with a triple backticks block containing the
    YAML formalization and nothing else. In the case if I respond you with
    some error message should you still respond with the YAML formalization.
    List values under a key must be indented. For example a response should look like:
    ```yaml
    Predicates:
    - Predicate1(2)
    - Predicate2(1)
    Constants:
    - Constant1
    - Constant2
    Constraints:
    - ForAll(x, Predicate1(x, Constant1))
    - Implies(Predicate2(Constant2), Exists(y, Predicate1(y, Constant2)))
    Conclusion:
    - Predicate2(Constant1)
    ```
    If e.g. Constants is empty, still include the section as:
    ```yaml
    Constants:
    -
    ```
    The syntax for FOL is as follows and nothing else:
    - Bounded variables are represented by lowercase letters (e.g., x, y, z).
    - Constants are represented by capitalized words (e.g., Socrates, Alice).
    - Predicates are represented by their name followed by their arguments in
      parentheses (e.g., Human(x), Loves(Alice, Bob)). Each predicate must
      include its arity in the "Predicates" section and respect that arity in
      its usage. An argument of a predicate can be either a constant or
      a bounded variable, it cannot be another predicate.
    - Logical connectives include:
      - And: conjunction (e.g., And(P, Q)) (exactly 2 arguments)
      - Or: disjunction (e.g., Or(P, Q)) (exactly 2 arguments)
      - Not: negation (e.g., Not(P)) (exactly 1 argument)
      - Xor: exclusive or (e.g., Xor(P, Q)) (exactly 2 arguments)
      - Implies: implication (e.g., Implies(P, Q)) (exactly 2 arguments)
      - Iff: biconditional (e.g., Iff(P, Q)) (exactly 2 arguments)
      - Equals: equality (e.g., Equals(x, y)) (exactly 2 arguments)
    - Quantifiers include:
      - ForAll: universal quantification (e.g., ForAll(x, P(x))) (exactly 2 arguments)
      - Exists: existential quantification (e.g., Exists(x, P(x))) (exactly 2 arguments)

    Tips:
    - Formalize statements like "Either P or Q" as Xor(P, Q).
    """

    sentences: list[str]
    prefix: dp.AnswerPrefix

    __parser__ = dp.last_code_block.trim.response


@strategy
def folio_oneshot(
    puzzle: str,
) -> Strategy[Branch | Fail, FormalizeIP, bool | None]:
    sentences = puzzle.strip().split("\n")
    yield from dp.ensure(len(sentences) > 0, "The puzzle is empty.")
    formalization_response = yield from dp.interact(
        step=lambda prefix, _,: FormalizeFOLOneShot(
            sentences=sentences,
            prefix=prefix,
        ).using(lambda p: p.formalize, FormalizeIP),
        process=lambda formalization_yaml, _: check_constraints(
            formalization_yaml, step_type="All", add_permanently=False
        ).using(dp.just_compute),
    )
    assert isinstance(formalization_response, Z3Response)
    if formalization_response.status == "unknown":
        return None
    else:
        return formalization_response.status == "unsat"


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
) -> Strategy[Branch | Fail, FormalizeIP, Z3Response]:
    formalization_response = yield from dp.interact(
        step=lambda prefix, _: FormalizeFOLConstraint(
            sentence=sentence,
            previous_formalizations=previous_formalizations,
            step=step_type,
            prefix=prefix,
        ).using(lambda p: p.formalize, FormalizeIP),
        process=lambda formalization_yaml, _: check_constraints(
            formalization_yaml,
            step_type,
            add_permanently=False,
            additional_formalizations=previous_formalizations,
        ).using(dp.just_compute),
    )
    return formalization_response


@strategy
def folio_iterative_naive(
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
            formalize_single_sentence(
                sentence=sentence,
                previous_formalizations=previous_formalizations,
                step_type=step_type,
            ).using(lambda p: p.single_sentence, FolioIterativeIP),
        )
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


@strategy
def check_constraints(
    formalization_yaml: str,
    step_type: StepType,
    add_permanently: bool,
    additional_formalizations: list[str] = [],
    blacklist: Blacklist = [],
    timeout_in_seconds: float | None = None,
) -> Strategy[Compute, object, Z3Response | dp.Error]:
    if not formalization_yaml.strip():
        return dp.Error(
            label="fol_empty_formalization",
            meta={
                "error": "The formalization is empty. No YAML content found."
            },
        )
    ### if blacklist, for each item in blacklist, check if additional + item implies formalization_yaml
    ### and additional + formalization_yaml implies item. If so, they are equivalent and we should return an error.
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
                )
                old_implies_new = yield from dp.compute(run_fol_in_z3)(
                    old_formalizations,
                    step_type,
                    permanently=False,
                    equivalence_target=formalization_yaml,
                    timeout_in_seconds=timeout_in_seconds,
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
    response = yield from dp.compute(run_fol_in_z3)(
        formalizations,
        step_type,
        permanently=False,
        timeout_in_seconds=timeout_in_seconds,
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
        )
    return response


@dp.ensure_compatible(formalize_single_sentence)
def formalize_single_policy(
    max_rounds: int = 3,
    model_name: str = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    temperature: float | None = None,
) -> dp.Policy[Branch | Fail, FormalizeIP]:
    model = dp.standard_model(
        model_name, {"reasoning_effort": reasoning_effort}
    )
    ip = FormalizeIP(
        formalize=dp.take(1) @ dp.few_shot(model, temperature=temperature)
    )
    # ip = FormalizeIP(
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


@dp.ensure_compatible(folio_oneshot)
def folio_oneshot_policy(
    model_name: dp.StandardModelName = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    temperature: float | None = None,
    max_rounds: int = 3,
) -> dp.Policy[Branch | Fail, FormalizeIP]:
    model = dp.standard_model(
        model_name, {"reasoning_effort": reasoning_effort}
    )
    return dp.dfs(max_depth=max_rounds) & FormalizeIP(
        formalize=dp.take(1) @ dp.few_shot(model, temperature=temperature)
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
        )
    )
    # return dp.sequence((sp & ip for _ in range(max_restarts)))
    return sp & ip


if __name__ == "__main__":
    example_puzzle = """
        All humans are mortal.
        Socrates is a human.
        Conclusion: Socrates is mortal.
    """

    budget = dp.BudgetLimit({dp.NUM_REQUESTS: 4})
    res, _ = (
        folio_iterative_naive(example_puzzle)
        .run_toplevel(
            dp.PolicyEnv(
                demonstration_files=[],
                prompt_dirs=[Path(__file__).parent / "prompts"],
            ),
            folio_iterative_policy(),
        )
        .collect(budget=budget, num_generated=1)
    )
    print(res[0].tracked.value)


# @strategy
# def folio_baseline(
#     puzzle: str,
# ) -> Strategy[Branch | Fail, FormalizeIP, bool | None]:
#     sentences = puzzle.strip().split("\n")
#     previous_formalizations: list[str] = []
#     solution: bool | None = None
#     yield from dp.ensure(len(sentences) > 0, "The puzzle is empty.")

#     for step_index, sentence in enumerate(sentences):
#         step_type: StepType = (
#             "Conclusion" if step_index == len(sentences) - 1 else "Constraint"
#         )
#         formalization_response = yield from dp.interact(
#             step=lambda prefix, _,: FormalizeFOLConstraint(
#                 sentence=sentence,
#                 previous_formalizations=previous_formalizations,
#                 step=step_type,
#                 prefix=prefix,
#             ).using(lambda p: p.formalize, FormalizeIP),
#             process=lambda formalization_yaml, _: check_constraints(
#                 formalization_yaml, step_type, add_permanently=True
#             ).using(dp.just_compute),
#         )
#         assert isinstance(formalization_response, Z3Response)
#         if step_type == "Constraint":
#             previous_formalizations.append(
#                 formalization_response.formalizations[-1]
#             )
#         if step_type == "Conclusion":
#             if formalization_response.status == "unknown":
#                 solution = None
#             else:
#                 solution = formalization_response.status == "unsat"

#     return solution
