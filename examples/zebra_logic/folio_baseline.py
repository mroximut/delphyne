from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Never, assert_never

import pandas as pd
from z3_tools import Z3Response, run_fol_in_z3

import delphyne as dp
from delphyne import Branch, Compute, Fail, Strategy, strategy

SOLUTIONS_CSV: Path = (
    Path(__file__).resolve().parent
    / "datasets--yale-nlp--FOLIO"
    / "folio_v2_train.csv"
)


def ensure_solution(puzzle_id: int, solution: bool) -> bool:
    df = pd.read_csv(SOLUTIONS_CSV)  #  type: ignore
    ground_truth = (
        df.loc[df["example_id"] == puzzle_id, "label"].values[0] == "True"
    )  #  type: ignore
    assert isinstance(ground_truth, bool)
    return ground_truth == solution


type StepType = Literal["Constraint", "Conclusion", "All"]


@dataclass
class FormalizeFOLConstraint(dp.Query[dp.Response[str, Never]]):
    """
    Given a `sentence` in natural language, convert it into a first-order logic
    representation using the following YAML format with sections for
    "Predicates", "Constants", and either "Constraints" or "Conclusion".
    The "Predicates" section in the response contains the predicate definitions
    in the form "Predicate(Arity)", the constants section contains the list of
    constants used in the formulae, and the "Constraints" or "Conclusion"
    sections contain the actual formalized FOL formulae.
    If it is a "Conclusion" (in that case `sentence` starts with "Conclusion:"), do not
    include a "Constraints" section. If it is a "Constraint" (in that case `sentence` does
    not start with "Conclusion:"), do not include a "Conclusion" section.
    Use previously defined predicates and constants as needed. Make sure to include
    all predicates and constants used in the current formalization in the
    respective sections. If a predicate or constant is already defined in the
    previous formalizations, make sure you are consistent with the existing
    definition.
    For that, the `previous_formalizations` that I provide to you contains the FOL
    formalizations of previous sentences in the same context, and `step`
    indicates if the current sentence is a "Constraint" or a "Conclusion".
    Do not include `previous_formalizations` or `step` as sections in your YAML
    response. These are only provided for your context.
    Always end your answer with a triple backticks block containing the
    YAML formalization, also in the case if I respond with some error message.
    List values under a key must be indented. For example a response for a
    sentence that is not a "Conclusion" should look like:
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
      its usage.
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
    - "Either P or Q" means Xor(P, Q).
    """

    sentence: str
    previous_formalizations: list[str]
    step: StepType
    prefix: dp.AnswerPrefix

    __parser__ = dp.last_code_block.trim.response


@dataclass
class FolioBaselineIP:
    formalize: dp.PromptingPolicy


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
    all predicates and constants used in the current formalization in the
    respective sections. If a predicate or constant is already defined in the
    previous formalizations, make sure you are consistent with the existing
    definition.
    Always end your answer with a triple backticks block containing the
    YAML formalization, also in the case if I respond with some error message.
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
      its usage.
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
    - "Either P or Q" means Xor(P, Q).
    """

    sentences: list[str]
    prefix: dp.AnswerPrefix

    __parser__ = dp.last_code_block.trim.response


@strategy
def folio_oneshot(
    puzzle: str,
) -> "Strategy[Branch | Fail, FolioBaselineIP, bool | None]":
    sentences = puzzle.strip().split("\n")
    formalization_response = yield from dp.interact(
        step=lambda prefix, _,: FormalizeFOLOneShot(
            sentences=sentences,
            prefix=prefix,
        ).using(lambda p: p.formalize, FolioBaselineIP),
        process=lambda formalization_yaml, _: check_constraints(
            formalization_yaml, step_type="All", add_permanently=False
        ).using(dp.just_compute),
    )
    assert isinstance(formalization_response, Z3Response)
    if formalization_response.status == "unknown":
        return None
    else:
        return formalization_response.status == "unsat"


@strategy
def folio_baseline(
    puzzle: str,
) -> "Strategy[Branch | Fail, FolioBaselineIP, bool | None | dp.Error]":
    sentences = puzzle.strip().split("\n")
    previous_formalizations: list[str] = []
    solution: bool | None = None

    if len(sentences) == 0:
        assert_never(
            (
                yield from dp.fail(
                    label="empty_puzzle",
                    message="The provided puzzle is empty.",
                )
            )
        )

    for step_index, sentence in enumerate(sentences):
        step_type: StepType = (
            "Conclusion" if step_index == len(sentences) - 1 else "Constraint"
        )
        formalization_response = yield from dp.interact(
            step=lambda prefix, _,: FormalizeFOLConstraint(
                sentence=sentence,
                previous_formalizations=previous_formalizations,
                step=step_type,
                prefix=prefix,
            ).using(lambda p: p.formalize, FolioBaselineIP),
            process=lambda formalization_yaml, _: check_constraints(
                formalization_yaml, step_type, add_permanently=True
            ).using(dp.just_compute),
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

    return solution


@strategy
def check_constraints(
    formalization_yaml: str,
    step_type: StepType,
    add_permanently: bool,
) -> "Strategy[Compute, object, Z3Response | dp.Error | str]":
    response = yield from dp.compute(run_fol_in_z3)(
        formalization_yaml, step_type, permanently=False
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
            formalization_yaml, step_type, permanently=True
        )
    if step_type in ("Conclusion", "All"):
        return response
    else:
        return formalization_yaml


@dp.ensure_compatible(folio_baseline)
def folio_baseline_policy(
    model_name: dp.StandardModelName = "gpt-5-nano",
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "low",
) -> dp.Policy[Branch | Fail, FolioBaselineIP]:
    model = dp.standard_model(
        model_name, {"reasoning_effort": reasoning_effort}
    )
    return dp.dfs() & FolioBaselineIP(
        formalize=dp.take(1) @ dp.few_shot(model)
    )


if __name__ == "__main__":
    example_puzzle = """
        All humans are mortal.
        Socrates is a human.
        Conclusion: Socrates is mortal.
    """

    budget = dp.BudgetLimit({dp.NUM_REQUESTS: 4})
    res, _ = (
        folio_baseline(example_puzzle)
        .run_toplevel(
            dp.PolicyEnv(
                demonstration_files=[],
                prompt_dirs=[Path(__file__).parent / "prompts"],
            ),
            folio_baseline_policy(),
        )
        .collect(budget=budget, num_generated=1)
    )
    print(res[0].tracked.value)
