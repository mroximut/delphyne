import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Never, assert_never, cast

import pandas as pd
import z3  # type: ignore
from zebra_parser import (
    SolverState,
    ZebraPuzzle,
    build_solver,
    model_to_solution,
    parse_puzzle,
)

import delphyne as dp
from delphyne import Branch, Compute, Fail, Strategy, strategy

# pyright: strict
# fmt: on

SOLUTIONS_CSV: Path = (
    Path(__file__).resolve().parent
    / "datasets--allenai--ZebraLogicBench-private"
    / "zebra_logic_bench_solutions.csv"
)


def execute_constraint_snippet(
    snippet: str, *, state: SolverState
) -> list[z3.BoolRef]:
    global_vars = state.exec_global_vars
    local_vars: dict[str, object] = {}
    exec(snippet, global_vars, local_vars)

    candidate = local_vars.get("constraints")
    if candidate is None:
        raise ValueError("Expected the snippet to define `constraints`.")
    values: list[z3.BoolRef] = []
    for item in cast(Iterable[z3.BoolRef], candidate):
        values.append(item)
    return values


@dataclass
class FormalizeClue(dp.Query[dp.Response[str, Never]]):
    """
    Convert a single `clue` from a Zebra logic puzzle into Z3 constraints. The
    uniqueness contraints from `context` are already encoded in the solver.
    Each variable from `variable_names` is of type z3.Int() and
    available for use in your output snippet. A value of a variable is the
    number of the house that object occupies. Additionally, note that
    `previous_snippets` are used for formalizing previous clues, and `step`
    indicates the index of the current clue being formalized (0-based).

    Return Python code wrapped in a triple-backtick block. The code must define
    a list named `constraints` containing z3.BoolRef (if there is only one
    constraint, then it is a one-element list) capturing only the information
    from the provided clue. Do not modify the solver or call `check`. You must
    prepend `z3.` to all Z3 API calls. Do not include any import statements.
    """

    context: str
    clue: str
    previous_snippets: list[str]
    variable_names: list[str]
    step: int
    prefix: dp.AnswerPrefix

    __parser__ = dp.last_code_block.trim.response


@dataclass
class ZebraBaselineIP:
    formalize: dp.PromptingPolicy


def check_solution(
    solution: list[list[str]], puzzle_id: str, solutions_df: pd.DataFrame
) -> bool:
    """Check if the given solution matches the expected solution for the
    puzzle ID."""

    expected_solution_row = solutions_df[solutions_df["id"] == puzzle_id]
    if expected_solution_row.empty:
        raise ValueError(
            f"No expected solution found for puzzle ID {puzzle_id}."
        )
    expected_solution_str = expected_solution_row.iloc[0]["solution"]
    expected_solution = json.loads(expected_solution_str)
    return solution == expected_solution


@strategy
def zebra_baseline(
    puzzle: str, puzzle_id: str | None = None
) -> "Strategy[Branch | Fail, ZebraBaselineIP, list[list[str]]]":
    try:
        parsed: ZebraPuzzle = parse_puzzle(puzzle, id=puzzle_id)
    except ValueError as exc:
        assert_never((yield from dp.fail("parse_error", message=str(exc))))

    try:
        state: SolverState = build_solver(parsed)
    except RuntimeError as exc:
        assert_never(
            (yield from dp.fail("solver_build_error", message=str(exc)))
        )

    code_history: list[str] = []
    for step_idx, clue in enumerate(parsed.clues):
        # LLM formalizes clue, gets feedback if wrong, retries
        snippet = yield from dp.interact(
            step=lambda prefix, _,: FormalizeClue(
                context=parsed.context,
                clue=clue,
                previous_snippets=code_history,
                variable_names=state.variable_names,
                step=step_idx,
                prefix=prefix,
            ).using(lambda p: p.formalize, ZebraBaselineIP),
            process=lambda snippet, _: check_and_add_constraints(
                state, snippet, is_last=(step_idx == len(parsed.clues) - 1)
            ).using(dp.just_compute),
        )

        code_history.append(snippet)

    state.solver.check()  # type: ignore
    solution = model_to_solution(state)
    if puzzle_id:
        solutions_df = pd.read_csv(SOLUTIONS_CSV)  # type: ignore
        yield from dp.ensure(check_solution(solution, puzzle_id, solutions_df))
    return solution


@strategy
def check_and_add_constraints(
    state: SolverState, snippet: str, is_last: bool
) -> dp.Strategy[Compute, object, str | dp.Error]:
    def smt_call(snippet: str) -> str:
        try:
            print(snippet)
            constraints = execute_constraint_snippet(snippet, state=state)
            state.last_constraints[0] = constraints
            result = state.solver.check(*state.last_constraints[0])  # type: ignore
            return str(result)
        except Exception as exc:
            return f"__error__:{type(exc).__name__}:{str(exc)}"

    sat_result = yield from dp.compute(smt_call)(snippet)

    if sat_result.startswith("__error__"):
        _, exc_type, exc_msg = sat_result.split(":", 2)
        return dp.Error(
            label="constraint_execution_error",
            meta={
                "error": exc_type + "\n" + exc_msg,
                "snippet": snippet,
            },
        )
    elif sat_result == "sat":
        state.solver.add(*state.last_constraints[0])  # type: ignore
        # if is_last:
        #     print("Final check...")
        #     def check_sat() -> str:
        #         result = state.solver.check()  # type: ignore
        #         return str(result)
        #     _ = yield from dp.compute(check_sat)()
        return snippet
    else:
        return dp.Error(
            label="unsatisfiable_constraints",
            meta={
                "snippet": snippet,
                "error": "The formalized constraints make the puzzle "
                "unsatisfiable.",
            },
        )


@dp.ensure_compatible(zebra_baseline)
def zebra_baseline_policy(
    model_name: dp.StandardModelName = "gpt-5-mini",
) -> dp.Policy[Branch | Fail, ZebraBaselineIP]:
    model = dp.standard_model(model_name)
    return dp.dfs() & ZebraBaselineIP(formalize=dp.few_shot(model))


if __name__ == "__main__":
    example_puzzle = """
    There are 2 houses, numbered 1 to 2 from left to right, as seen from across 
    the street. Each house is occupied by a different person. Each house has a 
    unique attribute for each of the following characteristics:
    - Each person has a unique name: `Arnold`, `Eric`
    - People use unique phone models: `samsung galaxy s21`, `iphone 13`
    - Each person has a unique type of pet: `dog`, `cat`

    ## Clues:
    1. The person who owns a dog is directly left of Arnold.
    2. The person who uses an iPhone 13 is the person who has a cat.
    """
    puzzle_id = "lgp-test-2x3-36"
    state = build_solver(parse_puzzle(example_puzzle, id=puzzle_id))
    # constraints = execute_constraint_snippet(
    #     "constraints = [type_of_pet_dog == name_arnold - 1]", state=state
    # )
    # for c in constraints:
    #     state.solver.add(c)  # type: ignore
    # constraints = execute_constraint_snippet(
    #     "constraints = [phone_models_iphone_13 == type_of_pet_cat]",
    #     state=state,
    # )
    # for c in constraints:
    #     state.solver.add(c)  # type: ignore
    # assert state.solver.check() == z3.sat  # type: ignore
    # print(model_to_solution(state))
    # assert check_solution(model_to_solution(state), puzzle_id, solutions_csv)

    budget = dp.BudgetLimit({dp.NUM_REQUESTS: 2})
    res, _ = (
        zebra_baseline(example_puzzle, puzzle_id)
        .run_toplevel(
            dp.PolicyEnv(
                demonstration_files=[],
                prompt_dirs=[Path(__file__).parent / "prompts"],
            ),
            zebra_baseline_policy(),
        )
        .collect(budget=budget, num_generated=1)
    )
    print(res)
