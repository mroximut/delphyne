from dataclasses import dataclass, replace
from typing import Literal

import z3  # type: ignore
from fol import (
    Const,
    Formalization,
    FormalizationParser,
    Not,
    PredicateDef,
    StrFormalization,
    Z3Interpreter,
    pretty_print,
)

type StepType = Literal["Constraint", "All"]

_global_z3_solver: z3.Solver | None = None
_global_z3_context: dict[str, object] | None = None
_global_predicates: set[PredicateDef] = set()
_global_constants: set[Const] = set()
_base_context: dict[str, object] = {
    "z3": z3,
    "__sort__": z3.DeclareSort("Object"),  # type: ignore
}


def _init_global_z3_solver():
    global _global_z3_solver, _global_z3_context
    if _global_z3_solver is not None:
        return
    _global_z3_solver = z3.Solver()
    _global_z3_context = _base_context.copy()


def _reset_global_z3_solver_and_context():
    global _global_z3_solver, _global_z3_context
    _global_z3_solver = z3.Solver()
    _global_z3_context = _base_context.copy()


def _get_global_z3_solver() -> tuple[z3.Solver, dict[str, object]]:
    global _global_z3_solver, _global_z3_context
    if _global_z3_solver is None:
        _init_global_z3_solver()
    assert _global_z3_solver is not None
    assert _global_z3_context is not None
    return _global_z3_solver, _global_z3_context


def _get_global_predicates_and_constants() -> tuple[
    set[PredicateDef], set[Const]
]:
    global _global_predicates, _global_constants
    return _global_predicates, _global_constants


def _set_global_predicates_and_constants(
    predicates: set[PredicateDef], constants: set[Const]
):
    global _global_predicates, _global_constants
    _global_predicates = predicates
    _global_constants = constants


def _reset_global_predicates_and_constants():
    global _global_predicates, _global_constants
    _global_predicates = set()
    _global_constants = set()


@dataclass
class Z3Response:
    """Result of running something in the global Z3 solver.
    status: One of "sat", "unsat", "unknown", or "error".
    model: The model string produced by Z3 for satisfiable problems,
        or None otherwise.
    error: An error description if something went wrong, else None.
    """

    formalizations: list[StrFormalization]
    status: Literal[
        "sat", "unsat", "unknown", "error", "not_run", "nop", "equivalent"
    ]
    model: str | None
    error: str | None


def make_formalization(
    str_formalizations: list[StrFormalization],
) -> Formalization:
    prev_predicates, prev_constants = _get_global_predicates_and_constants()
    return FormalizationParser.parse_multiple(
        str_formalizations,
        previous_predicates=prev_predicates,
        previous_constants=prev_constants,
    )


def check_implication_in_z3(
    str_formalization_1: list[StrFormalization],
    str_formalization_2: list[StrFormalization],
    timeout_in_seconds: float | None = None,
) -> Z3Response:
    _reset_global_predicates_and_constants()
    fml_1 = make_formalization(str_formalization_1)
    _set_global_predicates_and_constants(
        predicates=fml_1.predicates, constants=fml_1.constants
    )
    fml_2 = make_formalization(str_formalization_2)

    compare_conclusions = not fml_1.formulae and not fml_2.formulae

    forward = Formalization(
        predicates=fml_1.predicates | fml_2.predicates,
        constants=fml_1.constants | fml_2.constants,
        formulae=fml_1.conclusion if compare_conclusions else fml_1.formulae,
        conclusion=fml_2.conclusion if compare_conclusions else fml_2.formulae,
    )
    return run_fml_in_z3(forward, "All", timeout_in_seconds=timeout_in_seconds)


def run_fol_in_z3(
    str_formalizations: list[StrFormalization],
    step_type: StepType,
    reset_on_start: bool = True,
    persist_predicates_and_constants: bool = False,
    persist_solver: bool = False,
    timeout_in_seconds: float | None = None,
) -> Z3Response:
    if reset_on_start:
        _reset_global_predicates_and_constants()
    try:
        formalization = make_formalization(str_formalizations)
    except Exception as e:
        return Z3Response(
            formalizations=str_formalizations,
            status="error",
            model=None,
            error="Could not parse formalization: " + str(e),
        )
    ret = run_fml_in_z3(
        formalization=formalization,
        step_type=step_type,
        reset_on_start=reset_on_start,
        persist_predicates_and_constants=persist_predicates_and_constants,
        persist_solver=persist_solver,
        timeout_in_seconds=timeout_in_seconds,
    )
    return replace(ret, formalizations=str_formalizations)


def run_fml_in_z3(
    formalization: Formalization,
    step_type: StepType,
    reset_on_start: bool = True,
    persist_predicates_and_constants: bool = False,
    persist_solver: bool = False,
    timeout_in_seconds: float | None = None,
) -> Z3Response:
    if reset_on_start:
        _reset_global_z3_solver_and_context()
    if timeout_in_seconds is not None:
        z3.set_param("timeout", int(timeout_in_seconds * 1000))  # type: ignore
    else:
        z3.set_param("timeout", 0)  # type: ignore

    solver, context = _get_global_z3_solver()

    solver.set(unsat_core=True)  # type: ignore
    status = "not_run"
    model_str = None
    error = None
    new_predicates: set[PredicateDef] = set()
    new_constants: set[Const] = set()
    try:
        new_predicates = formalization.predicates
        new_constants = formalization.constants

        for p in new_predicates:
            context = Z3Interpreter.register_predicate(p, context)
        for c in new_constants:
            context = Z3Interpreter.register_constant(c, context)
        if step_type in ("Constraint", "All"):
            for fml in formalization.formulae:
                z3_formula = Z3Interpreter.interpret(fml, context)
                solver.assert_and_track(z3_formula, pretty_print(fml))  # type: ignore
        if step_type == "All":
            for q in formalization.conclusion:
                z3_conclusion = Z3Interpreter.interpret(Not(q), context)
                solver.assert_and_track(z3_conclusion, pretty_print(Not(q)))  # type: ignore

        result = solver.check()  # type: ignore
        if result == z3.sat:
            model = solver.model()
            model_str = str(model)
            status = "sat"
        elif result == z3.unsat:
            model = solver.unsat_core()
            model_str = str(model)
            status = "unsat"
        else:
            model_str = None
            status = "unknown"
    except Exception as e:
        status = "error"
        model_str = None
        error = f"{type(e).__name__}: {str(e)}"
    finally:
        if persist_predicates_and_constants:
            _set_global_predicates_and_constants(
                predicates=new_predicates, constants=new_constants
            )
        if not persist_solver:
            _reset_global_z3_solver_and_context()
        return Z3Response(
            formalizations=[],
            status=status,
            model=model_str,
            error=error,
        )
