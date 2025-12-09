from dataclasses import dataclass
from typing import Literal

import z3  # type: ignore
from fol import Not, PredicateDef, YamlListParser, Z3Interpreter

_global_z3_solver: z3.Solver | None = None
_global_z3_context: dict[str, object] | None = None
_global_predicates: set[PredicateDef] = set()
_global_constants: set[str] = set()


def init_global_z3_solver() -> None:
    global _global_z3_solver, _global_z3_context
    if _global_z3_solver is not None:
        return
    _global_z3_solver = z3.Solver()
    _global_z3_context = {"z3": z3, "__sort__": z3.DeclareSort("Object")}  # type: ignore


def _get_global_z3_solver() -> tuple[z3.Solver, dict[str, object]]:
    global _global_z3_solver, _global_z3_context
    if _global_z3_solver is None:
        init_global_z3_solver()
    assert _global_z3_solver is not None
    assert _global_z3_context is not None
    return _global_z3_solver, _global_z3_context


def _get_global_predicates_and_constants() -> tuple[
    set[PredicateDef], set[str]
]:
    global _global_predicates, _global_constants
    return _global_predicates, _global_constants


@dataclass
class Z3Response:
    """Result of running something in the global Z3 solver.
    status: One of "sat", "unsat", "unknown", or "error".
    model: The model string produced by Z3 for satisfiable problems,
        or None otherwise.
    error: An error description if something went wrong, else None.
    """

    status: str
    model: str | None
    error: str | None


def run_fol_in_z3(
    yaml_list: str, step_type: Literal["Constraint", "Conclusion"]
) -> Z3Response:
    solver, context = _get_global_z3_solver()
    predicates, constants = _get_global_predicates_and_constants()

    try:
        predicates, constants, formulae, conclusion = YamlListParser.parse(
            yaml_list, predicates, constants
        )
        for p in predicates:
            Z3Interpreter.register_predicate(p, context)
        for c in constants:
            Z3Interpreter.register_constant(c, context)
        if formulae and step_type == "Constraint":
            solver.push()
            for f in formulae:
                z3_formula = Z3Interpreter.interpret(f, context)
                solver.add(z3_formula)  # type: ignore
        if conclusion and step_type == "Conclusion":
            solver.push()
            for q in conclusion:
                z3_conclusion = Z3Interpreter.interpret(Not(q), context)
                solver.add(z3_conclusion)  # type: ignore

        result = solver.check()  # type: ignore
        if result == z3.sat:
            model = solver.model()
            model_str = str(model)
            status = "sat"
        elif result == z3.unsat:
            model_str = None
            status = "unsat"
        else:
            model_str = None
            status = "unknown"
        error = None
    except Exception as exc:
        status = "error"
        model_str = None
        error = f"{type(exc).__name__}: {str(exc)}"
    finally:
        return Z3Response(status=status, model=model_str, error=error)  # type: ignore
