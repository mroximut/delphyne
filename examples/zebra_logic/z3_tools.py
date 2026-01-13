from dataclasses import dataclass
from typing import Literal

import z3  # type: ignore
from fol import Not, PredicateDef, YamlFormalizationParser, Z3Interpreter

type StepType = Literal["Constraint", "Conclusion", "All"]

_global_z3_solver: z3.Solver | None = None
_global_z3_context: dict[str, object] | None = None
_global_predicates: set[PredicateDef] = set()
_global_constants: set[str] = set()
_base_context: dict[str, object] = {
    "z3": z3,
    "__sort__": z3.DeclareSort("Object"),  # type: ignore
}


def init_global_z3_solver() -> None:
    global _global_z3_solver, _global_z3_context
    if _global_z3_solver is not None:
        return
    _global_z3_solver = z3.Solver()
    _global_z3_context = _base_context.copy()


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

    formalizations: list[str]
    status: str
    model: str | None
    error: str | None


# if not permanently, delete new predicates and constants after use
def run_fol_in_z3(
    formalizations: list[str],
    step_type: StepType,
    permanently: bool,
    equivalence_target: str | None = None,
    timeout_in_seconds: float | None = None,
) -> Z3Response:
    if timeout_in_seconds is not None:
        z3.set_param("timeout", int(timeout_in_seconds * 1000))  # type: ignore
    else:
        z3.set_param("timeout", 0)  # type: ignore

    solver, context = _get_global_z3_solver()
    predicates, constants = _get_global_predicates_and_constants()
    status = "not_run"
    model_str = None
    error = None
    pushed = 0
    new_predicates: set[PredicateDef] = set()
    new_constants: set[str] = set()
    new_context: dict[str, object] = _base_context.copy()
    try:
        new_predicates, new_constants, formulae, conclusion = (
            YamlFormalizationParser.parse_multiple(
                formalizations,
                previous_predicates=predicates,
                previous_constants=constants,
            )
        )
        eq_formulae = []
        if equivalence_target:
            assert step_type == "Constraint"
            assert permanently is False
            eq_predicates, eq_constants, eq_formulae, _ = (
                YamlFormalizationParser.parse_multiple(
                    [equivalence_target],
                    previous_predicates=predicates | new_predicates,
                    previous_constants=constants | new_constants,
                )
            )
            new_predicates |= eq_predicates
            new_constants |= eq_constants

        for p in new_predicates:
            assert context.get(p.name) is None
            Z3Interpreter.register_predicate(p, new_context)
        for c in new_constants:
            assert context.get(c) is None
            Z3Interpreter.register_constant(c, new_context)
        if formulae and step_type in ("Constraint", "All"):
            solver.push()
            pushed += 1
            for f in formulae:
                z3_formula = Z3Interpreter.interpret(f, context | new_context)
                solver.add(z3_formula)  # type: ignore
        if conclusion and step_type in ("Conclusion", "All"):
            solver.push()
            pushed += 1
            for q in conclusion:
                z3_conclusion = Z3Interpreter.interpret(
                    Not(q), context | new_context
                )
                solver.add(z3_conclusion)  # type: ignore

        if equivalence_target:
            for f in eq_formulae:
                solver.push()
                pushed += 1
                z3_formula = Z3Interpreter.interpret(
                    Not(f), context | new_context
                )
                solver.add(z3_formula)  # type: ignore

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
        if pushed and not permanently:
            for _ in range(pushed):
                solver.pop()
        if permanently:
            predicates.update(new_predicates)
            constants.update(new_constants)
            context.update(new_context)
        return Z3Response(
            formalizations=formalizations,
            status=status,
            model=model_str,
            error=error,
        )


def reset_global_z3_solver() -> None:
    """Resets the global Z3 solver, clearing all predicates and constants."""
    global \
        _global_z3_solver, \
        _global_z3_context, \
        _global_predicates, \
        _global_constants
    _global_z3_solver = None
    _global_z3_context = None
    _global_predicates = set()
    _global_constants = set()
