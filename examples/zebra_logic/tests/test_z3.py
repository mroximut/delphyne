import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from fol import (
    Const,
    Formalization,
    Predicate,
    PredicateDef,
    StrFormalization,
)
from folio_standard import (
    RunZ3Solver,
    Z3Declaration,
    _build_z3_namespace,
    _run_z3_solver,
    _validate_z3_expr,
)
from z3_tools import (
    check_implication_in_z3,
    run_fml_in_z3,
    run_fol_in_z3,
)


def _make_formalization(
    constraints: list[str], conclusion: list[str] | None = None
) -> list[StrFormalization]:
    return [
        StrFormalization(
            predicates=["P(1)"],
            constants=["a"],
            constraints=constraints,
            conclusion=conclusion,
        )
    ]


def test_run_fol_in_z3_sat() -> None:
    response = run_fol_in_z3(
        _make_formalization(["P(a)"]), step_type="Constraint"
    )
    assert response.status == "sat"
    assert response.error is None


def test_run_fol_in_z3_unsat() -> None:
    response = run_fol_in_z3(
        _make_formalization(["P(a)", "Not(P(a))"]), step_type="Constraint"
    )
    assert response.status == "unsat"


def test_run_fol_in_z3_parse_error() -> None:
    response = run_fol_in_z3(
        _make_formalization(["P(x)"]), step_type="Constraint"
    )
    assert response.status == "error"
    assert response.error is not None


def test_run_fol_in_z3_conclusion_only_entailment() -> None:
    formalizations = [
        StrFormalization(
            predicates=["P(1)"],
            constants=["a"],
            constraints=["P(a)"],
            conclusion=["P(a)"],
        )
    ]
    response = run_fol_in_z3(formalizations, step_type="All")
    assert response.status == "unsat"


def test_check_implication_in_z3_trivial() -> None:
    fml = _make_formalization(["P(a)"])
    response = check_implication_in_z3(fml, fml)
    assert response.status == "unsat"


def test_run_fml_in_z3_constraints_only() -> None:
    formalization = Formalization(
        predicates={PredicateDef("P", 1)},
        constants={Const("a")},
        formulae=[Predicate("P", [Const("a")])],
        conclusion=[],
    )
    response = run_fml_in_z3(formalization, step_type="Constraint")
    assert response.status == "sat"
    assert response.error is None


def test_run_fml_in_z3_conclusion_unsat() -> None:
    formalization = Formalization(
        predicates={PredicateDef("P", 1)},
        constants={Const("a")},
        formulae=[Predicate("P", [Const("a")])],
        conclusion=[Predicate("P", [Const("a")])],
    )
    response = run_fml_in_z3(formalization, step_type="All")
    assert response.status == "unsat"


def test_run_fml_in_z3_persist_predicates_and_constants() -> None:
    formalization = Formalization(
        predicates={PredicateDef("P", 1)},
        constants={Const("a")},
        formulae=[Predicate("P", [Const("a")])],
        conclusion=[],
    )
    response = run_fml_in_z3(
        formalization,
        step_type="Constraint",
        persist_predicates_and_constants=True,
    )
    assert response.status == "sat"
    follow_up = run_fol_in_z3(
        _make_formalization(["P(a)"]), step_type="Constraint"
    )
    assert follow_up.status == "sat"


def test_run_fml_in_z3_reset_solver_between_runs() -> None:
    formalization = Formalization(
        predicates={PredicateDef("P", 1)},
        constants={Const("a")},
        formulae=[Predicate("P", [Const("a")])],
        conclusion=[],
    )
    first = run_fml_in_z3(formalization, step_type="Constraint")
    second = run_fml_in_z3(formalization, step_type="Constraint")
    assert first.status == "sat"
    assert second.status == "sat"


def test_run_fol_in_z3_entails_conclusion_end_to_end() -> None:
    formalizations = [
        StrFormalization(
            predicates=["Human(1)", "Mortal(1)"],
            constants=["Socrates"],
            constraints=[
                "Human(Socrates)",
                "ForAll(x, Implies(Human(x), Mortal(x)))",
            ],
            conclusion=["Mortal(Socrates)"],
        )
    ]
    response = run_fol_in_z3(formalizations, step_type="All")
    assert response.status == "unsat"
    assert response.error is None


def test_run_fol_in_z3_non_entailment_end_to_end() -> None:
    formalizations = [
        StrFormalization(
            predicates=["Human(1)", "Mortal(1)"],
            constants=["Socrates"],
            constraints=["Human(Socrates)"],
            conclusion=["Mortal(Socrates)"],
        )
    ]
    response = run_fol_in_z3(formalizations, step_type="All")
    assert response.status == "sat"


def test_validate_z3_expr_rejects_attributes() -> None:
    err = _validate_z3_expr("x.__class__")
    assert err is not None
    assert "Attribute" in err


def test_build_z3_namespace_is_restricted() -> None:
    ns = _build_z3_namespace()
    assert ns.get("__builtins__") == {}
    assert "Int" in ns
    assert "eval" not in ns


def test_run_z3_solver_unsat() -> None:
    call = RunZ3Solver(
        declarations=[Z3Declaration(name="x", expr="Int('x')")],
        constraints=["x > 0", "x < 0"],
    )
    ret = _run_z3_solver(call)
    assert ret.status == "unsat"
    assert ret.error is None


def test_run_z3_solver_end_to_end_sat_model() -> None:
    call = RunZ3Solver(
        declarations=[
            Z3Declaration(name="x", expr="Int('x')"),
            Z3Declaration(
                name="f",
                expr="Function('f', IntSort(), IntSort())",
            ),
        ],
        constraints=["x == 0", "f(x) == x + 1"],
    )
    ret = _run_z3_solver(call)
    assert ret.status == "sat"
    assert ret.error is None
    assert ret.model is not None


def test_run_z3_solver_invalid_declaration_name() -> None:
    call = RunZ3Solver(
        declarations=[Z3Declaration(name="1x", expr="Int('x')")],
        constraints=[],
    )
    ret = _run_z3_solver(call)
    assert ret.status == "error"
    assert ret.error is not None
