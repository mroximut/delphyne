import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from fol import (  # noqa: E402
    Const,
    Equals,
    ForAll,
    Formalization,
    Implies,
    Not,
    Predicate,
    PredicateDef,
    StrFormalization,
    Var,
)
import z3_tools as z3t  # noqa: E402
from z3_tools import (  # noqa: E402
    Z3Response,
    check_implication_in_z3,
    make_formalization,
    run_fml_in_z3,
    run_fol_in_z3,
)


DEMOS = Path(__file__).parents[1] / "demos"


@pytest.fixture(autouse=True)
def reset_z3_tool_globals() -> None:
    z3t._reset_global_predicates_and_constants()
    z3t._reset_global_z3_solver_and_context()
    yield
    z3t._reset_global_predicates_and_constants()
    z3t._reset_global_z3_solver_and_context()


def unary_form(
    constraints: list[str],
    conclusion: list[str] | None = None,
) -> list[StrFormalization]:
    return [
        StrFormalization(
            predicates=["P(1)"],
            constants=["a"],
            constraints=constraints,
            conclusion=conclusion,
        )
    ]


def implication_form(include_rule: bool = True) -> list[StrFormalization]:
    constraints = ["Human(Socrates)"]
    if include_rule:
        constraints.append("ForAll(x, Implies(Human(x), Mortal(x)))")
    return [
        StrFormalization(
            predicates=["Human(1)", "Mortal(1)"],
            constants=["Socrates"],
            constraints=constraints,
            conclusion=["Mortal(Socrates)"],
        )
    ]


def demo_answer_formalizations(demo_name: str) -> list[StrFormalization]:
    demo = yaml.safe_load((DEMOS / demo_name).read_text())
    answers = [
        query["answers"][0]["answer"]
        for entry in demo
        for query in entry["queries"]
    ]
    return [StrFormalization(**answer) for answer in answers]


def test_make_formalization_uses_previously_persisted_symbols() -> None:
    z3t._set_global_predicates_and_constants(
        {PredicateDef("P", 1)}, {Const("a")}
    )

    formalization = make_formalization(
        [StrFormalization(constraints=["P(a)"], conclusion=["P(a)"])]
    )

    assert formalization.predicates == {PredicateDef("P", 1)}
    assert formalization.constants == {Const("a")}
    assert formalization.formulae == [Predicate("P", [Const("a")])]
    assert formalization.conclusion == [Predicate("P", [Const("a")])]


@pytest.mark.parametrize(
    ("constraints", "expected_status"),
    [
        (["P(a)"], "sat"),
        (["P(a)", "Not(P(a))"], "unsat"),
        (["Equals(a, a)"], "sat"),
        (["Not(Equals(a, a))"], "unsat"),
    ],
)
def test_run_fol_in_z3_constraint_mode_reports_satisfiability(
    constraints: list[str], expected_status: str
) -> None:
    response = run_fol_in_z3(
        unary_form(constraints),
        step_type="Constraint",
    )

    assert response.status == expected_status
    assert response.error is None
    assert response.formalizations == unary_form(constraints)
    assert response.model is not None


def test_run_fol_in_z3_constraint_mode_ignores_conclusions() -> None:
    response = run_fol_in_z3(
        unary_form(["P(a)"], conclusion=["P(a)"]),
        step_type="Constraint",
    )

    assert response.status == "sat"
    assert response.error is None


def test_run_fol_in_z3_all_mode_proves_and_refutes_conclusions() -> None:
    entailed = run_fol_in_z3(
        implication_form(include_rule=True), step_type="All"
    )
    not_entailed = run_fol_in_z3(
        implication_form(include_rule=False), step_type="All"
    )

    assert entailed.status == "unsat"
    assert entailed.error is None
    assert entailed.model is not None

    assert not_entailed.status == "sat"
    assert not_entailed.error is None
    assert not_entailed.model is not None


def test_folio_oneshot_demo_proves_conclusion() -> None:
    formalizations = demo_answer_formalizations("folio_oneshot.demo.yaml")

    parsed = make_formalization(formalizations)
    response = run_fol_in_z3(formalizations, step_type="All")

    assert len(formalizations) == 1
    assert {p.name for p in parsed.predicates} == {
        "RegularlyDrinkCoffee",
        "DependentOnCaffeine",
        "WantsAddiction",
        "Student",
        "AwareThatCaffeineDrug",
    }
    assert {c.name for c in parsed.constants} == {"Rina"}
    assert len(parsed.formulae) == 5
    assert len(parsed.conclusion) == 1
    assert response.status == "unsat"
    assert response.error is None


def test_folio_iterative_demo_proves_conclusion() -> None:
    formalizations = demo_answer_formalizations("folio_iterative.demo.yaml")

    parsed = make_formalization(formalizations)
    response = run_fol_in_z3(formalizations, step_type="All")

    assert len(formalizations) == 4
    assert {p.name for p in parsed.predicates} == {
        "RegularlyDrinkCoffee",
        "DependentOnCaffeine",
        "WantsAddiction",
        "Student",
        "AwareThatCaffeineDrug",
    }
    assert {c.name for c in parsed.constants} == {"Rina"}
    assert len(parsed.formulae) == 5
    assert len(parsed.conclusion) == 1
    assert response.status == "unsat"
    assert response.error is None


def test_folio_iterative_demo_can_be_executed_incrementally() -> None:
    declarations, first_constraints, second_constraints, conclusion = (
        demo_answer_formalizations("folio_iterative.demo.yaml")
    )

    declare_response = run_fol_in_z3(
        [declarations],
        step_type="Constraint",
        persist_predicates_and_constants=True,
        persist_solver=True,
    )
    first_response = run_fol_in_z3(
        [first_constraints],
        step_type="Constraint",
        reset_on_start=False,
        persist_predicates_and_constants=True,
        persist_solver=True,
    )
    second_response = run_fol_in_z3(
        [second_constraints],
        step_type="Constraint",
        reset_on_start=False,
        persist_predicates_and_constants=True,
        persist_solver=True,
    )
    final_response = run_fol_in_z3(
        [conclusion],
        step_type="All",
        reset_on_start=False,
        persist_solver=True,
    )

    assert declare_response.status == "sat"
    assert first_response.status == "sat"
    assert second_response.status == "sat"
    assert final_response.status == "unsat"
    assert final_response.error is None


def test_run_fol_in_z3_returns_parse_errors_without_raising() -> None:
    response = run_fol_in_z3(unary_form(["P(x)"]), step_type="Constraint")

    assert response == Z3Response(
        formalizations=unary_form(["P(x)"]),
        status="error",
        model=None,
        error=response.error,
    )
    assert response.error is not None
    assert response.error.startswith("Could not parse formalization:")
    assert "Unknown symbol in term position" in response.error


def test_run_fml_in_z3_constraint_and_all_modes_with_ast_input() -> None:
    formalization = Formalization(
        predicates={PredicateDef("P", 1), PredicateDef("Q", 1)},
        constants={Const("a")},
        formulae=[
            Predicate("P", [Const("a")]),
            ForAll(
                Var("x"),
                Implies(
                    Predicate("P", [Var("x")]),
                    Predicate("Q", [Var("x")]),
                ),
            ),
        ],
        conclusion=[Predicate("Q", [Const("a")])],
    )

    constraints_only = run_fml_in_z3(formalization, step_type="Constraint")
    all_steps = run_fml_in_z3(formalization, step_type="All")

    assert constraints_only.status == "sat"
    assert constraints_only.error is None
    assert all_steps.status == "unsat"
    assert all_steps.error is None


def test_run_fml_in_z3_catches_interpreter_errors() -> None:
    missing_constant = Formalization(
        predicates={PredicateDef("P", 1)},
        constants=set(),
        formulae=[Predicate("P", [Const("a")])],
        conclusion=[],
    )

    response = run_fml_in_z3(missing_constant, step_type="Constraint")

    assert response.status == "error"
    assert response.model is None
    assert response.error is not None
    assert response.error.startswith("KeyError:")


def test_run_fml_in_z3_resets_solver_by_default() -> None:
    contradictory = Formalization(
        predicates={PredicateDef("P", 1)},
        constants={Const("a")},
        formulae=[
            Predicate("P", [Const("a")]),
            Not(Predicate("P", [Const("a")])),
        ],
        conclusion=[],
    )
    satisfiable = Formalization(
        predicates={PredicateDef("P", 1)},
        constants={Const("a")},
        formulae=[Predicate("P", [Const("a")])],
        conclusion=[],
    )

    first = run_fml_in_z3(contradictory, step_type="Constraint")
    second = run_fml_in_z3(satisfiable, step_type="Constraint")

    assert first.status == "unsat"
    assert second.status == "sat"


def test_run_fml_in_z3_can_persist_solver_assertions() -> None:
    assert (
        run_fol_in_z3(
            unary_form(["P(a)"]),
            step_type="Constraint",
            persist_solver=True,
        ).status
        == "sat"
    )

    follow_up = run_fol_in_z3(
        unary_form(["Not(P(a))"]),
        step_type="Constraint",
        reset_on_start=False,
        persist_solver=True,
    )

    assert follow_up.status == "unsat"


def test_persisted_symbols_allow_later_incremental_parse() -> None:
    first = run_fol_in_z3(
        unary_form(["P(a)"]),
        step_type="Constraint",
        persist_predicates_and_constants=True,
    )
    predicates, constants = z3t._get_global_predicates_and_constants()

    assert first.status == "sat"
    assert predicates == {PredicateDef("P", 1)}
    assert constants == {Const("a")}

    follow_up = run_fol_in_z3(
        [StrFormalization(constraints=["P(a)"])],
        step_type="Constraint",
        reset_on_start=False,
    )

    assert follow_up.status == "sat"
    assert follow_up.error is None


def test_non_persisted_predicates_and_constants_are_cleared() -> None:
    response = run_fol_in_z3(
        unary_form(["P(a)"]),
        step_type="Constraint",
        persist_predicates_and_constants=False,
    )

    assert response.status == "sat"
    assert z3t._get_global_predicates_and_constants() == (set(), set())


def test_reset_on_start_clears_incremental_symbol_context() -> None:
    run_fol_in_z3(
        unary_form(["P(a)"]),
        step_type="Constraint",
        persist_predicates_and_constants=True,
    )

    response = run_fol_in_z3(
        [StrFormalization(constraints=["P(a)"])],
        step_type="Constraint",
        reset_on_start=True,
    )

    assert response.status == "error"
    assert response.error is not None
    assert "Unknown function symbol" in response.error


def test_check_implication_in_z3_with_constraints() -> None:
    stronger = unary_form(["P(a)", "Q(a)"])
    stronger[0].predicates = ["P(1)", "Q(1)"]
    weaker = [
        StrFormalization(
            predicates=["P(1)", "Q(1)"],
            constants=["a"],
            constraints=["P(a)"],
        )
    ]
    unrelated = [
        StrFormalization(
            predicates=["P(1)", "Q(1)"],
            constants=["a"],
            constraints=["Q(a)"],
        )
    ]

    assert check_implication_in_z3(stronger, weaker).status == "unsat"
    assert check_implication_in_z3(weaker, unrelated).status == "sat"


def test_check_implication_in_z3_compares_conclusions() -> None:
    p_conclusion = [
        StrFormalization(
            predicates=["P(1)", "Q(1)"],
            constants=["a"],
            conclusion=["P(a)"],
        )
    ]
    p_or_q_conclusion = [
        StrFormalization(
            predicates=["P(1)", "Q(1)"],
            constants=["a"],
            conclusion=["Or(P(a), Q(a))"],
        )
    ]

    assert (
        check_implication_in_z3(p_conclusion, p_or_q_conclusion).status
        == "unsat"
    )
    assert (
        check_implication_in_z3(p_or_q_conclusion, p_conclusion).status
        == "sat"
    )


def test_check_implication_in_z3_resets_previous_global_symbols() -> None:
    z3t._set_global_predicates_and_constants(
        {PredicateDef("Old", 1)}, {Const("o")}
    )

    response = check_implication_in_z3(
        unary_form(["P(a)"]), unary_form(["P(a)"])
    )
    predicates, constants = z3t._get_global_predicates_and_constants()

    assert response.status == "unsat"
    assert predicates == {PredicateDef("P", 1)}
    assert constants == {Const("a")}


def test_timeout_parameter_is_accepted_for_easy_queries() -> None:
    response = run_fol_in_z3(
        unary_form(["P(a)"]),
        step_type="Constraint",
        timeout_in_seconds=0.5,
    )

    assert response.status == "sat"
    assert response.error is None


def test_global_solver_helpers_initialize_and_reset_context() -> None:
    z3t._global_z3_solver = None
    z3t._global_z3_context = None

    solver, context = z3t._get_global_z3_solver()

    assert solver is z3t._global_z3_solver
    assert context is z3t._global_z3_context
    assert "__sort__" in context
    assert "z3" in context

    context["temporary"] = object()
    z3t._reset_global_z3_solver_and_context()
    _, reset_context = z3t._get_global_z3_solver()

    assert "temporary" not in reset_context


def test_run_fol_in_z3_parses_multi_block_declarations() -> None:
    response = run_fol_in_z3(
        [
            StrFormalization(predicates=["P(1)"], constants=["a"]),
            StrFormalization(
                predicates=["Q(1)"],
                constants=["b"],
                constraints=["P(a)", "Q(b)"],
                conclusion=["P(a)"],
            ),
        ],
        step_type="All",
    )

    assert response.status == "unsat"
    assert response.error is None


def test_run_fol_in_z3_handles_equality_reasoning() -> None:
    response = run_fml_in_z3(
        Formalization(
            predicates={PredicateDef("P", 1)},
            constants={Const("a"), Const("b")},
            formulae=[
                Equals(Const("a"), Const("b")),
                Predicate("P", [Const("a")]),
            ],
            conclusion=[Predicate("P", [Const("b")])],
        ),
        step_type="All",
    )

    assert response.status == "unsat"
    assert response.error is None
