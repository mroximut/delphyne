# pyright: basic
import os
import sys

import pytest
import z3

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from fol import (  # noqa: E402
    And,
    Const,
    Equals,
    Exists,
    FOLParser,
    ForAll,
    Formalization,
    FormalizationParser,
    Iff,
    Implies,
    Not,
    Or,
    Predicate,
    PredicateDef,
    StrFormalization,
    Var,
    Xor,
    Z3Interpreter,
    pretty_print,
    pretty_print_arg,
)


def test_const_and_predicate_def_identity_is_name_based():
    assert Const("a", sort="Person") == Const("a", sort="Place")
    assert len({Const("a", sort="Person"), Const("a", sort="Place")}) == 1

    assert PredicateDef("P", 1) == PredicateDef("P", 2)
    assert len({PredicateDef("P", 1), PredicateDef("P", 2)}) == 1


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        (Predicate("P", [Const("a"), Var("x")]), "P(a, x)"),
        (Equals(Const("a"), Var("x")), "Equals(a, x)"),
        (Not(Predicate("P", [Const("a")])), "Not(P(a))"),
        (
            And(Predicate("P", [Const("a")]), Predicate("Q", [Const("b")])),
            "And(P(a), Q(b))",
        ),
        (
            Or(Predicate("P", [Const("a")]), Predicate("Q", [Const("b")])),
            "Or(P(a), Q(b))",
        ),
        (
            Xor(Predicate("P", [Const("a")]), Predicate("Q", [Const("b")])),
            "Xor(P(a), Q(b))",
        ),
        (
            Implies(
                Predicate("P", [Const("a")]), Predicate("Q", [Const("a")])
            ),
            "Implies(P(a), Q(a))",
        ),
        (
            Iff(Predicate("P", [Const("a")]), Predicate("Q", [Const("a")])),
            "Iff(P(a), Q(a))",
        ),
        (ForAll(Var("x"), Predicate("P", [Var("x")])), "ForAll(x, P(x))"),
        (Exists(Var("x"), Predicate("P", [Var("x")])), "Exists(x, P(x))"),
    ],
)
def test_pretty_print_covers_formula_variants(formula, expected: str):
    assert pretty_print(formula) == expected


def test_pretty_print_arg_handles_vars_and_constants():
    assert pretty_print_arg(Var("x")) == "x"
    assert pretty_print_arg(Const("Socrates")) == "Socrates"


def test_formalization_add_merges_symbols_and_concatenates_formulae():
    left = Formalization(
        predicates={PredicateDef("P", 1)},
        constants={Const("a")},
        formulae=[Predicate("P", [Const("a")])],
        conclusion=[],
    )
    right = Formalization(
        predicates={PredicateDef("Q", 1)},
        constants={Const("b")},
        formulae=[],
        conclusion=[Predicate("Q", [Const("b")])],
    )

    combined = left + right

    assert {p.name for p in combined.predicates} == {"P", "Q"}
    assert {c.name for c in combined.constants} == {"a", "b"}
    assert combined.formulae == left.formulae
    assert combined.conclusion == right.conclusion


def test_parse_nested_quantifiers_and_connectives():
    preds = {
        PredicateDef("Human", 1),
        PredicateDef("Mortal", 1),
        PredicateDef("Friend", 2),
    }
    consts = {Const("Socrates")}

    parsed = FOLParser.parse(
        "ForAll(x, Implies(Human(x), Exists(y, And(Friend(x, y), "
        "Or(Mortal(y), Equals(y, Socrates))))))",
        preds,
        consts,
    )

    assert isinstance(parsed, ForAll)
    assert parsed.variable == Var("x")
    assert isinstance(parsed.body, Implies)
    assert isinstance(parsed.body.consequent, Exists)
    exists_body = parsed.body.consequent.body
    assert isinstance(exists_body, And)
    assert exists_body.left == Predicate("Friend", [Var("x"), Var("y")])
    assert isinstance(exists_body.right, Or)
    assert exists_body.right.right == Equals(Var("y"), Const("Socrates"))


def test_parse_nary_and_or_are_right_associative_binary_trees():
    preds = {PredicateDef("P", 1), PredicateDef("Q", 1), PredicateDef("R", 1)}
    parsed = FOLParser.parse("ForAll(x, And(P(x), Q(x), R(x)))", preds, set())

    assert parsed == ForAll(
        Var("x"),
        And(
            Predicate("P", [Var("x")]),
            And(Predicate("Q", [Var("x")]), Predicate("R", [Var("x")])),
        ),
    )

    parsed_or = FOLParser.parse(
        "ForAll(x, Or(P(x), Q(x), R(x)))", preds, set()
    )
    assert parsed_or == ForAll(
        Var("x"),
        Or(
            Predicate("P", [Var("x")]),
            Or(Predicate("Q", [Var("x")]), Predicate("R", [Var("x")])),
        ),
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "Xor(P(a), Q(a))",
            Xor(Predicate("P", [Const("a")]), Predicate("Q", [Const("a")])),
        ),
        (
            "Iff(P(a), Q(a))",
            Iff(Predicate("P", [Const("a")]), Predicate("Q", [Const("a")])),
        ),
        ("Not(Equals(a, b))", Not(Equals(Const("a"), Const("b")))),
    ],
)
def test_parse_binary_special_forms(source: str, expected: object):
    preds = {PredicateDef("P", 1), PredicateDef("Q", 1)}
    consts = {Const("a"), Const("b")}

    assert FOLParser.parse(source, preds, consts) == expected


@pytest.mark.parametrize(
    "source",
    [
        "P(a",
        "P(a))",
        "And(P(a), Q(a)",
        "And(P(a), Q(a)))",
    ],
)
def test_parse_repairs_trailing_parenthesis_imbalance(source: str):
    preds = {PredicateDef("P", 1), PredicateDef("Q", 1)}
    consts = {Const("a")}

    parsed = FOLParser.parse(source, preds, consts)

    assert isinstance(parsed, Predicate | And)


@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("x", "Unsupported or disallowed node type"),
        ("P(a=1)", "Keyword arguments are not supported"),
        ("ForAll(P(a), P(a))", "variable must be a simple name"),
        ("ForAll(x)", "expects 2 arguments"),
        ("Not(P(a), Q(a))", "Not expects 1 argument"),
        ("And(P(a))", "And expects at least 2 arguments"),
        ("Xor(P(a))", "expects exactly 2 arguments"),
        ("P(a, b)", "expected 1 arguments"),
        ("Unknown(a)", "Unknown function symbol"),
        ("P(x)", "Unknown symbol in term position"),
        ("P(f(a))", "Unsupported term node"),
    ],
)
def test_parse_rejects_invalid_surface_syntax(source: str, match: str):
    preds = {PredicateDef("P", 1), PredicateDef("Q", 1)}
    consts = {Const("a"), Const("b")}

    with pytest.raises(ValueError, match=match):
        FOLParser.parse(source, preds, consts)


def test_formalization_parser_parses_and_merges_multiple_blocks():
    parsed = FormalizationParser.parse_multiple(
        [
            StrFormalization(
                predicates=["P(1)"],
                constants=["a"],
                constraints=["P(a)"],
                conclusion=["P(a)"],
            ),
            StrFormalization(
                predicates=["Q(2)"],
                constants=["b"],
                constraints=["Q(a, b)"],
                conclusion=["None"],
            ),
            StrFormalization(constraints=[], conclusion=[]),
        ]
    )

    assert {p.name: p.arity for p in parsed.predicates} == {"P": 1, "Q": 2}
    assert {c.name for c in parsed.constants} == {"a", "b"}
    assert parsed.formulae == [
        Predicate("P", [Const("a")]),
        Predicate("Q", [Const("a"), Const("b")]),
    ]
    assert parsed.conclusion == [Predicate("P", [Const("a")])]


def test_formalization_parser_reuses_symbols_and_detects_conflicts():
    previous_predicates = {PredicateDef("P", 1)}
    previous_constants = {Const("a")}

    parsed = FormalizationParser.parse_multiple(
        [StrFormalization(predicates=["P(1)"], constants=["a"])],
        previous_predicates=previous_predicates,
        previous_constants=previous_constants,
    )
    assert parsed.predicates == previous_predicates
    assert parsed.constants == previous_constants

    with pytest.raises(ValueError, match="conflicting arity"):
        FormalizationParser.parse_multiple(
            [StrFormalization(predicates=["P(2)"])],
            previous_predicates=previous_predicates,
        )


def test_z3_interpreter_registers_symbols_idempotently():
    context: dict[str, object] = {"__sort__": z3.DeclareSort("Object")}

    context = Z3Interpreter.register_predicate(PredicateDef("P", 1), context)
    first_predicate = context["P"]
    context = Z3Interpreter.register_predicate(PredicateDef("P", 1), context)
    assert context["P"] is first_predicate

    context = Z3Interpreter.register_constant(Const("a"), context)
    first_constant = context["a"]
    context = Z3Interpreter.register_constant(Const("a"), context)
    assert context["a"] is first_constant


def test_z3_interpreter_translates_formulae_into_solver_semantics():
    context: dict[str, object] = {"__sort__": z3.DeclareSort("Object")}
    for predicate in [
        PredicateDef("Human", 1),
        PredicateDef("Mortal", 1),
        PredicateDef("Likes", 2),
    ]:
        context = Z3Interpreter.register_predicate(predicate, context)
    context = Z3Interpreter.register_constant(Const("Socrates"), context)

    solver = z3.Solver()
    solver.add(
        Z3Interpreter.interpret(
            Predicate("Human", [Const("Socrates")]), context
        )
    )
    solver.add(
        Z3Interpreter.interpret(
            ForAll(
                Var("x"),
                Implies(
                    Predicate("Human", [Var("x")]),
                    Predicate("Mortal", [Var("x")]),
                ),
            ),
            context,
        )
    )
    solver.add(
        z3.Not(
            Z3Interpreter.interpret(
                Predicate("Mortal", [Const("Socrates")]), context
            )
        )
    )

    assert solver.check() == z3.unsat


@pytest.mark.parametrize(
    "formula",
    [
        And(Predicate("P", [Const("a")]), Predicate("Q", [Const("a")])),
        Or(Predicate("P", [Const("a")]), Predicate("Q", [Const("a")])),
        Xor(Predicate("P", [Const("a")]), Predicate("Q", [Const("a")])),
        Iff(Predicate("P", [Const("a")]), Predicate("Q", [Const("a")])),
        Exists(Var("x"), Predicate("P", [Var("x")])),
        Equals(Const("a"), Const("a")),
    ],
)
def test_z3_interpreter_accepts_all_formula_variants(formula):
    context: dict[str, object] = {"__sort__": z3.DeclareSort("Object")}
    for predicate in [PredicateDef("P", 1), PredicateDef("Q", 1)]:
        context = Z3Interpreter.register_predicate(predicate, context)
    context = Z3Interpreter.register_constant(Const("a"), context)

    translated = Z3Interpreter.interpret(formula, context)

    assert z3.is_bool(translated)


def test_z3_interpreter_reports_missing_context_entries():
    context: dict[str, object] = {"__sort__": z3.DeclareSort("Object")}

    with pytest.raises(KeyError, match="Predicate 'P' not found"):
        Z3Interpreter.interpret(Predicate("P", [Const("a")]), context)

    context = Z3Interpreter.register_predicate(PredicateDef("P", 1), context)
    with pytest.raises(KeyError):
        Z3Interpreter.interpret(Predicate("P", [Const("a")]), context)
