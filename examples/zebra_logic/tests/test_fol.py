import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import pytest
import z3  # type: ignore
from fol import (
    And,
    Const,
    Equals,
    FOLParser,
    ForAll,
    FormalizationParser,
    Not,
    Predicate,
    PredicateDef,
    StrFormalization,
    Var,
    Z3Interpreter,
    pretty_print,
)


def test_pretty_print_simple() -> None:
    f = And(Predicate("P", [Const("a")]), Not(Equals(Const("a"), Var("x"))))
    assert pretty_print(f) == "And(P(a), Not(Equals(a, x)))"


def test_parse_forall_binds_variable() -> None:
    preds = {PredicateDef("Human", 1)}
    consts = {Const("Socrates")}
    parsed = FOLParser.parse("ForAll(x, Human(x))", preds, consts)
    assert isinstance(parsed, ForAll)
    assert parsed.variable == Var("x")
    assert isinstance(parsed.body, Predicate)
    assert parsed.body.name == "Human"
    assert parsed.body.args == [Var("x")]


def test_parse_equals_and_not() -> None:
    preds = set()
    consts = {Const("a"), Const("b")}
    parsed = FOLParser.parse("Not(Equals(a, b))", preds, consts)
    assert isinstance(parsed, Not)
    assert isinstance(parsed.formula, Equals)
    assert parsed.formula.left == Const("a")
    assert parsed.formula.right == Const("b")


def test_parse_nary_and_is_right_associative() -> None:
    preds = {PredicateDef("P", 1), PredicateDef("Q", 1), PredicateDef("R", 1)}
    consts = set()
    parsed = FOLParser.parse("ForAll(x, And(P(x), Q(x), R(x)))", preds, consts)
    assert isinstance(parsed, ForAll)
    body = parsed.body
    assert isinstance(body, And)
    assert isinstance(body.left, Predicate)
    assert body.left.name == "P"
    assert isinstance(body.right, And)
    assert isinstance(body.right.left, Predicate)
    assert body.right.left.name == "Q"
    assert isinstance(body.right.right, Predicate)
    assert body.right.right.name == "R"


def test_parse_unknown_symbol_in_term_raises() -> None:
    preds = {PredicateDef("P", 1)}
    consts = set()
    with pytest.raises(ValueError, match="Unknown symbol in term position"):
        FOLParser.parse("P(x)", preds, consts)


def test_formalization_parser_conflicting_arity_raises() -> None:
    preds = {PredicateDef("P", 1)}
    form = StrFormalization(predicates=["P(2)"])
    with pytest.raises(ValueError, match="conflicting arity"):
        FormalizationParser.parse_multiple([form], previous_predicates=preds)


def test_formalization_parser_parse_multiple_merges() -> None:
    forms = [
        StrFormalization(
            predicates=["P(1)"],
            constants=["a"],
            constraints=["P(a)"],
        ),
        StrFormalization(
            predicates=["Q(1)"],
            constants=["b"],
            constraints=["Q(b)"],
            conclusion=["P(a)", "Q(b)"],
        ),
    ]
    parsed = FormalizationParser.parse_multiple(forms)
    assert {p.name for p in parsed.predicates} == {"P", "Q"}
    assert {c.name for c in parsed.constants} == {"a", "b"}
    assert len(parsed.formulae) == 2
    assert len(parsed.conclusion) == 2


def test_formalization_parser_reuses_previous_symbols() -> None:
    prev_preds = {PredicateDef("P", 1)}
    prev_consts = {Const("a")}
    forms = [
        StrFormalization(
            predicates=["P(1)"],
            constants=["a"],
            constraints=["P(a)"],
        )
    ]
    parsed = FormalizationParser.parse_multiple(
        forms,
        previous_predicates=prev_preds,
        previous_constants=prev_consts,
    )
    assert {p.name for p in parsed.predicates} == {"P"}
    assert {c.name for c in parsed.constants} == {"a"}


def test_z3_interpreter_missing_predicate_raises() -> None:
    context = {"__sort__": z3.DeclareSort("Object")}  # type: ignore
    formula = Predicate("P", [Const("a")])
    with pytest.raises(KeyError, match="Predicate 'P' not found"):
        Z3Interpreter.interpret(formula, context)


def test_z3_interpreter_registers_and_interprets() -> None:
    context = {"__sort__": z3.DeclareSort("Object")}  # type: ignore
    context = Z3Interpreter.register_predicate(PredicateDef("P", 1), context)
    context = Z3Interpreter.register_constant(Const("a"), context)
    formula = Predicate("P", [Const("a")])
    z3_expr = Z3Interpreter.interpret(formula, context)
    assert str(z3_expr) == "P(a)"
