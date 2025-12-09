import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, assert_never

import yaml
import z3  # type: ignore


@dataclass(frozen=True)
class BoundedVar:
    name: str
    sort: str | None = None


@dataclass(frozen=True)
class Const:
    name: str
    sort: str | None = None


type Term = BoundedVar | Const


@dataclass(frozen=True)
class Predicate:
    name: str
    args: List[Term]


@dataclass(frozen=True)
class Equals:
    left: Term
    right: Term


@dataclass(frozen=True)
class Not:
    formula: "Formula"


@dataclass(frozen=True)
class And:
    left: "Formula"
    right: "Formula"


@dataclass(frozen=True)
class Or:
    left: "Formula"
    right: "Formula"


@dataclass(frozen=True)
class Xor:
    left: "Formula"
    right: "Formula"


@dataclass(frozen=True)
class Implies:
    antecedent: "Formula"
    consequent: "Formula"


@dataclass(frozen=True)
class ForAll:
    variable: BoundedVar
    body: "Formula"


@dataclass(frozen=True)
class Exists:
    variable: BoundedVar
    body: "Formula"


@dataclass(frozen=True)
class PredicateDef:
    name: str
    arity: int
    arg_sorts: List[str] | None = None


type Formula = (
    Predicate | Not | And | Or | Xor | Equals | Implies | ForAll | Exists
)


class FOLParser:
    @staticmethod
    def parse(
        expr: str,
        predicates: Set[PredicateDef],
        constants: Set[str],
    ) -> Formula:
        """Parse an expression using Python's ast into Formula.

        Supported surface syntax (all as Python expressions):

        - Predicate applications:  Human(x), Mortal(Socrates)
        - Connectives:            Not(phi), And(phi, psi), Or(phi, psi),
                                  Xor(phi, psi), Implies(phi, psi)
        - Equality:               Equals(t1, t2)
        - Quantifiers:            ForAll(x, body), Exists(x, body)

        Variables vs constants are distinguished using `bound_vars` and the
        provided `constants` set.
        """

        node = ast.parse(expr, mode="eval").body
        return FOLParser._from_ast(
            node, predicates, constants, bound_vars=set()
        )

    @staticmethod
    def _from_ast(
        node: ast.AST,
        predicates: Set[PredicateDef],
        constants: Set[str],
        bound_vars: Set[str],
    ) -> Formula:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fun = node.func.id

            # Quantifiers: ForAll(x, body), Exists(x, body)
            if fun in {"ForAll", "Exists"}:
                if len(node.args) != 2:
                    raise ValueError(f"{fun} expects 2 arguments (var, body)")
                var_node, body_node = node.args
                if not isinstance(var_node, ast.Name):
                    raise ValueError(f"{fun} variable must be a simple name")
                var_name = var_node.id
                new_bound = set(bound_vars) | {var_name}
                body_formula = FOLParser._from_ast(
                    body_node, predicates, constants, new_bound
                )
                var = BoundedVar(var_name)
                return (
                    ForAll(var, body_formula)
                    if fun == "ForAll"
                    else Exists(var, body_formula)
                )

            # Negation: Not(phi)
            if fun == "Not":
                if len(node.args) != 1:
                    raise ValueError("Not expects 1 argument")
                sub = FOLParser._from_ast(
                    node.args[0], predicates, constants, bound_vars
                )
                return Not(sub)

            # Binary connectives and equality: And, Or, Xor, Implies, Equals
            if fun in {"And", "Or", "Xor", "Implies", "Equals"}:
                if len(node.args) != 2:
                    raise ValueError(f"{fun} expects 2 arguments")
                left_node, right_node = node.args
                if fun == "Equals":
                    left_term = FOLParser._term_from_ast(
                        left_node, constants, bound_vars
                    )
                    right_term = FOLParser._term_from_ast(
                        right_node, constants, bound_vars
                    )
                    return Equals(left_term, right_term)

                left_formula = FOLParser._from_ast(
                    left_node, predicates, constants, bound_vars
                )
                right_formula = FOLParser._from_ast(
                    right_node, predicates, constants, bound_vars
                )
                if fun == "And":
                    return And(left_formula, right_formula)
                if fun == "Or":
                    return Or(left_formula, right_formula)
                if fun == "Xor":
                    return Xor(left_formula, right_formula)
                if fun == "Implies":
                    return Implies(left_formula, right_formula)

            # Predicate application: P(t1, ..., tn)
            predicate_arity: Dict[str, int] = {
                p.name: p.arity for p in predicates
            }
            if fun in predicate_arity:
                expected_arity = predicate_arity[fun]
                if len(node.args) != expected_arity:
                    raise ValueError(
                        f"Predicate '{fun}' expected {expected_arity} "
                        f"arguments, got {len(node.args)}"
                    )
                args: List[Term] = [
                    FOLParser._term_from_ast(a, constants, bound_vars)
                    for a in node.args
                ]
                return Predicate(fun, args)

            raise ValueError(f"Unknown function symbol: {fun}")

        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    @staticmethod
    def _term_from_ast(
        node: ast.AST, constants: Set[str], bound_vars: Set[str]
    ) -> Term:
        if isinstance(node, ast.Name):
            if node.id in bound_vars:
                return BoundedVar(node.id)
            if node.id in constants:
                return Const(node.id)
            raise ValueError(f"Unknown symbol in term position: {node.id}")
        raise ValueError(f"Unsupported term node: {ast.dump(node)}")


class YamlListParser:
    @staticmethod
    def parse(
        yaml_text: str,
        previous_predicates: Set[PredicateDef] = set(),
        previous_constants: Set[str] = set(),
    ) -> Tuple[Set[PredicateDef], Set[str], List[Formula], List[Formula]]:
        data = yaml.safe_load(yaml_text)
        # Parse predicate declarations of the form "Name(Arity)"
        predicates: Set[PredicateDef] = set(previous_predicates)
        constants: Set[str] = set(previous_constants)

        if "Predicates" in data:
            for p in data.get("Predicates", []):
                if p is None:
                    continue
                if not isinstance(p, str):
                    raise TypeError(
                        f"Predicate must be a string, got {type(p)}: {p!r}"
                    )
                name, arity = p.split("(")
                predicates.add(
                    PredicateDef(name=name, arity=int(arity[:-1]))
                )  # remove trailing ')'

        if "Constants" in data:
            for c in data.get("Constants", []):
                if c is None:
                    continue
                if not isinstance(c, str):
                    raise TypeError(
                        f"Constant must be a string, got {type(c)}: {c!r}"
                    )
                constants.add(c)

        formulae: List[Formula] = []
        conclusion: List[Formula] = []

        if "Constraints" in data:
            formulae = [
                FOLParser.parse(r, predicates, constants)
                for r in data.get("Constraints", [])
                if r is not None
            ]

        if "Conclusion" in data:
            conclusion = [
                FOLParser.parse(r, predicates, constants)
                for r in data.get("Conclusion", [])
                if r is not None
            ]

        return predicates, constants, formulae, conclusion


class Z3Interpreter:
    @staticmethod
    def register_predicate(
        predicate: PredicateDef, context: Dict[str, Any]
    ) -> z3.FuncDeclRef:
        """Register a predicate in Z3 as an uninterpreted function."""
        if context.get(predicate.name) is not None:
            print(f"Predicate '{predicate.name}' already registered.")
            return context[predicate.name]

        obj_sort: z3.SortRef = context["__sort__"]
        arg_sorts = [obj_sort] * predicate.arity
        func = z3.Function(  # type: ignore
            predicate.name,
            *arg_sorts,
            z3.BoolSort(),  # type: ignore
        )
        context[predicate.name] = func
        return func

    @staticmethod
    def register_constant(
        constant: str, context: Dict[str, Any]
    ) -> z3.ExprRef:
        """Register a constant in Z3 as a z3 constant."""
        if context.get(constant) is not None:
            print(f"Constant '{constant}' already registered.")
            return context[constant]

        obj_sort: z3.SortRef = context["__sort__"]
        const = z3.Const(constant, obj_sort)  # type: ignore
        context[constant] = const
        return const

    @staticmethod
    def interpret(formula: Formula, context: Dict[str, Any]) -> z3.BoolRef:
        """Interpret a `Formula` into a z3 BoolRef.

        The `context` maps predicate names to uninterpreted z3 functions.
        All terms are of a single z3 sort, declared in `context["__sort__"]`.
        """

        def term_to_z3(t: Term) -> z3.ExprRef:
            match t:
                case BoundedVar(name=name):
                    return z3.Const(name, context["__sort__"])  # type: ignore
                case Const(name=name):
                    return context[name]
                case _:
                    assert_never(t)

        def go(f: Formula) -> z3.BoolRef:
            match f:
                case Predicate(name=name, args=args):
                    if name not in context:
                        raise KeyError(
                            f"Predicate '{name}' not found in context"
                        )
                    func = context[name]
                    z3_args = [term_to_z3(a) for a in args]
                    return func(*z3_args)

                case Equals(left=left, right=right):
                    return term_to_z3(left) == term_to_z3(right)  # type: ignore

                case Not(formula=sub):
                    return z3.Not(go(sub))  # type: ignore

                case And(left=left, right=right):
                    return z3.And(go(left), go(right))  # type: ignore

                case Or(left=left, right=right):
                    return z3.Or(go(left), go(right))  # type: ignore

                case Xor(left=left, right=right):
                    return z3.Xor(go(left), go(right))  # type: ignore

                case Implies(antecedent=antecedent, consequent=consequent):
                    return z3.Implies(go(antecedent), go(consequent))  # type: ignore

                case ForAll(variable, body=body):
                    v = term_to_z3(variable)
                    return z3.ForAll([v], go(body))  # type: ignore

                case Exists(variable, body=body):
                    v = term_to_z3(variable)
                    return z3.Exists([v], go(body))  # type: ignore

                case _:
                    assert_never(f)

        return go(formula)


def _example_usage() -> None:
    # --- [Usage] ---
    input = """
    Predicates:
    - Human(1)
    - Mortal(1)
    Constants:
    - Socrates
    Constraints:
    - Human(Socrates)
    - ForAll(x, Implies(Human(x), Mortal(x)))
    Conclusion:
    - Mortal(Socrates)    
    """
    predicates: Set[PredicateDef] = set()
    constants: Set[str] = set()
    predicates, constants, formulae, conclusion = YamlListParser.parse(
        input, predicates, constants
    )
    print("Parsed Predicates:", predicates)
    print("Parsed Constants:", constants)
    print("Parsed Formulae:", formulae)
    print("Parsed Conclusion:", conclusion)
    context: Dict[str, Any] = {"__sort__": z3.DeclareSort("Object")}  # type: ignore
    for p in predicates:
        Z3Interpreter.register_predicate(p, context)
    for c in constants:
        Z3Interpreter.register_constant(c, context)

    solver = z3.Solver()
    for f in formulae:
        z3_formula = Z3Interpreter.interpret(f, context)
        solver.add(z3_formula)  # type: ignore

    for q in conclusion:
        z3_conclusion = Z3Interpreter.interpret(Not(q), context)
        solver.add(z3_conclusion)  # type: ignore

    if solver.check() == z3.sat:  # type: ignore
        model = solver.model()
        print("Satisfiable. Model:")
        print(model)
    else:
        print("Unsatisfiable.")


if __name__ == "__main__":
    _example_usage()
