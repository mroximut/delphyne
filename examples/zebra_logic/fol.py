import ast
from dataclasses import dataclass
from typing import Any

import z3  # type: ignore

type Sort = str


@dataclass(frozen=True)
class Var:
    name: str
    sort: Sort | None = None


@dataclass(frozen=True)
class Const:
    name: str
    sort: Sort | None = None

    def __eq__(self, other: object):
        if not isinstance(other, Const):
            return NotImplemented
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


type Atomic = Var | Const


@dataclass(frozen=True)
class Predicate:
    name: str
    args: list[Atomic]


@dataclass(frozen=True)
class Equals:
    left: Atomic
    right: Atomic


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
class Iff:
    left: "Formula"
    right: "Formula"


@dataclass(frozen=True)
class Implies:
    antecedent: "Formula"
    consequent: "Formula"


@dataclass(frozen=True)
class ForAll:
    variable: Var
    body: "Formula"


@dataclass(frozen=True)
class Exists:
    variable: Var
    body: "Formula"


@dataclass(frozen=True)
class PredicateDef:
    name: str
    arity: int
    arg_sorts: list[Sort] | None = None

    def __eq__(self, other: object):
        if not isinstance(other, PredicateDef):
            return NotImplemented
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


type Formula = (
    Predicate | Not | And | Or | Xor | Equals | Implies | ForAll | Exists | Iff
)


@dataclass
class StrFormalization:
    predicates: list[str] | None = None
    constants: list[str] | None = None
    constraints: list[str] | None = None
    conclusion: list[str] | None = None


@dataclass
class Formalization:
    predicates: set[PredicateDef]
    constants: set[Const]
    formulae: list[Formula]
    conclusion: list[Formula]

    def __add__(self, other: "Formalization"):
        return Formalization(
            predicates=self.predicates | other.predicates,
            constants=self.constants | other.constants,
            formulae=self.formulae + other.formulae,
            conclusion=self.conclusion + other.conclusion,
        )


def pretty_print(formula: Formula) -> str:
    match formula:
        case Predicate(name=name, args=args):
            args_str = ", ".join(pretty_print_arg(a) for a in args)
            return f"{name}({args_str})"
        case Equals(left=left, right=right):
            return (
                f"Equals({pretty_print_arg(left)}, {pretty_print_arg(right)})"
            )
        case Not(formula=sub):
            return f"Not({pretty_print(sub)})"
        case And(left=left, right=right):
            return f"And({pretty_print(left)}, {pretty_print(right)})"
        case Or(left=left, right=right):
            return f"Or({pretty_print(left)}, {pretty_print(right)})"
        case Xor(left=left, right=right):
            return f"Xor({pretty_print(left)}, {pretty_print(right)})"
        case Implies(antecedent=antecedent, consequent=consequent):
            return f"Implies({pretty_print(antecedent)}, {pretty_print(consequent)})"
        case ForAll(variable=variable, body=body):
            return f"ForAll({variable.name}, {pretty_print(body)})"
        case Exists(variable=variable, body=body):
            return f"Exists({variable.name}, {pretty_print(body)})"
        case Iff(left=left, right=right):
            return f"Iff({pretty_print(left)}, {pretty_print(right)})"


def pretty_print_arg(arg: Atomic) -> str:
    match arg:
        case Var(name=name):
            return name
        case Const(name=name):
            return name


class FOLParser:
    @staticmethod
    def parse(
        formula: str,
        predicates: set[PredicateDef],
        constants: set[Const],
    ) -> Formula:
        """Parse an expression using Python's ast into Formula.

        Supported surface syntax (all as Python expressions):

        - Predicate applications:  Human(x), Mortal(Socrates)
        - Connectives:            Not(phi), And(phi, psi), Or(phi, psi),
                                  Xor(phi, psi), Implies(phi, psi)
        - Equality:               Equals(t1, t2)
        - Quantifiers:            ForAll(x, body), Exists(x, body)

        And/Or accept 2+ arguments and are auto-nested into binary trees.
        Trailing unmatched ')' characters are auto-stripped before parsing.

        Variables vs constants are distinguished using the
        provided `constants` set.
        """

        # Auto-repair: balance unmatched parentheses
        formula = formula.strip()
        # If there are more closing parens than opening, remove trailing
        # ')' once and then re-add exactly the deficit required to balance.
        open_count = formula.count("(")
        close_count = formula.count(")")

        if close_count > open_count:
            formula = formula.rstrip(")")
            deficit = formula.count("(") - formula.count(")")
            if deficit > 0:
                formula = formula + ")" * deficit
        elif open_count > close_count:
            formula = formula + ")" * (open_count - close_count)

        node = ast.parse(formula, mode="eval").body
        return FOLParser._from_ast(
            node, predicates, constants, bound_vars=set(), source=formula
        )

    @staticmethod
    def _from_ast(
        node: ast.AST,
        predicates: set[PredicateDef],
        constants: set[Const],
        bound_vars: set[Var],
        source: str | None = None,
    ) -> Formula:
        # Provide a short source snippet helper for better error messages
        def _seg(n: ast.AST) -> str:
            try:
                return ast.get_source_segment(source or "", n) or ast.dump(n)
            except Exception:
                return ast.dump(n)

        if not isinstance(node, ast.Call):
            raise ValueError(
                f"Unsupported or disallowed node type {type(node).__name__}"
                + f" in expression: {_seg(node)}"
            )
        if not isinstance(node.func, ast.Name):
            raise ValueError(
                f"Unsupported function call form: {_seg(node.func)} in"
                + f" {source or '<expr>'}"
            )
        if node.keywords:
            raise ValueError(
                f"Keyword arguments are not supported: {_seg(node)}"
            )
        fun = node.func.id

        match fun:
            # Quantifiers: ForAll(x, body), Exists(x, body)
            case "ForAll" | "Exists":
                if len(node.args) != 2:
                    raise ValueError(
                        f"{fun} expects 2 arguments (var, body): {_seg(node)}"
                    )
                var_node, body_node = node.args
                if not isinstance(var_node, ast.Name):
                    raise ValueError(
                        f"{fun} variable must be a simple name: {_seg(var_node)}"
                    )
                var_name = var_node.id
                new_bound = set(bound_vars) | {Var(name=var_name)}
                body_formula = FOLParser._from_ast(
                    body_node, predicates, constants, new_bound, source=source
                )
                var = Var(var_name)
                return (
                    ForAll(var, body_formula)
                    if fun == "ForAll"
                    else Exists(var, body_formula)
                )

            # Negation: Not(phi)
            case "Not":
                if len(node.args) != 1:
                    raise ValueError(f"Not expects 1 argument: {_seg(node)}")
                sub = FOLParser._from_ast(
                    node.args[0],
                    predicates,
                    constants,
                    bound_vars,
                    source=source,
                )
                return Not(sub)

            # N-ary And/Or: accept 2+ arguments, fold into binary tree
            case "And" | "Or":
                if len(node.args) < 2:
                    raise ValueError(
                        f"{fun} expects at least 2 arguments: {_seg(node)}"
                    )
                sub_formulas = [
                    FOLParser._from_ast(
                        a, predicates, constants, bound_vars, source=source
                    )
                    for a in node.args
                ]
                op = And if fun == "And" else Or
                result = sub_formulas[-1]
                for sf in reversed(sub_formulas[:-1]):
                    result = op(sf, result)
                return result

            # Binary connectives and equality: Xor, Implies, Equals, Iff
            case "Xor" | "Implies" | "Equals" | "Iff":
                if len(node.args) != 2:
                    raise ValueError(
                        f"{fun} expects exactly 2 arguments: {_seg(node)}"
                    )
                left_node, right_node = node.args
                if fun == "Equals":
                    left_term = FOLParser._term_from_ast(
                        left_node, constants, bound_vars, source=source
                    )
                    right_term = FOLParser._term_from_ast(
                        right_node, constants, bound_vars, source=source
                    )
                    return Equals(left_term, right_term)
                else:
                    left_formula = FOLParser._from_ast(
                        left_node,
                        predicates,
                        constants,
                        bound_vars,
                        source=source,
                    )
                    right_formula = FOLParser._from_ast(
                        right_node,
                        predicates,
                        constants,
                        bound_vars,
                        source=source,
                    )
                    if fun == "Xor":
                        return Xor(left_formula, right_formula)
                    elif fun == "Implies":
                        return Implies(left_formula, right_formula)
                    else:  # fun == "Iff"
                        return Iff(left_formula, right_formula)

            # Predicate application: P(t1, ..., tn)
            case _ if fun in (preds := {p.name: p.arity for p in predicates}):
                expected_arity = preds[fun]
                if len(node.args) != expected_arity:
                    raise ValueError(
                        f"Predicate '{fun}' expected {expected_arity} arguments,"
                        + f" got {len(node.args)}: {_seg(node)}"
                    )
                args: list[Atomic] = [
                    FOLParser._term_from_ast(
                        a, constants, bound_vars, source=source
                    )
                    for a in node.args
                ]
                return Predicate(fun, args)

            case _:
                raise ValueError(f"Unknown function symbol: {fun}")

    @staticmethod
    def _term_from_ast(
        node: ast.AST,
        constants: set[Const],
        bound_vars: set[Var],
        source: str | None = None,
    ) -> Atomic:
        def _seg(n: ast.AST) -> str:
            try:
                return ast.get_source_segment(source or "", n) or ast.dump(n)
            except Exception:
                return ast.dump(n)

        # Only simple names are allowed in term position.
        if isinstance(node, ast.Name):
            if node.id in [v.name for v in bound_vars]:
                return Var(node.id)
            if node.id in [c.name for c in constants]:
                return Const(node.id)
            raise ValueError(
                f"Unknown symbol in term position: {node.id} (at {_seg(node)})"
            )
        # Reject other AST nodes in term position with context
        raise ValueError(f"Unsupported term node: {_seg(node)}")


class FormalizationParser:
    @staticmethod
    def parse_multiple(
        formalizations: list[StrFormalization],
        previous_predicates: set[PredicateDef] = set(),
        previous_constants: set[Const] = set(),
    ) -> Formalization:
        ret = Formalization(previous_predicates, previous_constants, [], [])
        for formalization_str in formalizations:
            new_formalization = FormalizationParser._parse(
                formalization_str,
                ret.predicates,
                ret.constants,
            )
            ret = ret + new_formalization
        return ret

    @staticmethod
    def _parse(
        formalization_str: StrFormalization,
        previous_predicates: set[PredicateDef] = set(),
        previous_constants: set[Const] = set(),
    ) -> Formalization:
        # Parse predicate declarations of the form "Name(Arity)"
        new_predicates: set[PredicateDef] = set()
        new_constants: set[Const] = set()

        if formalization_str.predicates is not None:
            for p in formalization_str.predicates:
                p_name, p_arity_raw = p.split("(")
                p_arity = int(p_arity_raw.rstrip(")"))  # remove trailing ')'
                for prev in previous_predicates:
                    if prev.name == p_name and prev.arity != p_arity:
                        raise ValueError(
                            f"Predicate '{p_name}' already declared with "
                            f"arity {prev.arity}, got conflicting arity "
                            f"{p_arity} in declaration '{p}'"
                        )
                    elif prev.name == p_name:
                        break
                else:
                    new_predicates.add(
                        PredicateDef(name=p_name, arity=p_arity)
                    )

        if formalization_str.constants is not None:
            for c in formalization_str.constants:
                if c not in [const.name for const in previous_constants]:
                    new_constants.add(Const(name=c))

        predicates = previous_predicates | new_predicates
        constants = previous_constants | new_constants
        formulae: list[Formula] = []
        conclusion: list[Formula] = []

        if (
            (cons := formalization_str.constraints) is not None
            and cons != []
            and cons != ["None"]
        ):
            formulae = [
                FOLParser.parse(fml, predicates, constants) for fml in cons
            ]

        if (
            (conc := formalization_str.conclusion) is not None
            and conc != []
            and conc != ["None"]
        ):
            conclusion = [
                FOLParser.parse(fml, predicates, constants) for fml in conc
            ]

        return Formalization(
            predicates=predicates,
            constants=constants,
            formulae=formulae,
            conclusion=conclusion,
        )


class Z3Interpreter:
    @staticmethod
    def register_predicate(
        predicate: PredicateDef, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Register a predicate in Z3 as an uninterpreted function."""
        if predicate.name in context:
            return context  # already registered
        obj_sort: z3.SortRef = context["__sort__"]
        arg_sorts = [obj_sort] * predicate.arity
        func = z3.Function(  # type: ignore
            predicate.name,
            *arg_sorts,
            z3.BoolSort(),  # type: ignore
        )
        context[predicate.name] = func
        return context

    @staticmethod
    def register_constant(
        constant: Const, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Register a constant in Z3 as a z3 constant."""
        if constant.name in context:
            return context  # already registered
        obj_sort: z3.SortRef = context["__sort__"]
        const = z3.Const(constant.name, obj_sort)  # type: ignore
        context[constant.name] = const
        return context

    @staticmethod
    def interpret(formula: Formula, context: dict[str, Any]) -> z3.BoolRef:
        """Interpret a `Formula` into a z3 BoolRef.

        The `context` maps predicate names to uninterpreted z3 functions.
        All terms are of a single z3 sort, declared in `context["__sort__"]`.
        """

        def term_to_z3(t: Atomic) -> z3.ExprRef:
            match t:
                case Var(name=name):
                    return z3.Const(name, context["__sort__"])  # type: ignore
                case Const(name=name):
                    return context[name]

        def go(fml: Formula) -> z3.BoolRef:
            match fml:
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

                case Iff(left=left, right=right):
                    return go(left) == go(right)  # type: ignore

        return go(formula)
