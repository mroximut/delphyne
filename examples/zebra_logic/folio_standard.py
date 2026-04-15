import ast
from dataclasses import dataclass

from z3_tools import run_fol_in_z3

import delphyne as dp
from delphyne import (
    Branch,
    Compute,
    Fail,
    Strategy,
    ensure_compatible,
    strategy,
)


@dataclass
class CheckDeduction(dp.Query[str]):
    """
    Does the conclusion logically follow from the premises?

    The answer "True" means the conclusion follows from the premises.
    The answer "False" means the conclusion does NOT follow from the premises.
    The answer "Unknown" means it cannot be determined whether the conclusion
    follows from the premises.
    Answer in a triple backtick code block with the language set to "text".
    The answer should be a single word: "True", "False", or "Unknown".
    """

    sentences: list[str]

    __parser__ = dp.last_code_block.trim


@dataclass
class CheckZ3(dp.AbstractTool[str]):
    """
    Check the satisfiability of a list of first-order logic formulas
    using the Z3 SMT solver. Each formula should be a YAML-formatted
    FOL formalization string. The tool returns the solver's verdict
    (sat, unsat, unknown, or error) along with the model or error
    details.
    """

    formalizations: list[str]


@dataclass
class DeductionAnswer:
    answer: str


@dataclass
class CheckDeductionWithZ3(dp.Query[dp.Response[DeductionAnswer, CheckZ3]]):
    sentences: list[str]
    prefix: dp.AnswerPrefix = ()

    __parser__ = dp.final_tool_call.response


def _parse_bool(result_str: str) -> bool | None:
    return (
        True
        if result_str == "True"
        else False
        if result_str == "False"
        else None
    )


@dataclass
class FolioOnlyAskIP:
    check_deduction: dp.PromptingPolicy


@strategy
def _run_z3_tool(tool_call: CheckZ3) -> Strategy[Compute, object, str]:
    response = yield from dp.compute(run_fol_in_z3)(
        tool_call.formalizations,
        step_type="All",
        permanently=False,
    )
    parts = [f"Status: {response.status}"]
    if response.model is not None:
        parts.append(f"Model/Core: {response.model}")
    if response.error is not None:
        parts.append(f"Error: {response.error}")
    return "\n".join(parts)


@strategy
def folio_only_ask(
    puzzle: str,
) -> Strategy[Branch | Fail, FolioOnlyAskIP, bool | None]:
    sentences = puzzle.strip().split("\n")
    yield from dp.ensure(len(sentences) > 0, "The puzzle is empty.")
    result = yield from dp.branch(
        CheckDeduction(sentences=sentences).using(
            lambda p: p.check_deduction, FolioOnlyAskIP
        )
    )
    return _parse_bool(result)


@strategy
def folio_formalization_agent(
    puzzle: str,
) -> Strategy[Branch | Fail, FolioOnlyAskIP, bool | None]:
    sentences = puzzle.strip().split("\n")
    yield from dp.ensure(len(sentences) > 0, "The puzzle is empty.")
    result = yield from dp.interact(
        step=lambda prefix, _: CheckDeductionWithZ3(
            sentences=sentences, prefix=prefix
        ).using(lambda p: p.check_deduction, FolioOnlyAskIP),
        process=lambda ans, _: dp.const_space(ans),
        tools={
            CheckZ3: lambda call: _run_z3_tool(call).using(dp.just_compute)
        },
    )
    return _parse_bool(result.answer)


@ensure_compatible(folio_only_ask)
@ensure_compatible(folio_formalization_agent)
def folio_ask_policy(
    model_name: dp.StandardModelName = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    num_requests: int = 10,
) -> dp.Policy[Branch | Fail, FolioOnlyAskIP]:
    budget = dp.BudgetLimit({dp.NUM_REQUESTS: num_requests})
    model = dp.standard_model(
        model_name, {"reasoning_effort": reasoning_effort}
    )
    return dp.with_budget(budget) @ dp.dfs() & FolioOnlyAskIP(
        check_deduction=dp.few_shot(model=model)
    )


# --- Z3 constraint solver agent ---

# Whitelisted z3 names available in eval() expressions.
_Z3_ALLOWED_NAMES: set[str] = {
    # Sorts
    "IntSort",
    "BoolSort",
    "RealSort",
    "StringSort",
    "BitVecSort",
    "ArraySort",
    "SetSort",
    "DeclareSort",
    # Variable / constant constructors
    "Int",
    "Ints",
    "Bool",
    "Bools",
    "Real",
    "Reals",
    "BitVec",
    "BitVecs",
    "String",
    "Strings",
    "Const",
    "Consts",
    "FreshInt",
    "FreshBool",
    "FreshReal",
    "FreshConst",
    "Array",
    "K",
    # Literal values
    "IntVal",
    "RealVal",
    "BoolVal",
    "BitVecVal",
    "StringVal",
    "RatVal",
    # Function declarations
    "Function",
    # Logical connectives
    "And",
    "Or",
    "Not",
    "Implies",
    "Xor",
    "If",
    "Distinct",
    # Quantifiers
    "ForAll",
    "Exists",
    "Lambda",
    # Arithmetic helpers
    "Sum",
    "Product",
    # Enumerations / datatypes
    "EnumSort",
    "Datatype",
    # Set operations
    "EmptySet",
    "FullSet",
    "SetAdd",
    "SetDel",
    "SetUnion",
    "SetIntersect",
    "SetComplement",
    "IsMember",
    "IsSubset",
    # String operations
    "Length",
    "SubString",
    "IndexOf",
    "Contains",
    "PrefixOf",
    "SuffixOf",
    "Replace",
    "Concat",
    "Re",
    "InRe",
    "Union",
    "Star",
    "Plus",
    "Option",
    # Misc
    "simplify",
    "substitute",
}

# Whitelisted attribute names allowed on z3 objects.
_Z3_ALLOWED_ATTRS: set[str] = {
    "sort",
    "decl",
    "sexpr",
    "as_long",
    "as_fraction",
    "num_args",
    "arg",
    "children",
    "arity",
    "domain",
    "range",
    "name",
    "accessor",
    "constructor",
    "recognizer",
    "num_constructors",
    "declare",
    "create",
}


@dataclass
class Z3Declaration:
    """A named Z3 variable or function declaration."""

    name: str
    expr: str


@dataclass
class RunZ3Solver(dp.AbstractTool[str]):
    """
    Check satisfiability of constraints using the Z3 SMT solver.

    `declarations`: list of Z3 variable/function declarations. Each has a
      `name` (the variable name) and `expr` (a Z3 constructor expression).
      Example: [{"name": "x", "expr": "Int('x')"},
               {"name": "f", "expr": "Function('f', IntSort(), BoolSort())"}]

    `constraints`: list of Z3 boolean expressions as strings.
      Example: ["x + y > 5", "x < 10", "f(x) == True"]

    Available constructors: Int, Ints, Bool, Bools, Real, Reals,
      Const, Function, DeclareSort, EnumSort, Datatype, Array, ...
    Available functions: And, Or, Not, Implies, Xor, If, Distinct,
      ForAll, Exists, Lambda, Sum, Product, ...
    Available sorts: IntSort, BoolSort, RealSort, StringSort, ...

    The solver result (sat/unsat/unknown) and model are returned.
    """

    declarations: list[Z3Declaration]
    constraints: list[str]


def _validate_z3_expr(expr: str) -> str | None:
    """Validate a single expression string. Returns error or None."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return f"SyntaxError in '{expr}': {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr not in _Z3_ALLOWED_ATTRS:
                return (
                    f"Attribute '.{node.attr}' is not allowed. "
                    f"Allowed: {sorted(_Z3_ALLOWED_ATTRS)}"
                )
    return None


def _build_z3_namespace() -> dict[str, object]:
    """Build a restricted namespace with only whitelisted z3 names."""
    import z3  # type: ignore

    ns: dict[str, object] = {"__builtins__": {}}
    for name in _Z3_ALLOWED_NAMES:
        obj = getattr(z3, name, None)
        if obj is not None:
            ns[name] = obj
    # Python builtins needed for building expressions
    ns["True"] = True
    ns["False"] = False
    ns["None"] = None
    ns["range"] = range
    ns["len"] = len
    return ns


def _run_z3_solver(tool_call: RunZ3Solver) -> str:
    import z3  # type: ignore

    ns = _build_z3_namespace()

    # Evaluate declarations and add them to the namespace
    for decl in tool_call.declarations:
        if not decl.name.isidentifier() or decl.name.startswith("_"):
            return f"Error: Invalid variable name '{decl.name}'."
        err = _validate_z3_expr(decl.expr)
        if err is not None:
            return f"Error in declaration '{decl.name}': {err}"
        try:
            ns[decl.name] = eval(decl.expr, ns)  # noqa: S307
        except Exception as exc:
            return f"Error in declaration '{decl.name}': {type(exc).__name__}: {exc}"

    # Evaluate constraints and add to solver
    solver = z3.Solver()
    for i, constraint in enumerate(tool_call.constraints):
        err = _validate_z3_expr(constraint)
        if err is not None:
            return f"Error in constraint {i}: {err}"
        try:
            c = eval(constraint, ns)  # noqa: S307
        except Exception as exc:
            return f"Error in constraint {i}: {type(exc).__name__}: {exc}"
        solver.add(c)  # type: ignore

    # Check and format result
    result = solver.check()  # type: ignore
    parts = [f"Status: {result}"]
    if result == z3.sat:
        model = solver.model()
        parts.append(f"Model: {model}")
    elif result == z3.unsat:
        # Try to get unsat core if tracking was enabled
        pass
    return "\n".join(parts)


@dataclass
class CheckDeductionZ3Code(
    dp.Query[dp.Response[DeductionAnswer, RunZ3Solver]]
):
    sentences: list[str]
    prefix: dp.AnswerPrefix = ()

    __parser__ = dp.final_tool_call.response


@dataclass
class FolioZ3AgentIP:
    check_deduction: dp.PromptingPolicy


@strategy
def folio_z3_agent(
    puzzle: str,
) -> Strategy[Branch | Fail, FolioZ3AgentIP, bool | None]:
    sentences = puzzle.strip().split("\n")
    yield from dp.ensure(len(sentences) > 0, "The puzzle is empty.")
    result = yield from dp.interact(
        step=lambda prefix, _: CheckDeductionZ3Code(
            sentences=sentences, prefix=prefix
        ).using(lambda p: p.check_deduction, FolioZ3AgentIP),
        process=lambda ans, _: dp.const_space(ans),
        tools={RunZ3Solver: lambda call: dp.const_space(_run_z3_solver(call))},
    )
    return _parse_bool(result.answer)


@ensure_compatible(folio_z3_agent)
def folio_z3_agent_policy(
    model_name: dp.StandardModelName = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "medium",
    num_requests: int = 10,
) -> dp.Policy[Branch | Fail, FolioZ3AgentIP]:
    budget = dp.BudgetLimit({dp.NUM_REQUESTS: num_requests})
    model = dp.standard_model(
        model_name, {"reasoning_effort": reasoning_effort}
    )
    return dp.with_budget(budget) @ dp.dfs() & FolioZ3AgentIP(
        check_deduction=dp.few_shot(model=model)
    )
