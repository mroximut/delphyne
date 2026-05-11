import ast
from dataclasses import dataclass

import fol
from folio_oneshot import APIType
from z3_tools import Z3Response, run_fol_in_z3

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
class DeductionAnswer:
    answer: bool | None


@dataclass
class CheckDeduction(dp.Query[DeductionAnswer]):
    """
    Does the conclusion logically follow from the premises?

    The answer `True` means the conclusion follows from the premises.
    The answer `False` means the conclusion does NOT follow from the premises.
    The answer `None` means it cannot be determined whether the conclusion
    follows from the premises.
    """

    sentences: list[str]

    __parser__ = dp.structured


@dataclass
class CheckZ3(dp.AbstractTool[Z3Response]):
    """
    Check the satisfiability of a list of first-order logic formulas
    using the Z3 SMT solver.
    The tool returns the solver's verdict
    (sat, unsat, unknown, or error) along with the model or error
    details.
    """

    formalization: fol.StrFormalization


@dataclass
class CheckDeductionWithZ3(dp.Query[dp.Response[DeductionAnswer, CheckZ3]]):
    sentences: list[str]
    prefix: dp.AnswerPrefix = ()

    __parser__ = dp.final_tool_call.response


@dataclass
class FolioAskIP:
    check_deduction: dp.PromptingPolicy


@strategy
def _run_z3_tool(tool_call: CheckZ3) -> Strategy[Compute, object, Z3Response]:
    ret = yield from dp.compute(run_fol_in_z3)(
        [tool_call.formalization], step_type="All"
    )
    return ret


@strategy
def folio_only_ask(
    puzzle: str,
) -> Strategy[Branch | Fail, FolioAskIP, bool | None]:
    sentences = puzzle.strip().split("\n")
    yield from dp.ensure(len(sentences) > 0, "The puzzle is empty.")
    result = yield from dp.branch(
        CheckDeduction(sentences=sentences).using(
            lambda p: p.check_deduction, FolioAskIP
        )
    )
    return result.answer


@strategy
def folio_formalization_agent(
    puzzle: str,
) -> Strategy[Branch | Fail, FolioAskIP, bool | None]:
    sentences = puzzle.strip().split("\n")
    yield from dp.ensure(len(sentences) > 0, "The puzzle is empty.")
    result = yield from dp.interact(
        step=lambda prefix, _: CheckDeductionWithZ3(
            sentences=sentences, prefix=prefix
        ).using(lambda p: p.check_deduction, FolioAskIP),
        process=lambda ans, _: dp.const_space(ans),
        tools={
            CheckZ3: lambda call: _run_z3_tool(call).using(dp.just_compute)
        },
    )
    return result.answer


@ensure_compatible(folio_only_ask)
@ensure_compatible(folio_formalization_agent)
def folio_ask_policy(
    model_name: dp.StandardModelName = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    num_requests: int = 10,
    api_type: APIType = "responses",
) -> dp.Policy[Branch | Fail, FolioAskIP]:
    budget = dp.BudgetLimit({dp.NUM_REQUESTS: num_requests})
    model = dp.standard_model(
        model_name, {"reasoning_effort": reasoning_effort}, api_type=api_type
    )
    return dp.with_budget(budget) @ dp.dfs() & FolioAskIP(
        check_deduction=dp.few_shot(model=model)
    )


# --- Z3 constraint solver agent ---

# Whitelisted z3 names available in eval() expressions.
_Z3_ALLOWED_NAMES: list[str] = [
    # Core sorts / constructors
    "BoolSort",
    "DeclareSort",
    # Term / constant / function constructors
    "Const",
    "Function",
    # Logical connectives / predicates
    "Not",
    "And",
    "Or",
    "Xor",
    "Implies",
    "Distinct",
    "If",
    # Quantifiers
    "ForAll",
    "Exists",
    "Lambda",
    # Integer support
    "IntSort",
    "Int",
    "Ints",
    "IntVal",
    "FreshInt",
    "Sum",
    "Product",
    # Misc helpers
    "simplify",
    "substitute",
]

# Whitelisted attribute names allowed on z3 objects.
_Z3_ALLOWED_ATTRS: list[str] = []


@dataclass
class Z3Declaration:
    """A named Z3 variable or function declaration."""

    name: str
    expr: str


@dataclass
class Z3Result:
    status: str
    model: str | None
    error: str | None


@dataclass
class RunZ3Solver(dp.AbstractTool[Z3Result]):
    """
    Check satisfiability of constraints using the Z3 SMT solver.

    `declarations`: list of Z3 variable/function declarations. Each has a
    `name` (the variable name) and `expr` (a Z3 constructor expression).
    Example: [{"name": "x", "expr": "Int('x')"},
        {"name": "f", "expr": "Function('f', IntSort(), BoolSort())"}]

    `constraints`: list of Z3 boolean expressions as strings.
    Example: ["x + y > 5", "x < 10", "f(x) == True"]

    The solver result (sat/unsat/unknown) and model are returned.
    """

    declarations: list[Z3Declaration]
    constraints: list[str]


def _validate_z3_expr(expr: str) -> str | None:
    """Validate a single expression string. Returns error or None."""
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception as e:
        return f"Error in '{expr}': {str(e)}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr not in _Z3_ALLOWED_ATTRS:
                return (
                    f"Attribute '.{node.attr}' is not allowed. "
                    f"Allowed: {_Z3_ALLOWED_ATTRS}"
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


def _run_z3_solver(tool_call: RunZ3Solver) -> Z3Result:
    import z3  # type: ignore

    ns = _build_z3_namespace()
    ret = Z3Result(status="error", model=None, error=None)

    # Evaluate declarations and add them to the namespace
    for decl in tool_call.declarations:
        if not decl.name.isidentifier():
            ret.error = f"Error: Invalid variable name '{decl.name}'."
            return ret
        err = _validate_z3_expr(decl.expr)
        if err is not None:
            ret.error = f"Error in declaration '{decl.name}': {err}"
            return ret
        try:
            ns[decl.name] = eval(decl.expr, ns)  # noqa: S307
        except Exception as e:
            ret.error = (
                f"Error in declaration '{decl.name}':"
                + f"{type(e).__name__}: {e}"
            )
            return ret

    # Evaluate constraints and add to solver
    solver = z3.Solver()
    for i, constraint in enumerate(tool_call.constraints):
        err = _validate_z3_expr(constraint)
        if err is not None:
            ret.error = f"Error in constraint {i}: {err}"
            return ret
        try:
            c = eval(constraint, ns)  # noqa: S307
        except Exception as e:
            ret.error = f"Error in constraint {i}: {type(e).__name__}: {e}"
            return ret
        solver.add(c)  # type: ignore

    # Check and format result
    result = solver.check()  # type: ignore
    ret.status = str(result)
    if result == z3.sat:
        model = solver.model()
        ret.model = str(model)

    return ret


@strategy
def run_z3_solver(call: RunZ3Solver) -> Strategy[Compute, object, Z3Result]:
    ret = yield from dp.compute(_run_z3_solver)(call)
    return ret


@dataclass
class CheckDeductionZ3Code(
    dp.Query[dp.Response[DeductionAnswer, RunZ3Solver]]
):
    sentences: list[str]
    allowed_names: list[str]
    allowed_attrs: list[str]
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
            sentences=sentences,
            prefix=prefix,
            allowed_names=_Z3_ALLOWED_NAMES,
            allowed_attrs=_Z3_ALLOWED_ATTRS,
        ).using(lambda p: p.check_deduction, FolioZ3AgentIP),
        process=lambda ans, _: dp.const_space(ans),
        tools={
            RunZ3Solver: lambda call: run_z3_solver(call).using(
                dp.just_compute
            )
        },
    )
    return result.answer


@ensure_compatible(folio_z3_agent)
def folio_z3_agent_policy(
    model_name: dp.StandardModelName = "gpt-5-nano",
    reasoning_effort: dp.ReasoningEffort = "low",
    num_requests: int = 10,
    api_type: APIType = "responses",
) -> dp.Policy[Branch | Fail, FolioZ3AgentIP]:
    budget = dp.BudgetLimit({dp.NUM_REQUESTS: num_requests})
    model = dp.standard_model(
        model_name, {"reasoning_effort": reasoning_effort}, api_type=api_type
    )
    return dp.with_budget(budget) @ dp.dfs() & FolioZ3AgentIP(
        check_deduction=dp.few_shot(model=model)
    )
