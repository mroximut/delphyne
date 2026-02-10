"""
Iteratively calling a strategy or query.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import delphyne.core as dp
from delphyne.stdlib.environments import PolicyEnv
from delphyne.stdlib.nodes import Fail, Run, fail, run, spawn_node
from delphyne.stdlib.opaque import Opaque, OpaqueSpace
from delphyne.stdlib.policies import search_policy
from delphyne.stdlib.strategies import strategy
from delphyne.stdlib.streams import Stream, StreamTransformer


@dataclass
class Iteration(dp.Node):
    """
    Node type for the singleton trees induced by `iterate`.
    """

    next: Callable[
        [dp.Tracked[Any] | None],
        OpaqueSpace[Any, tuple[Any | None, Any]],
    ]

    def navigate(self) -> dp.Navigation:
        return (yield self.next(None))


def _iterate_inherited_tags(next: Any) -> Sequence[dp.SpaceBuilder[Any]]:
    return [next(None)]


@strategy(name="iterate", inherit_tags=_iterate_inherited_tags)
def _iterate[P, S, T](
    next: Callable[[S | None], Opaque[P, tuple[T | None, S]]],
) -> dp.Strategy[Iteration | Fail, P, T]:
    recv = yield spawn_node(Iteration, next=next)
    ret = cast(tuple[T | None, S], recv.action)
    yielded, _new_state = ret
    if yielded is None:
        yield from fail(label="no_element_yielded")
    else:
        return yielded


@search_policy
def _search_iteration[P, T](
    tree: dp.Tree[Iteration | Fail, P, T],
    env: PolicyEnv,
    policy: P,
    *,
    stop_on_reject: bool = True,
) -> dp.StreamGen[T]:
    state: dp.Tracked[Any] | None = None

    def _next_stream() -> dp.StreamGen[T]:
        nonlocal state
        assert isinstance(tree.node, Iteration)
        sol = yield from tree.node.next(state).stream(env, policy).first()
        if sol is not None:
            # Here, `sol` contains the value we are interested
            # in so it is tempting to just yield it. However, this
            # is not allowed since the attached reference would not
            # properly point to a success node. In our Haskell
            # implementation, such a bug would be caught by the type
            # system. Here, we catch it dynamically.
            yielded_and_new_state = sol.tracked
            state = yielded_and_new_state[1]
            child = tree.child(yielded_and_new_state)
            assert not isinstance(child.node, Iteration)
            if isinstance(child.node, dp.Success):
                yield dp.Solution(child.node.success)

    yield from Stream(_next_stream).loop(stop_on_reject=stop_on_reject)


def iterate[P, S, T](
    next: Callable[[S | None], Opaque[P, tuple[T | None, S]]],
    transform_stream: Callable[[P], StreamTransformer | None] | None = None,
    *,
    stop_on_reject: bool = True,
) -> Opaque[P, T]:
    """
    Iteratively call a strategy or query, repeatedly feeding back the
    last call's output state into a new call and yielding values along
    the way.

    A standard use case is to repeatedly call a query or strategy with a
    blacklist of previously generated values, so as to produce diverse
    success values.

    Arguments:
        next: A parametric opaque space, induced by a query or stratey
            that takes a state as an input (or `None` initially) and
            outputs a new state, along with a generated value.
        transform_stream: An optional mapping from the inner policy to a
            stream transformer to be applied to the resulting stream of
            generated values.

    Returns:
        An opaque space enumerating all generated values.
    """

    def iterate_policy(inner_policy: P):
        policy = _search_iteration(stop_on_reject=stop_on_reject)
        if transform_stream is not None:
            trans = transform_stream(inner_policy)
            if trans is not None:
                policy = trans @ policy
        return policy & inner_policy

    return _iterate(next).using(iterate_policy)


@dataclass
class Aggregation(dp.Node):
    """
    Node type for the singleton trees induced by `aggregate`.
    """

    space: OpaqueSpace[Any, Any]
    aggreg: Callable[[Sequence[Any]], OpaqueSpace[Any, Any]] | None = None

    def navigate(self) -> dp.Navigation:
        return (yield self.space)


@strategy(name="aggregate")
def _aggregate[P, T](
    space: Opaque[P, T],
    aggreg: Callable[[Sequence[T]], Opaque[P, Sequence[T]]] | None = None,
) -> dp.Strategy[Aggregation, P, Sequence[T]]:
    recv = yield spawn_node(Aggregation, space=space, aggreg=aggreg)
    sols = cast(Sequence[T], recv.action)
    return sols


@search_policy
def _search_aggregation[P, U](
    tree: dp.Tree[Aggregation, P, U],
    env: PolicyEnv,
    policy: P,
) -> dp.StreamGen[U]:
    assert isinstance(tree.node, Aggregation)
    sols = yield from tree.node.space.stream(env, policy).all()

    trackeds = [sol.tracked for sol in sols]
    child = tree.child(trackeds)
    if tree.node.aggreg is not None:
        aggregated = (
            yield from tree.node.aggreg(
                [tracked.value for tracked in trackeds]
            )
            .stream(env, policy)
            .first()
        )
        if aggregated is not None:
            child = tree.child(aggregated.tracked)

    assert isinstance(child.node, dp.Success)
    yield dp.Solution(child.node.success)


def aggregate[P, T](
    space: Opaque[P, T],
    aggreg: Callable[[Sequence[T]], Opaque[P, Sequence[T]]] | None = None,
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Run, P, Sequence[T]]:
    """Aggregate all values generated by an opaque space into a sequence."""

    def aggregate_policy(inner_policy: P):
        return _search_aggregation() & inner_policy

    values = yield from run(_aggregate(space).using(aggregate_policy))
    if aggreg is not None:
        aggregated = yield from run(aggreg(values))
    else:
        aggregated = values
    return aggregated
