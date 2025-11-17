import re
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import z3  # type: ignore

# pyright: strict
# fmt: on


@dataclass
class ZebraPuzzle:
    """Representation of a ZebraLogic instance."""

    id: str | None
    num_houses: int
    context: str
    attributes: dict[str, list[str]]
    clues: list[str]


def _make_identifier(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^0-9a-z]+", "_", text)
    text = text.strip("_")
    if not text:
        text = "value"
    if text[0].isdigit():
        text = f"_{text}"
    return text


def parse_puzzle(puzzle: str, id: str | None = None) -> ZebraPuzzle:
    """Extract the number of houses, attributes, and clues from a puzzle."""

    header, _, clues_part = puzzle.partition("## Clues:")
    house_match = re.search(r"There are (\d+) houses", header)
    if not house_match:
        raise ValueError("Could not determine the number of houses.")
    num_houses = int(house_match.group(1))

    attributes: dict[str, list[str]] = {}
    for i, bullet in enumerate(re.findall(r"-\s+[^\n]+", header)):
        values = re.findall(r"`([^`]+)`", bullet)
        if not values:
            raise ValueError(f"No attribute values found in bullet: {bullet}")
        label_match = re.search(r"unique\s+([^:]+):", bullet)
        label = label_match.group(1).strip() if label_match else None
        if not label or _make_identifier(label) in attributes:
            label = f"attribute_{i + 1}"
        if (i == 0) and label != "name":
            raise ValueError(
                "The first attribute must be 'name' representing the people."
            )
        attributes[_make_identifier(label)] = values

    clues: list[str] = []
    for line in clues_part.splitlines():
        match = re.match(r"\s*(\d+)\.", line)
        if match:
            clues.append(line[match.end() :].strip())

    if not clues:
        raise ValueError("No clues found in the puzzle description.")

    return ZebraPuzzle(
        id=id,
        num_houses=num_houses,
        context=header.strip(),
        attributes=attributes,
        clues=clues,
    )


class SolverState(NamedTuple):
    solver: z3.Solver
    z3vars_for_attr: dict[str, dict[str, z3.ArithRef]]
    exec_global_vars: dict[str, object]
    variable_names: list[str]


def build_solver(puzzle: ZebraPuzzle) -> SolverState:
    """Build a Z3 solver encoding the initial uniqueness constraints for the
    given ZebraPuzzle instance."""
    solver = z3.Solver()
    z3vars_for_attr: dict[str, dict[str, z3.ArithRef]] = {}
    variable_names: list[str] = []
    exec_global_vars: dict[str, object] = {
        "num_houses": puzzle.num_houses,
        "solver": solver,
        "z3": z3,
    }

    for attr, values in puzzle.attributes.items():
        z3vars: dict[str, z3.ArithRef] = {}
        for value in values:
            var_name = f"{attr}_{_make_identifier(value)}"
            variable: z3.ArithRef = z3.Int(var_name)  # type: ignore
            z3vars[value] = variable
            variable_names.append(var_name)
            exec_global_vars[var_name] = variable
            solver.add(variable >= 1)  # type: ignore
            solver.add(variable <= puzzle.num_houses)  # type: ignore
            solver.add(z3.Distinct(*z3vars.values()))  # type: ignore
        z3vars_for_attr[attr] = z3vars

    return SolverState(
        solver=solver,
        z3vars_for_attr=z3vars_for_attr,
        exec_global_vars=exec_global_vars,
        variable_names=variable_names,
    )


def model_to_solution(state: SolverState) -> list[list[str]]:
    """Convert a Z3 model to a solution array."""

    model = state.solver.model()
    num_houses = len(state.z3vars_for_attr[next(iter(state.z3vars_for_attr))])
    num_attributes = len(state.z3vars_for_attr)
    solution = np.empty((num_houses, num_attributes + 1), dtype=object)
    solution[:, 0] = np.arange(1, num_houses + 1).astype(str)

    for attr_index, (_, z3vars) in enumerate(state.z3vars_for_attr.items()):
        for value, var in z3vars.items():
            val = model.evaluate(var, model_completion=True)  # type: ignore
            house_num = int(str(val)) - 1
            solution[house_num, attr_index + 1] = value

    return solution.tolist()


if __name__ == "__main__":
    example_puzzle = """
    There are 2 houses, numbered 1 to 2 from left to right, as seen from across 
    the street. Each house is occupied by a different person. Each house has a 
    unique attribute for each of the following characteristics:
    - Each person has a unique name: `Arnold`, `Eric`
    - People use unique phone models: `samsung galaxy s21`, `iphone 13`
    - Each person has a unique type of pet: `dog`, `cat`

    ## Clues:
    1. The person who owns a dog is directly left of Arnold.
    2. The person who uses an iPhone 13 is the person who has a cat.
    """

    puzzle = parse_puzzle(example_puzzle)
    print(f"Context:\n{puzzle.context}\n")
    print(f"Number of houses: {puzzle.num_houses}")
    print("Attributes:")
    for attr, values in puzzle.attributes.items():
        print(f"  {attr}: {values}")
    print("Clues:")
    for clue in puzzle.clues:
        print(f"  - {clue}")

    solver_state = build_solver(puzzle)
    solver_state.solver.add(  # type: ignore
        solver_state.z3vars_for_attr["type_of_pet"]["dog"] + 1
        == solver_state.z3vars_for_attr["name"]["Arnold"]
    )
    solver_state.solver.add(  # type: ignore
        solver_state.z3vars_for_attr["phone_models"]["iphone 13"]
        == solver_state.z3vars_for_attr["type_of_pet"]["cat"]
    )
    print(solver_state.solver.check() == z3.sat)  # type: ignore
    print(solver_state.solver.model())  # type: ignore
