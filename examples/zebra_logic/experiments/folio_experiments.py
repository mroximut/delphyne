from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import delphyne as dp

SOLUTIONS_CSV: Path = (
    Path(__file__).resolve().parent.parent
    / "datasets--yale-nlp--FOLIO"
    / "folio_v2_train.csv"
)


def ensure_solution(puzzle_id: int, solution: bool) -> bool:
    df = pd.read_csv(SOLUTIONS_CSV)  #  type: ignore
    ground_truth = (
        df.loc[df["example_id"] == puzzle_id, "label"].values[0] == "True"
    )  #  type: ignore
    assert isinstance(ground_truth, bool)
    return ground_truth == solution


def load_folio_benchmark() -> dict[int, tuple[str, bool | None]]:
    df = pd.read_csv(SOLUTIONS_CSV)  #  type: ignore
    benchmarks: dict[int, tuple[str, bool]] = {}
    for _, row in df.iterrows():
        puzzle_id = int(row["example_id"])
        puzzle = row["premises"] + "\n" + "Conclusion: " + row["conclusion"]
        label = (
            True
            if row["label"] == "True"
            else False
            if row["label"] == "False"
            else None
        )
        benchmarks[puzzle_id] = (puzzle, label)
    return benchmarks


BENCHS = load_folio_benchmark()


def sample(len_keys: int, len_samples: int, seed: int = 42) -> list[int]:
    import random

    random.seed(seed)
    return random.sample(range(len_keys), len_samples)


@dataclass
class OneshotConfig:
    bench_id: int
    model_name: str
    max_rounds: int
    reasoning_effort: dp.ReasoningEffort
    reflect_if_sat: bool = False
    temperature: float | None = None
    max_dollar_budget: float | None = 0.2
    seed: int = 0

    def instantiate(self, context: object):
        budget: dict[str, float] = {}
        if self.max_dollar_budget is not None:
            budget[dp.DOLLAR_PRICE] = self.max_dollar_budget
        return dp.RunStrategyArgs(
            strategy="folio_oneshot",
            args={"puzzle": BENCHS[self.bench_id][0]},
            policy="folio_oneshot_policy",
            policy_args={
                "model_name": self.model_name,
                "reasoning_effort": self.reasoning_effort,
                "temperature": self.temperature,
                "max_rounds": self.max_rounds,
                "reflect_if_sat": self.reflect_if_sat,
            },
            budget=budget,
        )


@dataclass
class IterativeNaiveConfig:
    bench_id: int
    model_name: str
    max_restarts: int
    max_requests_per_attempt: int
    max_retries_per_sentence: int
    max_rounds_per_retry_of_sentence: int
    reasoning_effort: dp.ReasoningEffort
    temperature: float | None = None
    max_dollar_budget: float | None = 0.2
    seed: int = 0

    def instantiate(self, context: object):
        budget: dict[str, float] = {}
        if self.max_dollar_budget is not None:
            budget[dp.DOLLAR_PRICE] = self.max_dollar_budget
        return dp.RunStrategyArgs(
            strategy="folio_iterative_naive",
            args={"puzzle": BENCHS[self.bench_id][0]},
            policy="folio_iterative_policy",
            policy_args={
                "model_name": self.model_name,
                "reasoning_effort": self.reasoning_effort,
                "temperature": self.temperature,
                "max_restarts": self.max_restarts,
                "max_requests_per_attempt": self.max_requests_per_attempt,
                "max_retries_per_sentence": self.max_retries_per_sentence,
                "max_rounds_per_retry_of_sentence": (
                    self.max_rounds_per_retry_of_sentence
                ),
            },
            budget=budget,
        )


@dataclass
class IterativeBlacklistConfig:
    bench_id: int
    model_name: str
    max_restarts: int
    max_requests_per_attempt: int
    max_retries_per_sentence: int
    reasoning_effort: dp.ReasoningEffort
    reflect_if_sat: bool
    max_depth_for_reflect: int
    temperature: float | None = None
    max_dollar_budget: float | None = 0.2
    seed: int = 0

    def instantiate(self, context: object):
        budget: dict[str, float] = {}
        if self.max_dollar_budget is not None:
            budget[dp.DOLLAR_PRICE] = self.max_dollar_budget
        return dp.RunStrategyArgs(
            strategy="folio_iterative_blacklist",
            args={"puzzle": BENCHS[self.bench_id][0]},
            policy="folio_iterative_blacklist_policy",
            policy_args={
                "model_name": self.model_name,
                "reasoning_effort": self.reasoning_effort,
                "temperature": self.temperature,
                "max_restarts": self.max_restarts,
                "max_requests_per_attempt": self.max_requests_per_attempt,
                "max_retries_per_sentence": self.max_retries_per_sentence,
                "reflect_if_sat": self.reflect_if_sat,
                "max_depth_for_reflect": self.max_depth_for_reflect,
            },
            budget=budget,
        )


@dataclass
class AggregateConfig:
    bench_id: int
    model_name: str
    reasoning_effort: dp.ReasoningEffort
    max_rounds_each: int = 5
    temperature: float | None = None
    max_dollar_budget: float | None = 0.2
    seed: int = 0

    def instantiate(self, context: object):
        budget: dict[str, float] = {}
        if self.max_dollar_budget is not None:
            budget[dp.DOLLAR_PRICE] = self.max_dollar_budget
        return dp.RunStrategyArgs(
            strategy="folio_aggregate",
            args={"puzzle": BENCHS[self.bench_id][0]},
            policy="folio_aggregate_policy",
            policy_args={
                "model_name": self.model_name,
                "reasoning_effort": self.reasoning_effort,
                "max_rounds_each": self.max_rounds_each,
            },
            budget=budget,
        )
