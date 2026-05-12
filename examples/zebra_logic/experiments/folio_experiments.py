from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

import delphyne as dp
from delphyne.stdlib.standard_models import APIType

SOLUTIONS_CSV: Path = (
    Path(__file__).resolve().parent.parent
    / "datasets--yale-nlp--FOLIO"
    / "folio_v2_validation.csv"
)


def load_folio_benchmark() -> dict[int, tuple[str, bool | None]]:
    df = pd.read_csv(SOLUTIONS_CSV)  #  type: ignore
    benchmarks: dict[int, tuple[str, bool | None]] = {}
    for _, row in df.iterrows():
        puzzle_id = int(row["example_id"])
        puzzle: str = (
            row["premises"] + "\n" + "Conclusion: " + row["conclusion"]
        )
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


def sample(keys: list[int], len_samples: int, seed: int = 42) -> list[int]:
    import random

    random.seed(seed)
    return random.sample(keys, len_samples)


# SAMPLE_IDS_9feb_200 = [
#     id
#     for id in sample(list(BENCHS.keys()), len(BENCHS), seed=424242)
#     if BENCHS[id][1] is not None
# ][:200] ## train csv

# SAMPLE_IDS_5_may = [
#     id
#     for id in sample(list(BENCHS.keys()), len(BENCHS), seed=424242)
#     if BENCHS[id][1] is not None
# ][200:400] ## train csv

VALIDATION_IDS = [
    id for id in list(BENCHS.keys()) if BENCHS[id][1] is not None
]


@dataclass
class AggregateConfig:
    bench_id: int
    model_name: str
    reasoning_effort: dp.ReasoningEffort
    sequence_type: Literal["mixed", "all_normal", "all_normal_reflect"]
    max_dollar_budget: float | None
    max_rounds_each: int = 5
    number_of_experts: int = 3
    aggregation_type: Literal["majority_vote", "favor_unsat", "judge"] = (
        "judge"
    )
    temperature: float | None = None
    seed: int = 0
    api_type: APIType = "chat_completions"

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
                "sequence_type": self.sequence_type,
                "aggregation_type": self.aggregation_type,
                "api_type": self.api_type,
                "number_of_experts": self.number_of_experts,
            },
            budget=budget,
        )


@dataclass
class IterativeConfig:
    bench_id: int
    model_name: str
    reasoning_effort: dp.ReasoningEffort
    max_dollar_budget: float | None
    max_restarts: int = 2
    max_requests_per_attempt: int = 30
    max_retries_per_chunk: int = 3
    reflect_if_sat: bool = True
    majority_vote_size: int | None = 3
    number_of_chunks: int = 2
    max_depth_for_formalize: int = 5
    api_type: APIType = "chat_completions"
    temperature: float | None = None
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
                "max_retries_per_chunk": self.max_retries_per_chunk,
                "reflect_if_sat": self.reflect_if_sat,
                "max_depth_for_formalize": self.max_depth_for_formalize,
                "api_type": self.api_type,
                "majority_vote_size": self.majority_vote_size,
                "number_of_chunks": self.number_of_chunks,
            },
            budget=budget,
        )


@dataclass
class OnlyAskConfig:
    bench_id: int
    model_name: str
    reasoning_effort: dp.ReasoningEffort
    max_dollar_budget: float | None
    temperature: float | None = None
    seed: int = 0
    num_requests: int = 10
    api_type: APIType = "responses"

    def instantiate(self, context: object):
        budget: dict[str, float] = {}
        if self.max_dollar_budget is not None:
            budget[dp.DOLLAR_PRICE] = self.max_dollar_budget
        return dp.RunStrategyArgs(
            strategy="folio_only_ask",
            args={"puzzle": BENCHS[self.bench_id][0]},
            policy="folio_ask_policy",
            policy_args={
                "model_name": self.model_name,
                "reasoning_effort": self.reasoning_effort,
                "num_requests": self.num_requests,
                "api_type": self.api_type,
            },
            budget=budget,
        )


@dataclass
class FormalizationAgentConfig:
    bench_id: int
    model_name: str
    reasoning_effort: dp.ReasoningEffort
    max_dollar_budget: float | None
    temperature: float | None = None
    seed: int = 0
    num_requests: int = 10
    api_type: APIType = "responses"

    def instantiate(self, context: object):
        budget: dict[str, float] = {}
        if self.max_dollar_budget is not None:
            budget[dp.DOLLAR_PRICE] = self.max_dollar_budget
        return dp.RunStrategyArgs(
            strategy="folio_formalization_agent",
            args={"puzzle": BENCHS[self.bench_id][0]},
            policy="folio_ask_policy",
            policy_args={
                "model_name": self.model_name,
                "reasoning_effort": self.reasoning_effort,
                "num_requests": self.num_requests,
                "api_type": self.api_type,
            },
            budget=budget,
        )


@dataclass
class Z3AgentConfig:
    bench_id: int
    model_name: str
    reasoning_effort: dp.ReasoningEffort
    max_dollar_budget: float | None
    temperature: float | None = None
    seed: int = 0
    num_requests: int = 10
    api_type: APIType = "responses"

    def instantiate(self, context: object):
        budget: dict[str, float] = {}
        if self.max_dollar_budget is not None:
            budget[dp.DOLLAR_PRICE] = self.max_dollar_budget
        return dp.RunStrategyArgs(
            strategy="folio_z3_agent",
            args={"puzzle": BENCHS[self.bench_id][0]},
            policy="folio_z3_agent_policy",
            policy_args={
                "model_name": self.model_name,
                "reasoning_effort": self.reasoning_effort,
                "num_requests": self.num_requests,
                "api_type": self.api_type,
            },
            budget=budget,
        )


if __name__ == "__main__":
    print(len(VALIDATION_IDS))
    pass
