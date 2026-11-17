"""Cost estimates and the 25 USD budget ceiling for paid stages.

Token counts per call are assumptions, declared here so the estimate is
auditable. After a run, the budget ledger records the real spend from usage.
"""

import json
from pathlib import Path
from typing import Literal

# USD per million tokens (input, output), Anthropic price list cached 2026-11-03, prompts up to 100K tokens.
PRICES_USD_PER_MTOK = {"claude-opus-5-5": (4.00, 20.00), "claude-haiku-5-5": (0.10, 0.50)}
ANSWER_MODEL = "claude-opus-5-5"
JUDGE_MODEL = "claude-haiku-5-5"
BUDGET_USD = 25.0

ANSWER_TOKENS = (6000, 600)  # per answer: 5 excerpts + prompt, cited answer
JUDGE_TOKENS = (3000, 400)  # per judge call
JUDGE_CALLS_PER_METRIC = 4
JUDGE_METRICS = 2  # faithfulness, factual correctness
TESTSET_TOKENS = (2500, 500)  # per generated candidate
TESTSET_TRANSFORM_TOKENS = (1500, 300)  # per chunk, knowledge-graph extraction
TESTSET_TRANSFORM_CALLS_PER_CHUNK = 3


def _price(model: str, tokens: tuple[int, int]) -> float:
    pin, pout = PRICES_USD_PER_MTOK[model]
    return (tokens[0] * pin + tokens[1] * pout) / 1_000_000


def estimate_cost(n_questions: int, n_configs: int, stage: Literal["testset", "run", "evaluate"],
                  n_chunks: int = 0) -> float:
    if stage == "run":
        return round(n_questions * n_configs * _price(ANSWER_MODEL, ANSWER_TOKENS), 6)
    if stage == "evaluate":
        per_answer = JUDGE_METRICS * JUDGE_CALLS_PER_METRIC * _price(JUDGE_MODEL, JUDGE_TOKENS)
        return round(n_questions * n_configs * per_answer, 6)
    transforms = n_chunks * TESTSET_TRANSFORM_CALLS_PER_CHUNK * _price(JUDGE_MODEL, TESTSET_TRANSFORM_TOKENS)
    return round(n_questions * _price(JUDGE_MODEL, TESTSET_TOKENS) + transforms, 6)


def actual_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    return _price(model, (input_tokens, output_tokens))


def confirm(estimate: float, yes: bool) -> bool:
    print(f"Estimated cost: ${estimate:.2f}")
    if yes:
        return True
    return input("Proceed? [y/N] ").strip().lower() == "y"


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, path: Path = Path("results/budget.json")):
        self._path = Path(path)
        self._entries = json.loads(self._path.read_text()) if self._path.exists() else []

    @property
    def spent(self) -> float:
        return round(sum(e["usd"] for e in self._entries), 6)

    def check(self, estimate: float) -> None:
        if self.spent + estimate > BUDGET_USD:
            raise BudgetExceeded(
                f"Spent ${self.spent:.2f}; this stage (~${estimate:.2f}) would pass the ${BUDGET_USD:.0f} ceiling."
            )

    def record(self, stage: str, usd: float) -> None:
        self._entries.append({"stage": stage, "usd": round(usd, 6)})
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._entries, indent=2) + "\n")
