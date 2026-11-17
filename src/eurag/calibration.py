"""Judge calibration: a second, blind grading of a sample, compared with the Ragas judge.

The second grader is the Claude Code agent that built the project. Its grades
are not reproducible by third parties, and the report says so.
"""

import csv
import random
from pathlib import Path

SAMPLE_COLUMNS = ["question_id", "config_id", "question", "reference", "excerpts", "answer"]


def sample_for_calibration(rows: list[dict], texts: dict[tuple[str, str], dict], out: Path, n: int = 20,
                           seed: int = 11) -> Path:
    """Write a blind sample (no scores), stratified by configuration, for the second grader."""
    rng = random.Random(seed)
    scored = [r for r in rows if r.get("category") != "out_of_scope"]
    configs = sorted({r["config_id"] for r in scored})
    per_config, extra = divmod(n, len(configs))
    picked = []
    for i, cfg in enumerate(configs):
        pool = sorted((r for r in scored if r["config_id"] == cfg), key=lambda r: r["question_id"])
        picked += rng.sample(pool, min(len(pool), per_config + (1 if i < extra else 0)))
    rng.shuffle(picked)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        for r in picked:
            writer.writerow({"question_id": r["question_id"], "config_id": r["config_id"],
                             **texts[(r["config_id"], r["question_id"])]})
    return Path(out)


def agreement(judge: list[float], agent: list[float], threshold: float = 0.5) -> tuple[float, float]:
    """Percent agreement and Cohen's kappa after binarising both grades at threshold."""
    a = [v >= threshold for v in judge]
    b = [v >= threshold for v in agent]
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    kappa = 1.0 if expected == 1 else (observed - expected) / (1 - expected)
    return observed, kappa


def load_agent_grades(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return {(r["config_id"], r["question_id"]): {k: float(v) for k, v in r.items()
                                                     if k not in ("config_id", "question_id") and v != ""}
                for r in csv.DictReader(f)}
