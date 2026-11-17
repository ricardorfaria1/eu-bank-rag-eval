"""Comparative report (REPORT.md) and chart from the evaluation scores."""

import json
import math
from datetime import date
from pathlib import Path

from eurag.calibration import agreement, load_agent_grades
from eurag.stats import bootstrap_ci, paired_diff_ci

METRICS = ["hit_at_5", "rr", "citation_recall", "citation_precision", "faithfulness", "factual_correctness",
           "refusal_correct"]
METRIC_LABELS = {"hit_at_5": "Hit@5", "rr": "MRR", "citation_recall": "Citation recall",
                 "citation_precision": "Citation precision", "faithfulness": "Faithfulness",
                 "factual_correctness": "Factual correctness", "refusal_correct": "Correct refusals"}
BASELINE = "baseline"
FAIL_THRESHOLD = 0.5


def load_scores(scores_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for path in sorted(Path(scores_dir).glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        out[path.stem] = sorted(rows, key=lambda r: r["question_id"])
    return out


def classify_failure(row: dict) -> str | None:
    if row.get("hit_at_5") == 0.0:
        return "retrieval"
    fc = row.get("factual_correctness")
    if fc is not None and fc < FAIL_THRESHOLD:
        return "generation"
    return None


def _fmt(ci: tuple[float, float, float]) -> str:
    mean, low, high = ci
    if math.isnan(mean):  # no scored questions
        return "n/a"
    return f"{mean:.2f} [{low:.2f}, {high:.2f}]"


def _by_question(rows: list[dict], metric: str) -> dict[str, float | None]:
    return {r["question_id"]: r.get(metric) for r in rows}


def _chart(scores: dict[str, list[dict]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in METRICS if m != "refusal_correct"]
    configs = list(scores)
    width = 0.8 / len(configs)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, cfg in enumerate(configs):
        cis = [bootstrap_ci([r.get(m) for r in scores[cfg]]) for m in metrics]
        means = [c[0] for c in cis]
        errs = [[c[0] - c[1] for c in cis], [c[2] - c[0] for c in cis]]
        xs = [j + i * width for j in range(len(metrics))]
        ax.bar(xs, means, width, yerr=errs, capsize=3, label=cfg)
    ax.set_xticks([j + width * (len(configs) - 1) / 2 for j in range(len(metrics))])
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean (95% CI)")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_report(scores_dir: Path, out: Path = Path("REPORT.md"), chart: Path = Path("report/metrics.png"),
                 agent_grades: Path | None = None, meta: dict | None = None) -> None:
    scores = load_scores(scores_dir)
    configs = [BASELINE, *[c for c in scores if c != BASELINE]] if BASELINE in scores else list(scores)
    scores = {c: scores[c] for c in configs}
    meta = meta or {}
    first = next(iter(scores.values()))
    lines = ["# Evaluation report", ""]

    lines += ["## Run", ""]
    lines.append(f"- Date: {meta.get('date', date.today().isoformat())}")
    for key in ("answer_model", "judge_model", "corpus_lock_sha256", "total_cost_usd"):
        if key in meta:
            lines.append(f"- {key.replace('_', ' ').capitalize()}: {meta[key]}")
    langs = {lang: sum(r["language"] == lang for r in first) for lang in sorted({r["language"] for r in first})}
    cats = {c: sum(r["category"] == c for r in first) for c in sorted({r["category"] for r in first})}
    lines.append(f"- Questions evaluated: {len(first)} (by language: {langs}; by category: {cats})")
    lines += ["", f"![Metrics by configuration]({Path(chart).as_posix()})", ""]

    lines += ["## Results by configuration", "", "Mean with 95% bootstrap interval over questions.", ""]
    lines.append("| Metric | " + " | ".join(configs) + " |")
    lines.append("|---|" + "---|" * len(configs))
    for m in METRICS:
        lines.append(f"| {METRIC_LABELS[m]} | " + " | ".join(_fmt(bootstrap_ci([r.get(m) for r in scores[c]]))
                                                             for c in configs) + " |")
    lines.append("| Seconds per answer | " + " | ".join(
        f"{sum(r['seconds'] for r in scores[c]) / len(scores[c]):.1f}" for c in configs) + " |")
    lines.append("")

    lines += ["## Difference against baseline", "",
              "Paired by question. Inconclusive when the 95% interval contains zero.", ""]
    lines += ["| Config | Metric | Difference | Verdict |", "|---|---|---|---|"]
    for c in configs:
        if c == BASELINE:
            continue
        for m in METRICS:
            base, other = _by_question(scores[BASELINE], m), _by_question(scores[c], m)
            qids = sorted(set(base) & set(other))
            ci = paired_diff_ci([other[q] for q in qids], [base[q] for q in qids])
            if math.isnan(ci[0]):
                continue
            verdict = "inconclusive" if ci[1] <= 0 <= ci[2] else ("better" if ci[0] > 0 else "worse")
            lines.append(f"| {c} | {METRIC_LABELS[m]} | {ci[0]:+.2f} [{ci[1]:+.2f}, {ci[2]:+.2f}] | {verdict} |")
    lines.append("")

    lines += ["## Results by language", "", "| Config | Language | Hit@5 | Factual correctness |", "|---|---|---|---|"]
    for c in configs:
        for lang in sorted({r["language"] for r in scores[c]}):
            rows = [r for r in scores[c] if r["language"] == lang]
            lines.append(f"| {c} | {lang} | {_fmt(bootstrap_ci([r.get('hit_at_5') for r in rows]))} | "
                         f"{_fmt(bootstrap_ci([r.get('factual_correctness') for r in rows]))} |")
    lines.append("")

    lines += ["## Failures: retrieval or generation", "",
              (f"A question fails when the expected article is not in the top 5 (retrieval) or factual "
              f"correctness is below {FAIL_THRESHOLD} although retrieval hit (generation)."), "",
              "| Config | Retrieval failures | Generation failures | Questions |", "|---|---|---|---|"]
    for c in configs:
        kinds = [(r["question_id"], classify_failure(r)) for r in scores[c]]
        ret = [q for q, k in kinds if k == "retrieval"]
        gen = [q for q, k in kinds if k == "generation"]
        lines.append(f"| {c} | {len(ret)} | {len(gen)} | {', '.join(ret + gen) or '-'} |")
    lines.append("")

    if agent_grades and Path(agent_grades).exists():
        grades = load_agent_grades(agent_grades)
        lines += ["## Judge calibration", "",
                  ("A second grader (the Claude Code agent that built the project) graded a random sample blind. "
                  "Grades binarised at 0.5."), "", "| Metric | Pairs | Agreement | Cohen's kappa |", "|---|---|---|---|"]
        rows_by_key = {(r["config_id"], r["question_id"]): r for rows in scores.values() for r in rows}
        for m in ("faithfulness", "factual_correctness"):
            pairs = [(rows_by_key[k][m], g[m]) for k, g in grades.items()
                     if k in rows_by_key and rows_by_key[k].get(m) is not None and m in g]
            if pairs:
                pct, kappa = agreement([p[0] for p in pairs], [p[1] for p in pairs])
                lines.append(f"| {METRIC_LABELS[m]} | {len(pairs)} | {pct:.0%} | {kappa:.2f} |")
        lines.append("")

    lines += ["## Limitations", "",
              ("- The judge (Claude Haiku) is from the same model family as the answer model (Claude Opus), "
              "which can favour the answers."),
              f"- {len(first)} questions: small differences are often inconclusive.",
              "- References were reviewed by one person.",
              ("- The judge calibration grades come from the agent that built the project and cannot be "
              "reproduced by third parties."), ""]
    Path(out).write_text("\n".join(lines))
    _chart(scores, chart)
