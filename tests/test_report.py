import json
import shutil
from pathlib import Path

from eurag.report import build_report, classify_failure

FIX = Path(__file__).parent / "fixtures" / "scores_sample.jsonl"


def _scores_dir(tmp_path):
    d = tmp_path / "scores"
    d.mkdir()
    rows = [json.loads(line) for line in FIX.read_text().splitlines()]
    for cfg in ("baseline", "by-article", "by-article-hybrid"):
        (d / f"{cfg}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["config_id"] == cfg))
    return d


def test_report_sections_in_order(tmp_path):
    out = tmp_path / "REPORT.md"
    build_report(_scores_dir(tmp_path), out, chart=tmp_path / "metrics.png")
    text = out.read_text()
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings == ["## Run", "## Results by configuration", "## Difference between configurations",
                        "## Results by language", "## Failures: retrieval or generation", "## Limitations"]
    for cfg in ("baseline", "by-article", "by-article-hybrid"):
        assert cfg in text


def test_inconclusive_label_when_ci_crosses_zero(tmp_path):
    d = _scores_dir(tmp_path)
    shutil.copy(d / "baseline.jsonl", d / "by-article-hybrid.jsonl")
    for r in (d / "by-article-hybrid.jsonl").read_text().splitlines():
        assert json.loads(r)["config_id"] == "baseline"
    text_rows = [json.loads(line) for line in (d / "baseline.jsonl").read_text().splitlines()]
    (d / "by-article-hybrid.jsonl").write_text(
        "".join(json.dumps({**r, "config_id": "by-article-hybrid"}) + "\n" for r in text_rows))
    out = tmp_path / "REPORT.md"
    build_report(d, out, chart=tmp_path / "m.png")
    diff_section = out.read_text().split("## Difference between configurations")[1].split("\n## ")[0]
    hybrid_lines = [line for line in diff_section.splitlines() if "by-article-hybrid vs baseline" in line]
    assert hybrid_lines and all("inconclusive" in line for line in hybrid_lines)


def test_failure_breakdown_classifies_retrieval_vs_generation():
    assert classify_failure({"hit_at_5": 0.0, "factual_correctness": 0.9}) == "retrieval"
    assert classify_failure({"hit_at_5": 1.0, "factual_correctness": 0.2}) == "generation"
    assert classify_failure({"hit_at_5": 1.0, "factual_correctness": 0.9}) is None
    assert classify_failure({"hit_at_5": None, "factual_correctness": None}) is None


def test_report_writes_chart(tmp_path):
    chart = tmp_path / "metrics.png"
    build_report(_scores_dir(tmp_path), tmp_path / "REPORT.md", chart=chart)
    assert chart.exists() and chart.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_calibration_section_when_grades_exist(tmp_path):
    d = _scores_dir(tmp_path)
    grades = tmp_path / "agent_grades.csv"
    grades.write_text("question_id,config_id,faithfulness,factual_correctness\nq00,baseline,1,1\nq01,baseline,1,0\n")
    out = tmp_path / "REPORT.md"
    build_report(d, out, chart=tmp_path / "m.png", agent_grades=grades)
    assert "## Judge calibration" in out.read_text()


def test_report_shows_cost_and_tokens(tmp_path):
    out = tmp_path / "REPORT.md"
    build_report(_scores_dir(tmp_path), out, chart=tmp_path / "m.png", meta={"judge_cost_usd": "0.60"})
    text = out.read_text()
    assert "Answer cost per question (USD)" in text and "Input tokens per answer" in text
    assert "Judge cost" in text


def test_report_compares_each_config_with_previous(tmp_path):
    out = tmp_path / "REPORT.md"
    build_report(_scores_dir(tmp_path), out, chart=tmp_path / "m.png")
    diff = out.read_text().split("## Difference between configurations")[1].split("\n## ")[0]
    assert "by-article vs baseline" in diff and "by-article-hybrid vs by-article" in diff
