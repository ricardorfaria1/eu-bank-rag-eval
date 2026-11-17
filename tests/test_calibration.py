import csv
import math

from eurag.calibration import agreement, sample_for_calibration


def test_agreement_perfect_and_chance():
    assert agreement([1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]) == (1.0, 1.0)
    # 2x2 table: both yes 2, both no 1, judge yes/agent no 1 -> po=0.75, pe=(3/4*2/4)+(1/4*2/4)=0.5 -> kappa=0.5
    pct, kappa = agreement([1, 1, 1, 0], [1, 1, 0, 0])
    assert pct == 0.75 and math.isclose(kappa, 0.5)


def test_sample_has_no_scores_and_is_stratified(tmp_path):
    rows = [{"question_id": f"q{i}", "config_id": c, "category": "single_doc", "faithfulness": 0.9}
            for c in ("baseline", "by-article", "by-article-hybrid") for i in range(20)]
    texts = {(r["config_id"], r["question_id"]): {"question": "Q?", "reference": "R", "excerpts": "E", "answer": "A"}
             for r in rows}
    out = sample_for_calibration(rows, texts, tmp_path / "sample.csv", n=18, seed=11)
    with out.open(encoding="utf-8") as f:
        sample = list(csv.DictReader(f))
    assert len(sample) == 18
    assert {s["config_id"] for s in sample} == {"baseline", "by-article", "by-article-hybrid"}
    assert sum(s["config_id"] == "baseline" for s in sample) == 6
    assert not any("faithfulness" in k or "score" in k for k in sample[0])
    assert sample_for_calibration(rows, texts, tmp_path / "s2.csv", n=18, seed=11).read_text() == out.read_text()
