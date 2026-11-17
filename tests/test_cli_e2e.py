"""Whole pipeline offline: fake embeddings, fake chat model, stub judge metrics."""

import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import yaml

from eurag import cli
from tests.fakes import WordHashSparse, fake_dense, recording_chat

ROOT = Path(__file__).parent.parent
FIX = Path(__file__).parent / "fixtures"


class StubMetric:
    loops: ClassVar[set] = set()
    calls = 0

    async def ascore(self, **kwargs):
        import asyncio

        StubMetric.loops.add(id(asyncio.get_running_loop()))
        StubMetric.calls += 1
        return SimpleNamespace(value=0.75)


def _project(tmp_path):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "manifest.yaml").write_text(yaml.safe_dump([{
        "doc_id": "AMLR", "title": "AMLR", "aliases": ["AMLR"], "url": "https://example.test/amlr",
        "format": "html", "language": "en", "publisher": "EUR-Lex", "reuse_note": "ok"}]))
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    shutil.copy(FIX / "eurlex_excerpt.html", raw / "AMLR.html")
    ts = tmp_path / "data" / "testset"
    ts.mkdir(parents=True)
    with (ts / "testset.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "language", "category", "origin", "user_input", "reference", "reference_contexts",
                    "reference_citations", "review_note"])
        w.writerow(["q1", "en", "single_doc", "reviewed", "What is the subject matter of the AMLR?",
                    "Rules on AML measures.", "[]", json.dumps(["AMLR:1"]), ""])
        w.writerow(["q2", "en", "single_doc", "reviewed", "What are the definitions?", "Definitions.", "[]",
                    json.dumps(["AMLR:2"]), ""])
        w.writerow(["q3", "pt", "out_of_scope", "handwritten", "Qual é a taxa do BCE?", "Fora do âmbito.", "[]",
                    "[]", ""])


def test_offline_pipeline(tmp_path, monkeypatch):
    _project(tmp_path)
    monkeypatch.chdir(tmp_path)
    chats = []

    def chat_factory(model):
        chats.append(recording_chat("Rules on measures [AMLR, art. 1]."))
        return chats[-1]

    monkeypatch.setitem(cli.FACTORIES, "embeddings", lambda model: fake_dense())
    monkeypatch.setitem(cli.FACTORIES, "sparse", WordHashSparse)
    monkeypatch.setitem(cli.FACTORIES, "chat", chat_factory)
    monkeypatch.setitem(cli.FACTORIES, "judge_metrics",
                        lambda: {"faithfulness": StubMetric(), "factual_correctness": StubMetric()})

    assert cli.main(["ingest", "--skip-download"]) == 0
    assert (tmp_path / "data/chunks/article.jsonl").exists()
    assert cli.main(["run", "--config", "all", "--yes", "--limit", "3"]) == 0
    assert sum(len(c.prompts) for c in chats) == 9
    StubMetric.loops.clear()
    assert cli.main(["evaluate", "--config", "all", "--yes"]) == 0
    assert len((tmp_path / "results/scores/baseline.jsonl").read_text().splitlines()) == 3
    assert len(StubMetric.loops) == 1  # one event loop for every config: the judge client is reused safely
    StubMetric.calls = 0
    assert cli.main(["evaluate", "--config", "all", "--yes"]) == 0
    assert StubMetric.calls == 0  # already scored rows are not paid for again
    assert json.loads((tmp_path / "results/budget.json").read_text())[0]["stage"] == "run"
    assert cli.main(["report"]) == 0
    report = (tmp_path / "REPORT.md").read_text()
    for cfg in ("baseline", "by-article", "by-article-hybrid"):
        assert cfg in report

    chats.clear()
    assert cli.main(["run", "--config", "all", "--yes"]) == 0
    assert sum(len(c.prompts) for c in chats) == 0

    assert cli.main(["calibrate", "--sample", "--n", "3"]) == 0
    with (tmp_path / "calibration/sample.csv").open(encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == 3  # excerpts span lines inside quoted CSV fields
