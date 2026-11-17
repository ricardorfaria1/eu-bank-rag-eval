from pathlib import Path

import pytest
from pydantic import ValidationError

from eurag.config import load_all, load_config


def test_load_three_configs():
    configs = load_all(Path("configs"))
    assert [c.id for c in configs] == ["baseline", "by-article", "by-article-hybrid"]
    base, art, hyb = configs
    assert (base.chunking, base.retrieval) == ("fixed", "dense")
    assert (art.chunking, art.retrieval) == ("article", "dense")
    assert (hyb.chunking, hyb.retrieval) == ("article", "hybrid")
    assert all(c.top_k == 5 and c.answer_model == "claude-opus-5-5" for c in configs)


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("id: x\nchunking: fixed\nretrieval: dense\ntemperature: 0\n")
    with pytest.raises(ValidationError):
        load_config(p)


def test_fingerprint_changes_with_top_k():
    base = load_config(Path("configs/baseline.yaml"))
    assert base.fingerprint() != base.model_copy(update={"top_k": 3}).fingerprint()
    assert len(base.fingerprint()) == 12
