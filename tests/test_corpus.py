import json
from pathlib import Path

from eurag.corpus import SourceDoc, download_all, load_manifest


def test_manifest_ids_unique_and_known_formats():
    docs = load_manifest(Path("corpus/manifest.yaml"))
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids)) >= 10
    assert {d.format for d in docs} <= {"html", "pdf"}
    assert {d.language for d in docs} == {"en", "pt"}


def _doc(doc_id="X", manual=False):
    return SourceDoc(doc_id=doc_id, title="t", aliases=[doc_id], url=f"https://example.test/{doc_id}",
                     format="pdf", language="en", publisher="p", reuse_note="r", manual=manual)


def test_download_writes_lock(tmp_path):
    report = download_all([_doc()], tmp_path, fetch=lambda url: b"hello")
    lock = json.loads((tmp_path / "lock.json").read_text())
    assert (tmp_path / "X.pdf").read_bytes() == b"hello"
    assert lock["X"]["url"] == "https://example.test/X"
    assert lock["X"]["sha256"] == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert report["X"] == "downloaded"


def test_download_skips_unchanged(tmp_path):
    download_all([_doc()], tmp_path, fetch=lambda url: b"hello")
    calls = []
    report = download_all([_doc()], tmp_path, fetch=lambda url: calls.append(url) or b"hello")
    assert calls == [] and report["X"] == "unchanged"


def test_manual_doc_not_fetched(tmp_path):
    calls = []
    report = download_all([_doc("M", manual=True)], tmp_path, fetch=lambda url: calls.append(url) or b"x")
    assert calls == [] and report["M"] == "missing (manual download)"
    (tmp_path / "M.pdf").write_bytes(b"manual")
    report = download_all([_doc("M", manual=True)], tmp_path, fetch=lambda url: b"x")
    assert report["M"] == "manual"
    assert "M" in json.loads((tmp_path / "lock.json").read_text())
