"""Corpus manifest and reproducible download.

The source documents are never committed. The manifest says where each one
comes from; lock.json records the sha256 and retrieval time of what was used.
"""

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
import yaml
from pydantic import BaseModel, ConfigDict


class SourceDoc(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    title: str
    aliases: list[str]
    url: str
    format: Literal["html", "pdf"]
    language: Literal["en", "pt"]
    publisher: str
    reuse_note: str
    manual: bool = False

    def raw_path(self, raw_dir: Path) -> Path:
        return Path(raw_dir) / f"{self.doc_id}.{self.format}"


def load_manifest(path: Path = Path("corpus/manifest.yaml")) -> list[SourceDoc]:
    return [SourceDoc(**entry) for entry in yaml.safe_load(Path(path).read_text())]


def http_get(url: str) -> bytes:
    response = httpx.get(url, headers={"User-Agent": "eurag/0.1"}, follow_redirects=True, timeout=60)
    response.raise_for_status()
    return response.content


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_all(docs: list[SourceDoc], raw_dir: Path, fetch: Callable[[str], bytes] = http_get) -> dict:
    """Download every document not already present with the locked hash. Returns doc_id -> status."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    lock_path = raw_dir / "lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    report = {}
    for doc in docs:
        target = doc.raw_path(raw_dir)
        if doc.manual:
            if not target.exists():
                report[doc.doc_id] = "missing (manual download)"
                continue
            data = target.read_bytes()
            status = "manual"
        elif target.exists() and lock.get(doc.doc_id, {}).get("sha256") == _sha256(target.read_bytes()):
            report[doc.doc_id] = "unchanged"
            continue
        else:
            data = fetch(doc.url)
            target.write_bytes(data)
            status = "downloaded"
        lock[doc.doc_id] = {"sha256": _sha256(data), "retrieved_at": datetime.now(UTC).isoformat(), "url": doc.url}
        report[doc.doc_id] = status
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return report
