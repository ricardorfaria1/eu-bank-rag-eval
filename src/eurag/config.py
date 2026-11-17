"""RAG configurations, one YAML file each under configs/."""

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class RagConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    chunking: Literal["fixed", "article"]
    retrieval: Literal["dense", "hybrid"]
    top_k: int = 5
    embedding_model: str = "BAAI/bge-m3"
    answer_model: str = "claude-opus-5-5"

    def fingerprint(self) -> str:
        """Short hash of every setting; cached answers are only reused for the same fingerprint."""
        canonical = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    @property
    def index_name(self) -> str:
        return f"{self.chunking}-{self.retrieval}"


def load_config(path: Path) -> RagConfig:
    return RagConfig(**yaml.safe_load(Path(path).read_text()))


def load_all(directory: Path = Path("configs")) -> list[RagConfig]:
    return sorted((load_config(p) for p in Path(directory).glob("*.yaml")), key=lambda c: c.id)
