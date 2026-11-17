"""Answer cache: one JSONL file per config, keyed by config fingerprint and question id.

A resumed run never pays twice for the same answer, and a changed config
(new fingerprint) never reuses stale answers.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from eurag.config import RagConfig

if TYPE_CHECKING:
    from eurag.answer import Answer


class AnswerCache:
    def __init__(self, directory: Path):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _file(self, config_id: str) -> Path:
        return self._dir / f"{config_id}.jsonl"

    def _load(self, config_id: str) -> dict[tuple[str, str], dict]:
        path = self._file(config_id)
        if not path.exists():
            return {}
        rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
        return {(r["config_fingerprint"], r["question_id"]): r for r in rows}

    def get(self, config: RagConfig, question_id: str) -> "Answer | None":
        from eurag.answer import Answer

        row = self._load(config.id).get((config.fingerprint(), question_id))
        return Answer.model_validate(row) if row else None

    def put(self, answer: "Answer") -> None:
        with self._file(answer.config_id).open("a", encoding="utf-8") as f:
            f.write(answer.model_dump_json() + "\n")

    def all(self, config: RagConfig) -> list["Answer"]:
        from eurag.answer import Answer

        fp = config.fingerprint()
        return [Answer.model_validate(r) for (f, _), r in self._load(config.id).items() if f == fp]
