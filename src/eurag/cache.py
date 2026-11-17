"""Answer cache: one JSONL file per config.

An answer is reused only for the same config fingerprint, question id,
question text, corpus key (hash of the chunks) and prompt version. A resumed
run never pays twice; an edited question, config, corpus or prompt never
reuses a stale answer.
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
        rows = {}
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a line cut short by a crash mid-write
                continue
            rows[(r["config_fingerprint"], r["question_id"])] = r
        return rows

    def get(self, config: RagConfig, question_id: str, question: str | None = None,
            corpus_key: str = "") -> "Answer | None":
        from eurag.answer import PROMPT_VERSION, Answer

        row = self._load(config.id).get((config.fingerprint(), question_id))
        if not row:
            return None
        if question is not None and row.get("question") != question:
            return None
        if row.get("corpus_key", "") != corpus_key or row.get("prompt_version") != PROMPT_VERSION:
            return None
        return Answer.model_validate(row)

    def put(self, answer: "Answer") -> None:
        with self._file(answer.config_id).open("a", encoding="utf-8") as f:
            f.write(answer.model_dump_json() + "\n")

    def all(self, config: RagConfig, corpus_key: str = "") -> list["Answer"]:
        from eurag.answer import PROMPT_VERSION, Answer

        fp = config.fingerprint()
        return [Answer.model_validate(r) for (f, _), r in self._load(config.id).items()
                if f == fp and r.get("corpus_key", "") == corpus_key and r.get("prompt_version") == PROMPT_VERSION]
