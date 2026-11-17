"""eurag command line: ingest, testset, run, evaluate, calibrate, report.

Every stage reads and writes files under the current directory, so stages can
be rerun independently. Paid stages print a cost estimate and ask first.
"""

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from eurag.answer import answer_all, default_chat
from eurag.cache import AnswerCache
from eurag.calibration import sample_for_calibration
from eurag.chunking import Chunk, article_chunks, fixed_chunks
from eurag.citations import alias_map
from eurag.config import RagConfig, load_all
from eurag.convert import DocText, convert
from eurag.corpus import download_all, load_manifest
from eurag.costs import (
    ANSWER_MODEL,
    JUDGE_MODEL,
    BudgetExceeded,
    BudgetLedger,
    actual_cost,
    confirm,
    estimate_cost,
)
from eurag.evaluate import default_judge_llm, evaluate_answers, judge_metrics
from eurag.index import Retriever, build_index, default_embeddings, default_sparse
from eurag.report import build_report
from eurag.testset import (
    generate_candidates,
    load_testset,
    merge_review,
    translate_cases,
    write_testset,
)

RAW = Path("data/raw")
TEXT = Path("data/text")
CHUNKS = Path("data/chunks")
INDEX = Path("data/index")
CACHE = Path("data/cache/answers")
TESTSET = Path("data/testset/testset.csv")
SCORES = Path("results/scores")
CALIBRATION = Path("calibration")

# Swappable in tests, so the pipeline runs without downloads or API calls.
FACTORIES = {
    "embeddings": default_embeddings,
    "sparse": default_sparse,
    "chat": default_chat,
    "judge_metrics": lambda: judge_metrics(default_judge_llm(JUDGE_MODEL)),
}


def _corpus_key(strategy: str) -> str:
    """Hash of the chunk file: answers from an older corpus or chunking are never reused."""
    path = CHUNKS / f"{strategy}.jsonl"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else ""


def _read_chunks(strategy: str) -> list[Chunk]:
    path = CHUNKS / f"{strategy}.jsonl"
    return [Chunk.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def _texts() -> list[DocText]:
    docs = []
    for doc in load_manifest():
        if not doc.raw_path(RAW).exists():
            print(f"skip {doc.doc_id}: {doc.raw_path(RAW)} not found (manual download)", file=sys.stderr)
            continue
        cached = TEXT / f"{doc.doc_id}.txt"
        if not cached.exists() or cached.stat().st_mtime < doc.raw_path(RAW).stat().st_mtime:
            TEXT.mkdir(parents=True, exist_ok=True)
            cached.write_text(convert(doc, RAW).text)
        docs.append(DocText(doc_id=doc.doc_id, language=doc.language, url=doc.url, text=cached.read_text()))
    return docs


def cmd_ingest(args) -> int:
    if not args.skip_download:
        for doc_id, status in download_all(load_manifest(), RAW).items():
            print(f"{doc_id}: {status}")
    if args.download_only:
        return 0
    texts = _texts()
    CHUNKS.mkdir(parents=True, exist_ok=True)
    for strategy, fn in (("fixed", fixed_chunks), ("article", article_chunks)):
        chunks = [c for t in texts for c in fn(t)]
        (CHUNKS / f"{strategy}.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in chunks))
        print(f"{strategy}: {len(chunks)} chunks")
    configs = load_all()
    embeddings = FACTORIES["embeddings"](configs[0].embedding_model)
    for name in sorted({c.index_name for c in configs}):
        chunking, retrieval = name.split("-")
        sparse = FACTORIES["sparse"]() if retrieval == "hybrid" else None
        build_index(_read_chunks(chunking), name, INDEX, embeddings, sparse)
        print(f"index {name}: built")
    return 0


def _selected(config_arg: str) -> list[RagConfig]:
    configs = load_all()
    return configs if config_arg == "all" else [c for c in configs if c.id == config_arg]


def _paid_ok(estimate: float, yes: bool) -> bool:
    try:
        BudgetLedger().check(estimate)
    except BudgetExceeded as exc:
        print(exc, file=sys.stderr)
        return False
    return confirm(estimate, yes)


def cmd_testset(args) -> int:
    out_dir = TESTSET.parent
    if args.merge:
        cases = merge_review(out_dir / "reviewed.csv", out_dir / "handwritten.csv", TESTSET)
        print(f"{len(cases)} evaluated cases written to {TESTSET}")
        return 0
    chunks = _read_chunks("article")
    estimate = estimate_cost(args.n, 1, "testset", n_chunks=len(chunks))
    if not _paid_ok(estimate, args.yes):
        return 1
    try:
        cases = generate_candidates(chunks, JUDGE_MODEL, FACTORIES["embeddings"]("BAAI/bge-m3"), n=args.n)
        write_testset(cases, out_dir / "candidates.csv")  # saved before translation, so a failure loses nothing paid
        cases = translate_cases(cases, JUDGE_MODEL)
        write_testset(cases, out_dir / "candidates.csv")
    finally:
        BudgetLedger().record("testset (estimated)", estimate)
    print(f"{len(cases)} candidates written to {out_dir / 'candidates.csv'}; review them (docs/REVIEW_GUIDE.md)")
    return 0


def cmd_run(args) -> int:
    cases = load_testset(TESTSET)[: args.limit] if args.limit else load_testset(TESTSET)
    cache = AnswerCache(CACHE)
    configs = _selected(args.config)
    if not configs:
        print(f"unknown config {args.config!r}", file=sys.stderr)
        return 2
    keys = {cfg.id: _corpus_key(cfg.chunking) for cfg in configs}
    todo = sum(cache.get(cfg, c.id, c.user_input, keys[cfg.id]) is None for cfg in configs for c in cases)
    if todo and not _paid_ok(estimate_cost(todo, 1, "run"), args.yes):
        return 1
    aliases = alias_map(load_manifest())
    embeddings = FACTORIES["embeddings"](configs[0].embedding_model) if todo else None
    ledger = BudgetLedger()

    def record(answer) -> None:  # per answer, so an interrupted run still accounts for what it paid
        ledger.record("run", actual_cost(answer.model, answer.input_tokens, answer.output_tokens))

    for cfg in configs:
        pending = [c for c in cases if cache.get(cfg, c.id, c.user_input, keys[cfg.id]) is None]
        if not pending:
            print(f"{cfg.id}: all {len(cases)} answers cached")
            continue
        sparse = FACTORIES["sparse"]() if cfg.retrieval == "hybrid" else None
        retriever = Retriever(INDEX, cfg.index_name, embeddings, sparse)
        answers = answer_all([(c.id, c.user_input) for c in pending], cfg, retriever,
                             FACTORIES["chat"](cfg.answer_model), cache, aliases, keys[cfg.id], on_answer=record)
        cost = sum(actual_cost(a.model, a.input_tokens, a.output_tokens) for a in answers)
        print(f"{cfg.id}: {len(answers)} new answers, ${cost:.2f}")
    return 0


def _existing_scores(cfg_id: str) -> dict[str, dict]:
    path = SCORES / f"{cfg_id}.jsonl"
    if not path.exists():
        return {}
    return {r["question_id"]: r for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())}


def cmd_evaluate(args) -> int:
    cases = {c.id: c for c in load_testset(TESTSET)}
    cache = AnswerCache(CACHE)
    configs = _selected(args.config)
    if not configs:
        print(f"unknown config {args.config!r}", file=sys.stderr)
        return 2
    existing = {cfg.id: _existing_scores(cfg.id) for cfg in configs}

    def needs_scoring(cfg_id: str, qid: str) -> bool:
        row = existing[cfg_id].get(qid)
        return row is None or (args.retry_errors and bool(row["errors"]))

    answers = {cfg.id: [a for a in cache.all(cfg, _corpus_key(cfg.chunking))
                        if a.question_id in cases and needs_scoring(cfg.id, a.question_id)] for cfg in configs}
    n = sum(len(v) for v in answers.values())
    if n == 0:
        print("nothing to score: every answer already has scores (use --retry-errors to rescore failed ones)")
        return 0
    estimate = estimate_cost(n, 1, "evaluate")
    if not _paid_ok(estimate, args.yes):
        return 1

    async def score_all() -> dict[str, list[dict]]:
        # One event loop for every config: the async judge client must not outlive its loop.
        metrics = FACTORIES["judge_metrics"]()
        return {cfg.id: await evaluate_answers(answers[cfg.id], cases, metrics) for cfg in configs}

    try:
        new_rows = asyncio.run(score_all())
    finally:
        BudgetLedger().record("evaluate (estimated)", estimate)
    SCORES.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        rows = {**existing[cfg.id], **{r["question_id"]: r for r in new_rows[cfg.id]}}
        ordered = [rows[q] for q in sorted(rows)]
        (SCORES / f"{cfg.id}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
        failed = sum(bool(r["errors"]) for r in new_rows[cfg.id])
        print(f"{cfg.id}: {len(new_rows[cfg.id])} scored, {failed} with judge errors")
    return 0


def _answer_texts() -> dict[tuple[str, str], dict]:
    cases = {c.id: c for c in load_testset(TESTSET)}
    cache = AnswerCache(CACHE)
    texts = {}
    for cfg in load_all():
        for a in cache.all(cfg, _corpus_key(cfg.chunking)):
            if a.question_id in cases:
                excerpts = "\n\n".join(r.chunk.text for r in a.retrieved)
                texts[(cfg.id, a.question_id)] = {"question": a.question, "reference": cases[a.question_id].reference,
                                                 "excerpts": excerpts, "answer": a.text}
    return texts


def cmd_calibrate(args) -> int:
    if args.sample:
        rows = [json.loads(line) for p in SCORES.glob("*.jsonl") for line in p.read_text().splitlines() if line]
        out = sample_for_calibration(rows, _answer_texts(), CALIBRATION / "sample.csv", n=args.n)
        print(f"Blind sample written to {out}; grade it into {CALIBRATION / 'agent_grades.csv'}")
    return 0


def _meta() -> dict:
    meta = {"answer_model": ANSWER_MODEL, "judge_model": JUDGE_MODEL}
    lock = RAW / "lock.json"
    if lock.exists():
        meta["corpus_lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()[:12]
    ledger = BudgetLedger()
    meta["total_cost_usd"] = f"{ledger.spent:.2f}"
    meta["judge_cost_usd"] = f"{ledger.spent_on('evaluate'):.2f} (estimated)"
    return meta


def cmd_report(args) -> int:
    build_report(SCORES, Path("REPORT.md"), Path("report/metrics.png"),
                 agent_grades=CALIBRATION / "agent_grades.csv", meta=_meta())
    print("REPORT.md written")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eurag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="download, convert, chunk and index the corpus")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--skip-download", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("testset", help="generate candidate questions (paid) or merge the reviewed set")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_testset)

    for name, func, helptext in (("run", cmd_run, "answer the test set (paid)"),
                                 ("evaluate", cmd_evaluate, "score the answers (paid)")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--config", default="all")
        p.add_argument("--yes", action="store_true")
        if name == "run":
            p.add_argument("--limit", type=int, default=None)
        else:
            p.add_argument("--retry-errors", action="store_true", help="rescore rows whose judge call failed")
        p.set_defaults(func=func)

    p = sub.add_parser("calibrate", help="write a blind sample for the second grader")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--n", type=int, default=20)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("report", help="write REPORT.md and the chart")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)
