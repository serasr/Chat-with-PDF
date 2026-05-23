# Layer 1: Observability & Tracing
#
# Every request goes through a Trace object.
# Each step (retrieval, LLM) gets its own Span.
# Results are written to logs/traces.jsonl and logs/traces.db

import time
import json
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime

# ── Log destinations ──────────────────────────────────────────────────────────
LOGS_DIR   = Path("logs")
JSONL_PATH = LOGS_DIR / "traces.jsonl"
DB_PATH    = LOGS_DIR / "traces.db"

LOGS_DIR.mkdir(exist_ok=True)


# ── Database setup ────────────────────────────────────────────────────────────
# Creates the traces table if it doesn't exist yet
# Called once when tracer.py is first imported
def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            id                TEXT PRIMARY KEY,
            timestamp         TEXT,
            query             TEXT,
            answer            TEXT,
            latency_total     REAL,
            latency_retrieval REAL,
            latency_llm       REAL,
            chunks_retrieved  INTEGER,
            top_chunk_score   REAL,
            token_estimate    INTEGER,
            error             TEXT
        )
    """)
    con.commit()
    con.close()

_init_db()


# ── Span ──────────────────────────────────────────────────────────────────────
class Span:
    """
    Tracks one step inside a Trace.
    Used as a context manager:

        with trace.span("retrieval") as s:
            # do retrieval
            s.metadata["chunks"] = 8
    """
    def __init__(self, name: str):
        self.name     = name
        self.start    = None
        self.duration = None
        self.metadata = {}
        self.error    = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = round(time.perf_counter() - self.start, 4)
        if exc_type:
            self.error = str(exc_val)
        return False  # don't suppress exceptions


# ── Trace ─────────────────────────────────────────────────────────────────────
class Trace:
    """
    One full request lifecycle.
    Created at the start of every user question.
    Finished when the answer is returned.
    """
    def __init__(self, query: str):
        self.id            = str(uuid.uuid4())[:8]
        self.timestamp     = datetime.utcnow().isoformat()
        self.query         = query
        self.spans         = {}
        self.answer        = None
        self.error         = None
        self.latency_total = None
        self._start        = time.perf_counter()

    def span(self, name: str) -> Span:
        """Create and register a new span."""
        s = Span(name)
        self.spans[name] = s
        return s

    def finish(self, answer: str = None, error: str = None):
        """Call this when the request is complete."""
        self.answer        = answer
        self.error         = error
        self.latency_total = round(time.perf_counter() - self._start, 4)
        self._write()

    # ── Internal ──────────────────────────────────────────────────────────────
    def _to_dict(self) -> dict:
        return {
            "id":            self.id,
            "timestamp":     self.timestamp,
            "query":         self.query,
            "answer":        self.answer,
            "latency_total": self.latency_total,
            "spans": {
                name: {
                    "duration": s.duration,
                    "metadata": s.metadata,
                    "error":    s.error,
                }
                for name, s in self.spans.items()
            },
            "error": self.error,
        }

    def _write(self):
        d = self._to_dict()

        # 1. Append to JSONL file — one line per request
        with open(JSONL_PATH, "a") as f:
            f.write(json.dumps(d) + "\n")

        # 2. Insert into SQLite
        r = self.spans.get("retrieval", Span("retrieval"))
        l = self.spans.get("llm",       Span("llm"))

        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                d["id"],
                d["timestamp"],
                d["query"],
                d["answer"],
                d["latency_total"],
                r.duration,
                l.duration,
                r.metadata.get("chunks_retrieved"),
                r.metadata.get("top_score"),
                l.metadata.get("token_estimate"),
                d["error"],
            )
        )
        con.commit()
        con.close()

        # 3. Print summary to terminal
        self._print_summary()

    def _print_summary(self):
        r = self.spans.get("retrieval")
        l = self.spans.get("llm")
        print("\n" + "─" * 60)
        print(f"[TRACE {self.id}]  {self.timestamp}")
        print(f"  Query     : {self.query[:80]}")
        if r:
            print(f"  Retrieval : {r.duration}s  |  chunks={r.metadata.get('chunks_retrieved')}  |  top_score={r.metadata.get('top_score')}")
        if l:
            print(f"  LLM       : {l.duration}s  |  ~{l.metadata.get('token_estimate')} tokens")
        print(f"  Total     : {self.latency_total}s")
        if self.error:
            print(f"  ERROR     : {self.error}")
        print("─" * 60)