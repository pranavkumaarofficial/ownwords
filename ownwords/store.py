"""Sample storage. SQLite because it ships with Python and the whole corpus is
one file you can copy, diff and delete.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from .models import Sample

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id       TEXT PRIMARY KEY,
    text     TEXT NOT NULL,
    register TEXT NOT NULL,
    source   TEXT NOT NULL,
    date     TEXT,
    meta     TEXT NOT NULL DEFAULT '{}',
    words    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_register ON samples(register);
"""


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def add(self, samples: Iterable[Sample]) -> int:
        """Insert, skipping anything already stored. Returns the number added."""
        rows = [
            (s.id, s.text, s.register, s.source, s.date, json.dumps(s.meta), s.words)
            for s in samples
        ]
        before = self.count()
        self.db.executemany(
            "INSERT OR IGNORE INTO samples (id,text,register,source,date,meta,words)"
            " VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        self.db.commit()
        return self.count() - before

    def count(self, register: Optional[str] = None) -> int:
        q = "SELECT COUNT(*) FROM samples"
        args: tuple = ()
        if register:
            q += " WHERE register=?"
            args = (register,)
        return self.db.execute(q, args).fetchone()[0]

    def get(self, register: Optional[str] = None, min_words: int = 0) -> List[Sample]:
        q = "SELECT * FROM samples WHERE words >= ?"
        args: list = [min_words]
        if register:
            q += " AND register = ?"
            args.append(register)
        return [
            Sample(
                text=r["text"], register=r["register"], source=r["source"],
                date=r["date"], meta=json.loads(r["meta"]),
            )
            for r in self.db.execute(q, args)
        ]

    def registers(self) -> dict:
        return {
            r["register"]: {"samples": r["n"], "words": r["w"]}
            for r in self.db.execute(
                "SELECT register, COUNT(*) n, SUM(words) w FROM samples GROUP BY register"
            )
        }

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
