"""Core data types.

A Sample is one piece of writing known to be the author's own. Register is the
situation it was written for; style varies more across registers than it does
across topics, so every sample carries one and nothing is ever mixed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

# Registers are deliberately coarse. Finer splits starve each bucket of samples.
REGISTERS = (
    "chat",       # instructions to an agent, terse, lowercase, imperative
    "email",      # addressed to a person who can say no
    "longform",   # essays, SOPs, assignments, reports
    "notes",      # for yourself only
    "comment",    # issues, PRs, code review, forum replies
)


@dataclass
class Sample:
    text: str
    register: str
    source: str                      # where it came from, for provenance
    date: Optional[str] = None       # ISO date, if known
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.register not in REGISTERS:
            raise ValueError(
                "unknown register %r, expected one of %s" % (self.register, ", ".join(REGISTERS))
            )
        self.text = self.text.strip()

    @property
    def id(self) -> str:
        """Content hash, scoped to the register.

        Re-ingesting the same export twice is a no-op. The register is part of
        the hash because the same sentence written in two registers is two
        samples: without it, seeding the email register with a line that already
        exists in the chat corpus would silently drop it on the primary key.
        """
        key = "%s\x00%s" % (self.register, self.text)
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    @property
    def words(self) -> int:
        return len(self.text.split())

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["id"] = self.id
        return d
