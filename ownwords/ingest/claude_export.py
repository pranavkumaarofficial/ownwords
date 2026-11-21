"""Claude data export -> chat-register samples.

The filtering here matters more than the parsing. A raw export is mostly not
writing: it is pasted code, pasted error output, pasted documents, and one-word
replies. Feeding that to a style profile measures the code you pasted.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, List

from ..models import Sample

CODEY = re.compile(
    r"```|^\s*(?:def|class|import|from|function|const|let|SELECT|FROM)\s"
    r"|Traceback \(most recent call last\)|^\s*(?:at |File \")",
    re.M,
)
# A message that is mostly a paste has a low ratio of prose lines to total lines.
PASTE_LINE = re.compile(r"^[\s{}\[\]()<>;,:=|/\\#*+-]*$|^\s*\S{40,}\s*$")


def _text_of(msg: dict) -> str:
    parts = [
        b.get("text", "") for b in (msg.get("content") or [])
        if b.get("type") == "text"
    ]
    return ("\n".join(p for p in parts if p.strip()) or msg.get("text") or "").strip()


def _looks_like_writing(text: str, min_words: int, max_words: int) -> bool:
    n = len(text.split())
    if n < min_words or n > max_words:
        return False
    if CODEY.search(text):
        return False
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) > 6:
        junk = sum(1 for l in lines if PASTE_LINE.match(l))
        if junk / len(lines) > 0.3:
            return False
    # Needs sentence-ish structure. A bare list of filenames is not a voice sample.
    if not re.search(r"[a-z]{3,}\s+[a-z]{3,}\s+[a-z]{3,}", text, re.I):
        return False
    return True


def from_claude_export(
    path: Path,
    register: str = "chat",
    min_words: int = 15,
    max_words: int = 400,
) -> List[Sample]:
    """Parse conversations.json and return the author's own messages worth keeping.

    min_words is 15 rather than 5: below that there is no sentence rhythm to
    learn, and the short messages are overwhelmingly "yes", "go on", "try again".
    max_words caps the long pasted specs, which are structurally unlike anything
    you would write from scratch.
    """
    convos = json.loads(Path(path).read_text(encoding="utf-8"))
    convos.sort(key=lambda c: c.get("created_at") or "")

    out: List[Sample] = []
    seen = set()
    for c in convos:
        title = c.get("name") or "(untitled)"
        for m in c.get("chat_messages") or []:
            if m.get("sender") != "human":
                continue
            text = _text_of(m)
            if not _looks_like_writing(text, min_words, max_words):
                continue
            key = text[:200]
            if key in seen:          # retries and regenerations
                continue
            seen.add(key)
            out.append(Sample(
                text=text,
                register=register,
                source="claude-export",
                date=(m.get("created_at") or "")[:10] or None,
                meta={"conversation": title},
            ))
    return out


def iter_all_messages(path: Path) -> Iterator[str]:
    """Unfiltered, for inspecting what the filter threw away."""
    for c in json.loads(Path(path).read_text(encoding="utf-8")):
        for m in c.get("chat_messages") or []:
            if m.get("sender") == "human":
                t = _text_of(m)
                if t:
                    yield t
