"""Plain files -> samples.

This is the adapter for seeding a register by hand, which is how every register
other than chat gets populated. Paste ten sent emails into one file separated by
`---`, and the email register exists.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from ..models import Sample

SEPARATOR = re.compile(r"^\s*-{3,}\s*$", re.M)
FENCE = re.compile(r"```.*?```", re.S)
FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)


def from_textfiles(
    path: Path,
    register: str,
    split: bool = True,
    min_words: int = 25,
) -> List[Sample]:
    """One file, or a directory of files. With split=True each file is cut on
    lines of `---` so a single file can hold many samples."""
    p = Path(path)
    files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]

    out: List[Sample] = []
    for f in files:
        try:
            raw = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        raw = FRONTMATTER.sub("", raw)
        chunks = SEPARATOR.split(raw) if split else [raw]
        for i, chunk in enumerate(chunks):
            text = chunk.strip()
            if len(text.split()) < min_words:
                continue
            out.append(Sample(
                text=text,
                register=register,
                source="file:%s" % f.name,
                date=None,
                meta={"chunk": i} if len(chunks) > 1 else {},
            ))
    return out


def from_markdown_dir(
    path: Path,
    register: str = "longform",
    strip_code: bool = True,
    min_words: int = 60,
) -> List[Sample]:
    """A directory of .md files, one sample each.

    Code fences are removed by default: a post that is half shell commands would
    otherwise report a vocabulary and punctuation profile belonging to bash.
    """
    out: List[Sample] = []
    for f in sorted(Path(path).rglob("*.md")):
        raw = FRONTMATTER.sub("", f.read_text(encoding="utf-8", errors="replace"))
        if strip_code:
            raw = FENCE.sub("", raw)
        text = raw.strip()
        if len(text.split()) < min_words:
            continue
        out.append(Sample(
            text=text, register=register, source="md:%s" % f.name,
            meta={"path": str(f)},
        ))
    return out
