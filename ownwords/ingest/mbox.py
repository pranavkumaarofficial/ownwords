"""Sent mail -> email-register samples.

Gmail: Takeout exports Mail as .mbox. Point this at it with your own address and
it keeps only what you wrote, with quoted replies and signatures cut off --
quoted text is someone else's voice, and leaving it in would profile them.
"""
from __future__ import annotations

import mailbox
import re
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import List, Optional

from ..models import Sample

# Everything from the first of these onward is not the author's new prose.
CUTOFFS = [
    re.compile(r"^On .{10,80}wrote:\s*$", re.M),
    re.compile(r"^-{2,}\s*Original Message\s*-{2,}", re.M | re.I),
    re.compile(r"^_{10,}\s*$", re.M),
    re.compile(r"^--\s*$", re.M),                 # signature delimiter
    re.compile(r"^(?:Sent from my|Get Outlook for)\b", re.M | re.I),
    re.compile(r"^From:\s*\S+@", re.M),
]
QUOTED = re.compile(r"^\s*>.*$", re.M)
URL = re.compile(r"https?://\S+")


def _body(msg: mailbox.mboxMessage) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, "replace")
        return ""
    payload = msg.get_payload(decode=True)
    if not payload:
        return str(msg.get_payload())
    return payload.decode(msg.get_content_charset() or "utf-8", "replace")


def _clean(body: str) -> str:
    body = QUOTED.sub("", body)
    cut = len(body)
    for pat in CUTOFFS:
        m = pat.search(body)
        if m:
            cut = min(cut, m.start())
    return re.sub(r"\n{3,}", "\n\n", body[:cut]).strip()


def from_mbox(
    path: Path,
    my_address: str,
    register: str = "email",
    min_words: int = 30,
    max_words: int = 600,
    limit: Optional[int] = None,
) -> List[Sample]:
    """Messages you sent, cleaned.

    my_address is matched case-insensitively against the From header, so an
    All-Mail export works -- you do not need a Sent-only mbox.
    """
    me = my_address.lower()
    out: List[Sample] = []
    for msg in mailbox.mbox(str(Path(path))):
        if me not in (msg.get("From") or "").lower():
            continue
        text = _clean(_body(msg))
        # A mail that is mostly a link is a forward, not writing.
        if URL.sub("", text).split().__len__() < min_words:
            continue
        if len(text.split()) > max_words:
            continue
        date = None
        try:
            date = parsedate_to_datetime(msg.get("Date")).date().isoformat()
        except (TypeError, ValueError):
            pass
        out.append(Sample(
            text=text, register=register, source="mbox",
            date=date, meta={"subject": (msg.get("Subject") or "")[:120]},
        ))
        if limit and len(out) >= limit:
            break
    return out
