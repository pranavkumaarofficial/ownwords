"""AI-tell lint.

The style profile catches the wrong *shape*. This catches the specific
vocabulary and constructions that mark text as model-written regardless of
shape, which no percentile band will notice because the author's corpus has a
rate of roughly zero for all of them and zero is inside every band.

Each hit names a span, so the repair prompt can point at it instead of asking
for a vaguely more human rewrite.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Pattern, Tuple

# (name, pattern, what to do about it)
RULES: List[Tuple[str, Pattern, str]] = [
    ("significance-inflation", re.compile(
        r"\b(?:stands as|serves as|is a testament|underscor\w+|highlight\w+ the"
        r"|plays a (?:vital|crucial|key|significant|pivotal) role|marking a"
        r"|reflects? (?:a )?broader|represents a shift|turning point"
        r"|evolving landscape|indelible mark)\b", re.I),
     "delete the clause, or replace it with the concrete fact it is standing in for"),

    ("ing-tail", re.compile(
        r",\s+(?:highlighting|underscoring|emphasi[sz]ing|ensuring|reflecting"
        r"|symboli[sz]ing|contributing to|fostering|cultivating|showcasing"
        r"|demonstrating|enabling|allowing|paving)\b", re.I),
     "cut the trailing participle clause; it adds no information"),

    ("promo-adjective", re.compile(
        r"\b(?:robust|seamless|cutting-edge|groundbreaking|vibrant|profound"
        r"|breathtaking|nestled|boasts?|renowned|stunning|comprehensive"
        r"|state-of-the-art|unparalleled|invaluable)\b", re.I),
     "replace with a plain adjective or a measurement"),

    ("ai-vocab", re.compile(
        r"\b(?:delve|leverage|utili[sz]e|facilitate|myriad|plethora|realm"
        r"|landscape|tapestry|crucial|pivotal|moreover|furthermore"
        r"|additionally|consequently|nevertheless|holistic|synergy"
        r"|paradigm|intricate)\b", re.I),
     "use the ordinary word"),

    ("negative-parallelism", re.compile(
        r"\b(?:not just|not only|isn't just|it's not about)\b[^.?!]{0,80}?"
        r"\b(?:but|it's)\b", re.I),
     "state the point directly, without the setup"),

    ("rule-of-three", re.compile(
        r"\b\w+(?:ing|ity|ness|tion)\b,\s+\b\w+(?:ing|ity|ness|tion)\b,\s+and\s+"
        r"\b\w+(?:ing|ity|ness|tion)\b", re.I),
     "keep the one item that carries weight and drop the other two"),

    ("em-dash-aside", re.compile(r"—[^—]{3,60}—"),
     "use commas, or split into two sentences"),

    ("chatbot-artifact", re.compile(
        r"(?:I hope this helps|Let me know if|Feel free to|Great question"
        r"|Certainly[!,]|I'd be happy to|As an AI|In conclusion"
        r"|In today's\s+\w+\s+world|It'?s worth noting)", re.I),
     "delete"),

    ("hedge-stack", re.compile(
        r"\b(?:could|might|may)\s+(?:potentially|possibly|arguably)\b"
        r"|\bpotentially\s+(?:could|might)\b", re.I),
     "one hedge at most"),

    # Anchored to sentence starts, not line starts: these openers turn up
    # mid-paragraph far more often than at the top of one.
    ("filler-opener", re.compile(
        r"(?:\A|[.!?]\s+|\n\s*)(?:In order to|Due to the fact that"
        r"|It is important to note|It should be noted|When it comes to"
        r"|At its core|That said,|Needless to say|First and foremost)\b", re.I),
     "start with the subject of the sentence"),

    ("curly-quote", re.compile(r"[“”‘’]"),
     "straight quotes -- curly ones come from the model, not your keyboard"),

    ("emoji", re.compile(
        "[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF]"),
     "remove"),

    ("bold-label-list", re.compile(r"(?:^|\n)\s*[-*]?\s*\*\*[^*\n]{2,30}:?\*\*:?\s"),
     "write it as a sentence instead of a labelled bullet"),
]


# Typography, not phrasing. A model emits curly quotes and emoji, so these are
# worth flagging in *output*; but the author's own corpus is full of both (pasted
# text, and emoji he actually uses), so including them when measuring the corpus
# just manufactures false positives. Same split as SURFACE_KEYS in features.py:
# measure it, do not reproduce it, and do not judge the author by it.
TYPOGRAPHIC = frozenset({"curly-quote", "emoji"})


@dataclass
class Hit:
    rule: str
    span: str
    fix: str
    pos: int

    def describe(self) -> str:
        return '%s: "%s" -> %s' % (self.rule, self.span.strip()[:70], self.fix)


def lint(text: str, typography: bool = True) -> List[Hit]:
    """Find AI tells.

    typography=False drops the curly-quote and emoji rules, which is what you
    want when measuring the author's own writing rather than checking a rewrite.
    """
    hits: List[Hit] = []
    for name, pat, fix in RULES:
        if not typography and name in TYPOGRAPHIC:
            continue
        for m in pat.finditer(text):
            hits.append(Hit(name, m.group(0), fix, m.start()))
    hits.sort(key=lambda h: h.pos)
    return hits


# Measured on 80 held-out messages of the author's own writing against four
# AI-written passages and two human controls: his prose averages 0.03 hits per
# 100 words (95% of samples have none), AI prose runs 2.9 to 18.8, and the human
# controls score 0. At a threshold of 1.0 that is 4/4 AI caught, 1/80 false
# positive on his own writing, 0/2 on the controls.
# Reproduce: python -m tools.evaluate
ACCEPT_RATE = 1.0


def hit_rate(text: str, hits: Optional[List[Hit]] = None,
             typography: bool = True) -> float:
    """AI-tell hits per 100 words.

    This is the signal that actually separates the author's writing from a
    model's. The stylometric band score does not -- see tools/evaluate.py -- so
    it steers rewrites rather than judging them.
    """
    n = len(text.split())
    if not n:
        return 0.0
    hits = lint(text, typography=typography) if hits is None else hits
    return round(100.0 * len(hits) / n, 3)


def summary(hits: List[Hit]) -> str:
    if not hits:
        return "clean"
    from collections import Counter
    c = Counter(h.rule for h in hits)
    return ", ".join("%s x%d" % (k, v) for k, v in c.most_common())
