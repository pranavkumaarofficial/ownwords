"""Exemplar retrieval.

Few-shot style transfer works better when the examples are about something near
the input, because the model copies phrasing it can see being used on similar
material rather than transplanting it. BM25 over the samples in one register is
enough for that, and it needs no model, no embedding service and no network.

Written out rather than pulled from rank_bm25: it is twenty lines, and a corpus
of personal writing is not a dependency worth adding.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence, Tuple

from .models import Sample

STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "is", "are", "was", "be", "it", "this", "that", "i", "you", "my",
    "we", "as", "by", "from", "so", "if", "not", "do", "does", "can", "will",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75


def _tok(text: str) -> List[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP and len(t) > 2]


class BM25:
    def __init__(self, docs: Sequence[Sequence[str]]):
        self.docs = [Counter(d) for d in docs]
        self.lens = [sum(c.values()) for c in self.docs]
        self.avg_len = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {
            t: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for t, freq in df.items()
        }

    def scores(self, query: Sequence[str]) -> List[float]:
        out = []
        for c, dl in zip(self.docs, self.lens):
            s = 0.0
            for t in query:
                tf = c.get(t)
                if not tf:
                    continue
                denom = tf + K1 * (1 - B + B * dl / (self.avg_len or 1))
                s += self.idf.get(t, 0.0) * tf * (K1 + 1) / denom
            out.append(s)
        return out


class Retriever:
    """Picks exemplars for one register.

    min_words exists because very short samples teach the model nothing about
    sentence rhythm, and the whole point of an exemplar is rhythm.
    """

    def __init__(self, samples: Sequence[Sample], min_words: int = 30):
        self.samples = [s for s in samples if s.words >= min_words]
        if not self.samples:
            raise ValueError("no samples of >= %d words to retrieve from" % min_words)
        self.index = BM25([_tok(s.text) for s in self.samples])

    def top(self, text: str, k: int = 5, max_words: int = 350) -> List[Sample]:
        """Best k exemplars for this input.

        k defaults to 5: measured returns flatten past four or five examples, and
        every extra one costs tokens on every single rewrite.
        """
        scored: List[Tuple[float, int]] = list(
            zip(self.index.scores(_tok(text)), range(len(self.samples)))
        )
        scored.sort(key=lambda p: -p[0])

        picked: List[Sample] = []
        budget = max_words
        for score, i in scored:
            s = self.samples[i]
            if s.words > budget:
                continue
            picked.append(s)
            budget -= s.words
            if len(picked) >= k:
                break

        # A query with no lexical overlap scores zero everywhere; fall back to the
        # longest samples so the model still sees real writing instead of nothing.
        if not picked:
            picked = sorted(self.samples, key=lambda s: -s.words)[:k]
        return picked
