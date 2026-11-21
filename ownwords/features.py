"""Deterministic style measurement.

No model is involved here. Given text, this returns numbers. The same function
measures the corpus (to build a profile) and the rewritten output (to check it),
which is the only reason the check means anything -- if the metric that judges
the output were different from the metric that defined the target, the loop
would be comparing two things that were never the same.

Every feature is a rate per 100 words or a distribution statistic, so samples of
different lengths stay comparable.
"""
from __future__ import annotations

import re
import statistics as st
from typing import Dict, List, Sequence

# --- lexicons ---------------------------------------------------------------
# Small and hand-picked on purpose. A big generic list measures English; a small
# list aimed at the habits that actually vary between writers measures a writer.

HEDGES = {
    "maybe", "perhaps", "probably", "possibly", "somewhat", "fairly", "rather",
    "seems", "seem", "apparently", "arguably", "roughly", "about", "kind",
    "sort", "basically", "essentially", "generally", "usually", "mostly",
    "i think", "i guess", "i feel", "i believe", "not sure", "might",
}
INTENSIFIERS = {
    "very", "really", "extremely", "incredibly", "totally", "completely",
    "absolutely", "super", "so", "quite", "highly", "deeply", "massively",
}
FILLERS = {
    "actually", "literally", "obviously", "honestly", "frankly", "anyway",
    "well", "look", "right", "okay", "ok", "just", "even", "still", "though",
}
CONNECTIVES = {"and", "but", "so", "or", "because", "also", "plus", "then", "yet"}

# Deliberately matches both "don't" and "dont". Whether the apostrophe got typed
# is a keyboard fact; whether the writer contracts at all is a style fact, and
# only the second one is worth measuring or reproducing.
CONTRACTION_RE = re.compile(
    r"\b\w+['’](?:s|t|re|ve|ll|d|m)\b"
    # Only forms that are not also ordinary words. "its", "were", "ill" and
    # "lets" are excluded: counting them would measure vocabulary, not contraction.
    r"|\b(?:cant|dont|doesnt|didnt|isnt|arent|wasnt|werent|couldnt|wouldnt"
    r"|shouldnt|havent|hasnt|hadnt|im|ive|youre|youve|theyre|theyve"
    r"|thats|whats)\b",
    re.I,
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\s]+|\n+")
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.I)

# Surface metrics measure how fast the author was typing, not how he writes.
# They are still measured -- you want to know the corpus is full of them, and
# knowing that is how you avoid reproducing them -- but they are excluded from
# the constraints handed to the model and from the output check, unless asked
# for explicitly. Reproducing these would make output that reads as sloppy
# rather than as his.
SURFACE_KEYS = frozenset({"lower_start_rate"})

# Metrics where a *lower* value is not automatically better or worse; the
# comparison is always "inside the author's band or outside it".
METRIC_KEYS = (
    "sent_len_mean", "sent_len_median", "sent_len_sd", "sent_len_p90",
    "words_per_para", "lower_start_rate", "comma_rate", "semicolon_rate",
    "colon_rate", "emdash_rate", "question_rate", "exclaim_rate",
    "ellipsis_rate", "paren_rate", "contraction_rate", "first_person_rate",
    "hedge_rate", "intensifier_rate", "filler_rate", "init_connective_rate",
    "bullet_rate", "word_len_mean", "long_word_rate", "mattr",
)


def sentences(text: str) -> List[str]:
    """Split into sentences, preserving original casing and leading characters."""
    out = []
    for chunk in SENT_SPLIT_RE.split(text):
        chunk = chunk.strip()
        if chunk and WORD_RE.search(chunk):
            out.append(chunk)
    return out


def words(text: str) -> List[str]:
    return WORD_RE.findall(text)


def _syllables(word: str) -> int:
    n = len(VOWEL_GROUP_RE.findall(word))
    if word.lower().endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def _mattr(toks: Sequence[str], window: int = 100) -> float:
    """Moving-average type-token ratio. Plain TTR falls as text gets longer,
    which would make it a length measurement rather than a vocabulary one."""
    low = [t.lower() for t in toks]
    if len(low) < window:
        return len(set(low)) / len(low) if low else 0.0
    ratios = [
        len(set(low[i:i + window])) / window
        for i in range(0, len(low) - window + 1, max(1, window // 4))
    ]
    return st.mean(ratios)


def _phrase_hits(lowered: str, lexicon: set) -> int:
    """Count lexicon entries, handling the multi-word ones ("i think")."""
    n = 0
    single = set()
    for entry in lexicon:
        if " " in entry:
            n += lowered.count(entry)
        else:
            single.add(entry)
    for w in WORD_RE.findall(lowered):
        if w in single:
            n += 1
    return n


def measure(text: str) -> Dict[str, float]:
    """Return one style vector for one piece of text."""
    toks = words(text)
    n_w = len(toks)
    if n_w == 0:
        return {k: 0.0 for k in METRIC_KEYS}

    per100 = 100.0 / n_w
    sents = sentences(text)
    s_lens = [len(words(s)) for s in sents] or [n_w]
    paras = [p for p in re.split(r"\n\s*\n", text) if WORD_RE.search(p)] or [text]
    lowered = text.lower()

    # A sentence "starts lowercase" only if its first alphabetic character is a
    # lowercase letter -- bullets and quotes shouldn't count as capitalised.
    lower_starts = 0
    for s in sents:
        m = re.search(r"[A-Za-z]", s)
        if m and m.group(0).islower():
            lower_starts += 1

    init_conn = sum(
        1 for s in sents
        if (w := WORD_RE.findall(s)) and w[0].lower() in CONNECTIVES
    )

    f = {
        "sent_len_mean": st.mean(s_lens),
        "sent_len_median": float(st.median(s_lens)),
        "sent_len_sd": st.pstdev(s_lens) if len(s_lens) > 1 else 0.0,
        "sent_len_p90": float(sorted(s_lens)[min(len(s_lens) - 1, int(0.9 * len(s_lens)))]),
        "words_per_para": n_w / len(paras),
        "lower_start_rate": 100.0 * lower_starts / len(sents),
        "comma_rate": text.count(",") * per100,
        "semicolon_rate": text.count(";") * per100,
        "colon_rate": text.count(":") * per100,
        "emdash_rate": (text.count("—") + text.count(" - ")) * per100,
        "question_rate": text.count("?") * per100,
        "exclaim_rate": text.count("!") * per100,
        "ellipsis_rate": (text.count("...") + text.count("…")) * per100,
        "paren_rate": text.count("(") * per100,
        "contraction_rate": len(CONTRACTION_RE.findall(text)) * per100,
        "first_person_rate": sum(
            1 for w in toks if w.lower() in ("i", "me", "my", "mine", "myself")
        ) * per100,
        "hedge_rate": _phrase_hits(lowered, HEDGES) * per100,
        "intensifier_rate": _phrase_hits(lowered, INTENSIFIERS) * per100,
        "filler_rate": _phrase_hits(lowered, FILLERS) * per100,
        "init_connective_rate": 100.0 * init_conn / len(sents),
        "bullet_rate": len(BULLET_RE.findall(text)) * per100,
        "word_len_mean": st.mean(len(w) for w in toks),
        "long_word_rate": 100.0 * sum(1 for w in toks if _syllables(w) >= 3) / n_w,
        "mattr": _mattr(toks),
    }
    return {k: round(float(v), 4) for k, v in f.items()}


# --- function-word fingerprint ---------------------------------------------
# Content words tell you the topic. Function-word sequences tell you the person,
# which is why authorship attribution has leaned on them since long before
# neural models.

FUNCTION_WORDS = {
    "the", "a", "an", "and", "but", "or", "so", "if", "then", "than", "that",
    "this", "these", "those", "of", "to", "in", "on", "at", "for", "with",
    "from", "by", "as", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "can", "could", "will",
    "would", "should", "may", "might", "must", "not", "no", "it", "its",
    "i", "me", "my", "you", "your", "we", "our", "they", "them", "there",
    "what", "which", "who", "how", "when", "where", "why", "all", "any",
    "some", "more", "most", "very", "just", "only", "also", "even", "like",
    "want", "need", "make", "get", "know", "think", "see", "use",
}


def function_ngrams(text: str, n: int = 2) -> List[str]:
    """n-grams over function words only, so topic drops out."""
    seq = [w.lower() for w in words(text)]
    keep = [w if w in FUNCTION_WORDS else "·" for w in seq]
    grams = []
    for i in range(len(keep) - n + 1):
        g = keep[i:i + n]
        if all(x == "·" for x in g):
            continue
        grams.append(" ".join(g))
    return grams
