"""Does the pipeline's judgement actually work?

Two signals claim to tell the author's writing from a model's. This measures
both on held-out data, because a style tool that cannot be evaluated is a tool
that cannot be debugged -- you would never know whether a rewrite got better or
just different.

Method: hold out 80 of the author's own messages (>= 40 words), build the profile
from the rest, and score both the held-out set and a set of AI-written passages.
A usable signal puts a gap between them. The last negative is deliberately
good human prose that is not the author's, to check the signal is detecting
machine writing rather than just penalising competent writing.

    python -m tools.evaluate
"""
from __future__ import annotations

import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ownwords import features as F           # noqa: E402
from ownwords import lint as L               # noqa: E402
from ownwords.profile import Band, Profile, _pct   # noqa: E402
from ownwords.paths import corpus_db         # noqa: E402
from ownwords.store import Store             # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HELD_OUT = 80
MIN_WORDS = 40
SEED = 0

AI_NEGATIVES = [
    # 1: a cold email, the register this tool exists to fix
    "Dear Professor Chen,\n\nI hope this email finds you well. I am writing to "
    "express my profound interest in the research opportunities available within "
    "your laboratory, which stands as a testament to the cutting-edge work being "
    "conducted in distributed systems. In today's rapidly evolving technological "
    "landscape, I believe my background positions me as a robust candidate. "
    "During my two years as a software engineer, I had the opportunity to delve "
    "into a myriad of complex challenges, leveraging both theory and practice to "
    "deliver seamless solutions. It's not just my technical skills that I bring, "
    "but a genuine passion for inquiry, collaboration, and innovation.",
    # 2: technical description
    "The implementation leverages a robust architecture that facilitates seamless "
    "integration across multiple components. Moreover, comprehensive testing "
    "ensures reliability throughout the entire pipeline, underscoring our "
    "commitment to quality engineering practices and maintainable code.",
    # 3: listicle register, fewer vocabulary tells on purpose
    "It is important to note that this approach offers several key advantages. "
    "First, it significantly reduces complexity. Second, it enhances "
    "maintainability. Third, it improves overall system performance, thereby "
    "contributing to a better developer experience.",
    # 4: the project-announcement register
    "In conclusion, the project represents a pivotal step forward in the evolving "
    "landscape of developer tooling. By combining innovation, usability, and "
    "performance, it stands as a testament to what thoughtful design can achieve "
    "in today's competitive environment.",
]

# Not the author, not a model: a person writing plainly. The signal must stay
# quiet here, or it is measuring "well edited" rather than "machine written".
HUMAN_CONTROL = [
    "I spent the morning tracking down why the readiness probe was flapping. It "
    "turned out the health endpoint opens a database connection per request, and "
    "under the probe's one-second interval the pool was exhausted within a "
    "minute. I changed the probe to hit a static endpoint and the flapping "
    "stopped.",
    "We shipped the migration on Tuesday and rolled it back on Wednesday. The "
    "index rebuild held a lock the analytics job needed, and nobody noticed in "
    "staging because staging has no analytics job. That is the actual bug.",
]


def band_profile(samples, lo: float, hi: float) -> Profile:
    """Build a profile at arbitrary percentiles, to compare band widths."""
    vecs = [F.measure(s.text) for s in samples]
    ws = [s.words for s in samples]
    bands, overall = {}, {}
    for k in F.METRIC_KEYS:
        ser = sorted(v[k] for v in vecs)
        bands[k] = Band(_pct(ser, lo), _pct(ser, 0.5), _pct(ser, hi),
                        max(0.10 * _pct(ser, 0.99), 1e-3))
        overall[k] = sum(v[k] * w for v, w in zip(vecs, ws)) / sum(ws)
    return Profile("chat", bands, [], len(samples), sum(ws), overall)


def describe(name: str, values) -> str:
    v = sorted(values)
    return "%-22s n=%-4d mean %6.2f  median %6.2f  p10 %6.2f  p90 %6.2f" % (
        name, len(v), st.mean(v), st.median(v),
        v[int(0.1 * len(v))], v[int(0.9 * (len(v) - 1))],
    )


def main() -> int:
    db = corpus_db()
    if not db.exists():
        print("no corpus at %s -- run ownwords ingest first" % db)
        return 1

    with Store(db) as store:
        samples = [s for s in store.get(register="chat") if s.words >= MIN_WORDS]
    if len(samples) < HELD_OUT * 2:
        print("need at least %d samples of >=%d words, have %d"
              % (HELD_OUT * 2, MIN_WORDS, len(samples)))
        return 1

    random.seed(SEED)
    random.shuffle(samples)
    held, train = samples[:HELD_OUT], samples[HELD_OUT:]
    print("corpus: %d samples >=%dw -- %d train, %d held out\n"
          % (len(samples), MIN_WORDS, len(train), len(held)))

    # --- signal 1: AI-tell rate -------------------------------------------
    print("=== AI-tell rate (hits per 100 words) ===")
    # typography=False: see lint.TYPOGRAPHIC for why
    own = [L.hit_rate(s.text, typography=False) for s in held]
    print(describe("author, held out", own))
    print("%-22s %s" % ("AI passages",
                        "  ".join("%.2f" % L.hit_rate(t) for t in AI_NEGATIVES)))
    print("%-22s %s" % ("human control",
                        "  ".join("%.2f" % L.hit_rate(t) for t in HUMAN_CONTROL)))
    zero = sum(1 for x in own if x == 0.0)
    print("author samples with zero tells: %d/%d (%.0f%%)"
          % (zero, len(own), 100.0 * zero / len(own)))
    caught = sum(1 for t in AI_NEGATIVES if L.hit_rate(t) > L.ACCEPT_RATE)
    false_pos = sum(1 for x in own if x > L.ACCEPT_RATE)
    control_fp = sum(1 for t in HUMAN_CONTROL if L.hit_rate(t) > L.ACCEPT_RATE)
    print("at threshold %.1f: %d/%d AI caught, %d/%d author false-positive, "
          "%d/%d control false-positive"
          % (L.ACCEPT_RATE, caught, len(AI_NEGATIVES), false_pos, len(own),
             control_fp, len(HUMAN_CONTROL)))

    # --- signal 2: stylometric band score ---------------------------------
    print("\n=== stylometric band score (0-100) ===")
    for lo, hi in ((0.10, 0.90), (0.25, 0.75)):
        prof = band_profile(train, lo, hi)
        own_s = [prof.score(s.text) for s in held]
        ai_s = [prof.score(t) for t in AI_NEGATIVES]
        print(describe("p%d-p%d author" % (lo * 100, hi * 100), own_s))
        print("%-22s %s" % ("  AI passages",
                            "  ".join("%.1f" % x for x in ai_s)))
        thr = sorted(own_s)[int(0.1 * len(own_s))]
        below = sum(1 for x in ai_s if x < thr)
        print("  author 10th pct = %.1f -> AI below it: %d/%d"
              % (thr, below, len(ai_s)))

    print("""
Conclusion: the tell rate separates cleanly and the band score does not -- AI
passages land inside the author's own score distribution at either band width.
So the band score is used to generate numeric feedback for a repair pass, where
specific numbers do change what the model produces, and the tell rate is what
decides whether a rewrite is accepted.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
