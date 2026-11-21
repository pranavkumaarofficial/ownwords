"""Build a style profile for one register, and check text against it.

A profile is a band per metric, not a single number. "Your sentences average
14 words" is close to useless as a target, because writing where every sentence
is 14 words long is unmistakably machine-made. The band ([p10, p90] of the
author's own samples) carries the variation, which is the part that reads as
human.
"""
from __future__ import annotations

import json
import statistics as st
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import features as F
from .models import Sample

# Below this, a sample's rates are dominated by rounding -- one comma in nine
# words is 11 per 100 and means nothing.
MIN_WORDS_FOR_PROFILE = 25

# A metric only gets reported as off when it misses the band by more than this
# fraction of the band's own width. Without it, every metric with a narrow band
# fires on every rewrite and the report becomes noise.
TOLERANCE = 0.35


# A single metric can miss by an unbounded amount (one em dash in a corpus with
# none is a divide-by-almost-zero). Capping keeps one metric from deciding the
# whole score, while still ranking it as the worst offender.
MAX_MISS = 10.0


@dataclass
class Band:
    p10: float
    p50: float
    p90: float
    floor: float = 1e-3   # minimum meaningful width, set at build time

    @property
    def width(self) -> float:
        """Band width, floored.

        Many of these metrics are sparse: a writer who never uses semicolons has
        p10 = p90 = 0, and dividing by that width would report a single semicolon
        as an infinite deviation. The floor is a tenth of the largest value the
        author ever produced for this metric, so "never" still registers as a
        large miss without being an undefined one.
        """
        return max(self.p90 - self.p10, self.floor)

    def miss(self, value: float) -> float:
        """0.0 inside the band; otherwise distance outside, in band widths."""
        if value < self.p10:
            return min((self.p10 - value) / self.width, MAX_MISS)
        if value > self.p90:
            return min((value - self.p90) / self.width, MAX_MISS)
        return 0.0


@dataclass
class Deviation:
    metric: str
    value: float
    band: Band
    miss: float

    @property
    def direction(self) -> str:
        return "low" if self.value < self.band.p10 else "high"

    def describe(self) -> str:
        return "%s is %s: %.1f, you run %.1f-%.1f" % (
            self.metric, self.direction, self.value, self.band.p10, self.band.p90
        )


class Profile:
    def __init__(self, register: str, bands: Dict[str, Band],
                 ngrams: Sequence[str], n_samples: int, n_words: int,
                 overall: Optional[Dict[str, float]] = None):
        self.register = register
        self.bands = bands
        self.ngrams = list(ngrams)
        self.n_samples = n_samples
        self.n_words = n_words
        # Corpus-level rates, word-weighted. These answer "how often does he do
        # this per 100 words of writing", which is not what the median of the
        # per-sample rates answers: he hedges in roughly a third of his messages,
        # so the median message hedges zero times and a median-derived
        # instruction would read "never hedge". Sparse habits need this.
        self.overall = overall or {k: bands[k].p50 for k in bands}

    # --- construction ------------------------------------------------------
    @classmethod
    def build(cls, samples: List[Sample], register: str) -> "Profile":
        usable = [s for s in samples if s.words >= MIN_WORDS_FOR_PROFILE]
        if len(usable) < 15:
            raise ValueError(
                "register %r has %d samples of >=%d words; need at least 15 for a "
                "profile that isn't just noise" % (register, len(usable), MIN_WORDS_FOR_PROFILE)
            )
        vectors = [F.measure(s.text) for s in usable]
        weights = [s.words for s in usable]
        total_w = sum(weights)

        bands, overall = {}, {}
        for key in F.METRIC_KEYS:
            series = sorted(v[key] for v in vectors)
            bands[key] = Band(
                p10=_pct(series, 0.10), p50=_pct(series, 0.50), p90=_pct(series, 0.90),
                # Scale reference is p99 rather than the max, so one pathological
                # sample does not set the floor for everything.
                floor=max(0.10 * _pct(series, 0.99), 1e-3),
            )
            overall[key] = round(
                sum(v[key] * w for v, w in zip(vectors, weights)) / total_w, 4
            )

        counts: Counter = Counter()
        for s in usable:
            counts.update(set(F.function_ngrams(s.text, 2)))
        # Kept by how many *samples* use them, so one long rant can't dominate.
        top = [g for g, _ in counts.most_common(40)]
        return cls(register, bands, top, len(usable), total_w, overall)

    # --- use ---------------------------------------------------------------
    def check(self, text: str, tolerance: float = TOLERANCE,
              surface: bool = False) -> List[Deviation]:
        """Which metrics in this text fall outside the author's range.

        Surface metrics are skipped by default: correctly capitalised output
        should not be reported as a deviation just because the corpus is
        lowercase.
        """
        v = F.measure(text)
        devs = []
        for key, band in self.bands.items():
            if key in F.SURFACE_KEYS and not surface:
                continue
            miss = band.miss(v[key])
            if miss > tolerance:
                devs.append(Deviation(key, v[key], band, miss))
        devs.sort(key=lambda d: -d.miss)
        return devs

    def score(self, text: str) -> float:
        """0-100. 100 means every metric sits inside the band. Useful for
        comparing two candidate rewrites; not meaningful in the absolute."""
        v = F.measure(text)
        misses = [
            self.bands[k].miss(v[k]) for k in self.bands
            if k not in F.SURFACE_KEYS
        ]
        return round(100.0 * max(0.0, 1.0 - st.mean(misses)), 1)

    # --- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "register": self.register,
            "n_samples": self.n_samples,
            "n_words": self.n_words,
            "bands": {k: vars(b) for k, b in self.bands.items()},
            "overall": self.overall,
            "function_ngrams": self.ngrams,
        }

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Profile":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            register=d["register"],
            bands={k: Band(**v) for k, v in d["bands"].items()},
            ngrams=d.get("function_ngrams", []),
            n_samples=d["n_samples"],
            n_words=d["n_words"],
            overall=d.get("overall"),
        )

    # --- the part the model reads -----------------------------------------
    def as_constraints(self) -> str:
        """The profile as instructions. Only the metrics a writer can actually
        act on -- telling a model to hit an exact MATTR produces nothing."""
        b, o = self.bands, self.overall
        lines = [
            "Sentence length: average %.0f words, but vary it. Your own range runs"
            " %.0f-%.0f words per sentence, and the spread matters more than the"
            " average." % (b["sent_len_mean"].p50, b["sent_len_p90"].p10, b["sent_len_p90"].p90),
            "Paragraphs: about %.0f words." % b["words_per_para"].p50,
            "Commas: roughly %.1f per 100 words. Semicolons: %s. Em dashes: %s."
            % (o["comma_rate"], _freq(o["semicolon_rate"]), _freq(o["emdash_rate"])),
            "Contractions: %s." % _freq(o["contraction_rate"]),
            "First person (I/me/my): %.1f per 100 words." % o["first_person_rate"],
            "Hedging (maybe, probably, i think, basically): %s."
            % _freq(o["hedge_rate"]),
            "Intensifiers (really, very, so): %s." % _freq(o["intensifier_rate"]),
            "Discourse fillers (just, actually, honestly, still): %s."
            % _freq(o["filler_rate"]),
            "Sentences opening with and/but/so: %.0f%% of them."
            % o["init_connective_rate"],
            "Words of three or more syllables: %.0f%%." % o["long_word_rate"],
        ]
        if o["lower_start_rate"] > 20:
            lines.append(
                "Ignore the samples' typing shortcuts. %.0f%% of their sentences open"
                " lowercase and apostrophes are often missing, because they were typed"
                " fast. That is not style. Write correct, standard English and copy the"
                " structure instead." % o["lower_start_rate"]
            )
        if o["bullet_rate"] < 0.3:
            lines.append("Bullet lists: you almost never use them here. Write prose.")
        return "\n".join("- " + l for l in lines)

    def as_markdown(self) -> str:
        """Human-readable. This is the file to read when the output feels wrong
        and you want to know what the pipeline thinks your voice is."""
        rows = [
            "# Style profile: %s" % self.register,
            "",
            "Built from %d samples, %s words. Bands are the 10th-90th percentile of"
            " your own samples." % (self.n_samples, f"{self.n_words:,}"),
            "",
            "| metric | p10 | median | p90 | corpus rate |",
            "| --- | --- | --- | --- | --- |",
        ]
        for k in F.METRIC_KEYS:
            b = self.bands[k]
            rows.append("| %s | %.2f | %.2f | %.2f | %.2f |"
                        % (k, b.p10, b.p50, b.p90, self.overall.get(k, 0.0)))
        rows += ["", "## Constraints handed to the model", "", self.as_constraints()]
        rows += [
            "", "## Most frequent function-word bigrams", "",
            "Content words are masked as `·`, so these are sentence skeletons"
            " rather than topics.", "",
            ", ".join("`%s`" % g for g in self.ngrams[:25]),
        ]
        return "\n".join(rows)


def _freq(rate: float) -> str:
    """Phrase a sparse rate the way a writer can act on it. "0.3 per 100 words"
    invites the model to produce exactly zero; "about once every 300 words" does
    not."""
    if rate <= 0.02:
        return "never"
    if rate >= 1.0:
        return "%.1f per 100 words" % rate
    return "about once every %d words" % (round(100.0 / rate / 10) * 10)


def _pct(sorted_series: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. statistics.quantiles needs n>=2 and
    returns cut points rather than arbitrary q, so this is simpler here."""
    if not sorted_series:
        return 0.0
    if len(sorted_series) == 1:
        return float(sorted_series[0])
    pos = q * (len(sorted_series) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_series) - 1)
    frac = pos - lo
    return round(float(sorted_series[lo] * (1 - frac) + sorted_series[hi] * frac), 4)
