"""The rewrite loop.

One pass of "rewrite this in my voice" gives you text that is better than the
input and still not yours, and you have no way of knowing which. So the loop is:

    rewrite -> measure the output against the profile -> if it misses, say which
    metric missed by how much and rewrite again

The repair prompt carries numbers, not adjectives. "Your sentences average 24
words, you run 9-18" is actionable; "be more natural" is not, and a second pass
on that instruction mostly just shuffles synonyms.

The loop stops at max_passes whether or not it converged, and returns the best
candidate it saw along with the deviations that survived. A rewrite that cannot
be made to fit is information -- usually it means the input says something you
would never say, and no amount of restyling fixes that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from . import lint as L
from .llm import LLM
from .models import Sample
from .profile import Deviation, Profile
from .retrieve import Retriever

SYSTEM = """You rewrite text so that it reads as if one specific person wrote it.

You are given that person's measured writing habits, real samples of their \
writing, and a draft. The draft's meaning is fixed: keep every claim, number, \
name, link and commitment in it, and do not add any. Change only how it is said.

Rules that override everything else:
- Never invent a fact, figure or example to make a sentence sound more lived-in.
- If the draft makes a claim the person could not support, keep it as it is and \
list it at the end under NOTES. Do not quietly soften it.
- Copy the samples' structure: sentence length and its variation, punctuation \
habits, how directly they make a point. Do not copy their typing errors: \
lowercase sentence openings and missing apostrophes are keyboard artefacts, \
not style. Output is correctly spelled and capitalised.
- Do not imitate the samples' subject matter, opinions or specific phrases. You \
are copying how they write, not what they said.
- Output the rewritten text and nothing else. No preamble, no explanation, no \
markdown fence. A NOTES section at the very end is the only exception."""

REWRITE_TEMPLATE = """## Measured habits in this person's {register} writing

{constraints}

## Real samples of their {register} writing

{exemplars}

## Constructions to avoid

These mark text as machine-written. The draft below contains {n_hits}:

{lint_notes}

## The draft to rewrite

{draft}

Rewrite the draft in this person's voice. Same meaning, same facts, their rhythm."""

REPAIR_TEMPLATE = """Your rewrite is closer, but these measurements still sit \
outside the range this person writes in:

{deviations}

{lint_notes}
Here is your rewrite:

{candidate}

Fix the measurements listed above. Keep the meaning and every fact unchanged. \
Do not fix them by padding or deleting content -- change sentence structure, \
punctuation and word choice. Output the corrected text only."""


@dataclass
class Result:
    text: str
    score: float
    passes: int
    deviations: List[Deviation] = field(default_factory=list)
    lint_hits: List[L.Hit] = field(default_factory=list)
    history: List[str] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return L.hit_rate(self.text, self.lint_hits)

    @property
    def converged(self) -> bool:
        """Accept on the AI-tell rate, with the metrics as a secondary condition.

        The two signals are not equal. The tell rate separates his writing from
        a model's cleanly; the band score does not, so requiring every metric to
        sit inside its band would spend passes chasing a number that does not
        mean what it looks like it means.
        """
        return self.hit_rate <= L.ACCEPT_RATE and len(self.deviations) <= 2

    def report(self) -> str:
        lines = [
            "%s after %d pass%s"
            % ("accepted" if self.converged else "not accepted",
               self.passes, "" if self.passes == 1 else "es"),
            "ai-tells: %.2f per 100 words (accept at <= %.1f) -- %s"
            % (self.hit_rate, L.ACCEPT_RATE, L.summary(self.lint_hits)),
            "style-match %.1f/100 (steering signal, not a verdict)" % self.score,
        ]
        if self.deviations:
            lines.append("still off:")
            lines += ["  " + d.describe() for d in self.deviations]
        else:
            lines.append("all metrics inside your range")
        return "\n".join(lines)


def _format_exemplars(samples: Sequence[Sample]) -> str:
    out = []
    for i, s in enumerate(samples, 1):
        out.append("--- sample %d (%s) ---\n%s" % (i, s.date or "undated", s.text))
    return "\n\n".join(out)


def _format_lint(hits: Sequence[L.Hit], limit: int = 12) -> str:
    if not hits:
        return "(none found, but stay alert for them anyway)"
    return "\n".join("- " + h.describe() for h in hits[:limit])


def _strip_notes(text: str) -> str:
    """The model may append NOTES. Keep them out of the measured text, since a
    note about the draft is not part of the draft."""
    for marker in ("\nNOTES", "\n## NOTES", "\nNOTES:"):
        i = text.find(marker)
        if i != -1:
            return text[:i].rstrip()
    return text.strip()


def rewrite(
    draft: str,
    profile: Profile,
    retriever: Retriever,
    llm: LLM,
    k_exemplars: int = 5,
    max_passes: int = 3,
) -> Result:
    """Rewrite draft into the voice described by profile.

    Returns the best candidate seen, which is not always the last one -- a repair
    pass aimed at one metric can push another out, so the loop keeps the winner
    rather than trusting that later is better.
    """
    exemplars = retriever.top(draft, k=k_exemplars)
    hits = L.lint(draft)
    user = REWRITE_TEMPLATE.format(
        register=profile.register,
        constraints=profile.as_constraints(),
        exemplars=_format_exemplars(exemplars),
        n_hits="%d of them" % len(hits) if hits else "none that were detected",
        lint_notes=_format_lint(hits),
        draft=draft,
    )

    best: Optional[Result] = None
    history: List[str] = []

    for n in range(1, max_passes + 1):
        raw = llm.complete(SYSTEM, user)
        candidate = _strip_notes(raw)
        history.append(candidate)

        devs = profile.check(candidate)
        c_hits = L.lint(candidate)
        result = Result(
            text=candidate,
            score=profile.score(candidate),
            passes=n,
            deviations=devs,
            lint_hits=c_hits,
            history=list(history),
        )
        # Lint hits are weighted into the ranking so a metrically perfect rewrite
        # full of "leverage" cannot win.
        def rank(r: Result) -> float:
            # Tell rate dominates because it is the signal that discriminates;
            # the band score breaks ties between candidates that are both clean.
            return -100.0 * r.hit_rate + r.score

        if best is None or rank(result) > rank(best):
            best = result

        if result.converged or n == max_passes:
            break

        user = REPAIR_TEMPLATE.format(
            deviations="\n".join("- " + d.describe() for d in devs) or "- (none)",
            lint_notes=(
                "You also used these:\n%s\n" % _format_lint(c_hits) if c_hits else ""
            ),
            candidate=candidate,
        )

    assert best is not None
    best.passes = len(history)
    return best
