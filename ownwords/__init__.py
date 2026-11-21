"""ownwords -- rewrite text into one person's measured writing voice.

The pipeline:

    ingest   adapters -> Sample(text, register)        ownwords/ingest/
    profile  samples  -> per-metric bands + bigrams    ownwords/profile.py
    retrieve draft    -> 5 nearest real samples        ownwords/retrieve.py
    rewrite  draft    -> candidate                     ownwords/rewrite.py
    verify   candidate-> deviations, ai-tells          ownwords/features.py, ownwords/lint.py

Only the rewrite step calls a model. Everything that judges the output is
deterministic, local and free, which is what lets the loop run more than once.
"""
from .models import Sample, REGISTERS
from .profile import Profile
from .retrieve import Retriever
from .rewrite import rewrite, Result
from .store import Store
from .llm import LLM
from . import lint          # the module, not the function: `ownwords.lint.lint(text)`.
                            # Re-exporting the function here would shadow the
                            # submodule and break `from ownwords import lint as L`.

__version__ = "0.1.0"
__all__ = [
    "Sample", "REGISTERS", "Profile", "Retriever", "rewrite", "Result",
    "Store", "LLM", "lint",   # lint is the module
]
