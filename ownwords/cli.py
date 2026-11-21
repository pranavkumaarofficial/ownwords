"""Command line.

    ownwords ingest claude <conversations.json>
    ownwords ingest file <path> --register email
    ownwords ingest mbox <path.mbox> --address you@example.com
    ownwords status
    ownwords profile --register chat
    ownwords check <file>  --register chat     # measure without calling a model
    ownwords rewrite <file> --register email
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import ingest, paths
from .lint import ACCEPT_RATE, hit_rate, lint, summary
from .llm import LLM, LLMError
from .models import REGISTERS
from .profile import Profile
from .retrieve import Retriever
from .rewrite import rewrite as do_rewrite
from .store import Store

DB = paths.corpus_db()


def _profile_path(register: str) -> Path:
    return paths.profile_path(register)


def _read(arg: str) -> str:
    """A path, or - for stdin."""
    if arg == "-":
        return sys.stdin.read()
    return Path(arg).read_text(encoding="utf-8")


# --- commands ---------------------------------------------------------------
def cmd_ingest(a) -> int:
    store = Store(DB)
    if a.kind == "claude":
        samples = ingest.from_claude_export(Path(a.path), register=a.register)
    elif a.kind == "file":
        samples = ingest.from_textfiles(Path(a.path), register=a.register)
    elif a.kind == "md":
        samples = ingest.from_markdown_dir(Path(a.path), register=a.register)
    elif a.kind == "mbox":
        if not a.address:
            print("mbox ingest needs --address", file=sys.stderr)
            return 2
        samples = ingest.from_mbox(Path(a.path), a.address, register=a.register)
    else:
        print("unknown kind %r" % a.kind, file=sys.stderr)
        return 2

    added = store.add(samples)
    print("parsed %d samples, added %d new (%d were already stored)"
          % (len(samples), added, len(samples) - added))
    cmd_status(a)
    return 0


def cmd_status(a) -> int:
    store = Store(DB)
    regs = store.registers()
    if not regs:
        print("corpus is empty. start with: ownwords ingest file <your-writing.txt> --register email")
        return 0
    print("\ncorpus at %s" % DB)
    for r in REGISTERS:
        info = regs.get(r)
        prof = _profile_path(r)
        if not info:
            print("  %-9s --  no samples" % r)
            continue
        print("  %-9s %4d samples  %7s words  profile: %s"
              % (r, info["samples"], f"{info['words']:,}",
                 "built" if prof.exists() else "not built"))
    return 0


def cmd_profile(a) -> int:
    store = Store(DB)
    samples = store.get(register=a.register)
    if not samples:
        print("no samples in register %r" % a.register, file=sys.stderr)
        return 1
    try:
        prof = Profile.build(samples, a.register)
    except ValueError as e:
        print("cannot build profile: %s" % e, file=sys.stderr)
        return 1
    prof.save(_profile_path(a.register))
    md = paths.profile_path(a.register, "md")
    md.write_text(prof.as_markdown(), encoding="utf-8")
    print("profile for %r built from %d samples (%s words)"
          % (a.register, prof.n_samples, f"{prof.n_words:,}"))
    print("  %s\n  %s" % (_profile_path(a.register), md))
    print("\n%s" % prof.as_constraints())
    return 0


def cmd_check(a) -> int:
    """Measure text against a profile with no model call. Free, instant, and the
    fastest way to find out whether the profile itself is sane."""
    text = _read(a.path)
    hits = lint(text)
    rate = hit_rate(text, hits)
    print("ai-tells: %.2f per 100 words -- %s (%s at <= %.1f)"
          % (rate, summary(hits),
             "passes" if rate <= ACCEPT_RATE else "fails", ACCEPT_RATE))
    for h in hits[:20]:
        print("  " + h.describe())
    p = _profile_path(a.register)
    if not p.exists():
        print("\nno profile for %r yet; run: ownwords profile --register %s"
              % (a.register, a.register))
        return 0
    prof = Profile.load(p)
    devs = prof.check(text)
    # Labelled for what it is: the tell rate above is the acceptance gate, this
    # steers the repair pass. tools/evaluate.py shows why they differ.
    print("\nstyle-match %.1f/100 against %r (steering signal, not a verdict)"
          % (prof.score(text), a.register))
    if devs:
        print("outside your range:")
        for d in devs:
            print("  " + d.describe())
    else:
        print("every metric inside your range")
    return 0


def cmd_rewrite(a) -> int:
    p = _profile_path(a.register)
    if not p.exists():
        print("no profile for %r. run: ownwords profile --register %s"
              % (a.register, a.register), file=sys.stderr)
        return 1
    prof = Profile.load(p)
    store = Store(DB)
    retriever = Retriever(store.get(register=a.register))
    llm = LLM(backend=a.backend, model=a.model, temperature=a.temperature)
    if a.backend != "ollama":
        # The prompt carries up to five of your own samples as exemplars, so a
        # hosted backend sees pieces of your corpus on every rewrite.
        print("note: %s backend -- the draft and %d of your samples leave this machine"
              % (a.backend, a.exemplars), file=sys.stderr)

    try:
        result = do_rewrite(
            _read(a.path), prof, retriever, llm,
            k_exemplars=a.exemplars, max_passes=a.passes,
        )
    except LLMError as e:
        print("model call failed: %s" % e, file=sys.stderr)
        if a.backend == "ollama" and "cannot reach" in str(e):
            print("is Ollama running? start it, then: ollama pull %s" % llm.model,
                  file=sys.stderr)
        return 1

    if a.out:
        Path(a.out).write_text(result.text, encoding="utf-8")
        print(result.report(), file=sys.stderr)
        print("wrote %s" % a.out, file=sys.stderr)
    else:
        print(result.text)
        print("\n---\n%s" % result.report(), file=sys.stderr)
    return 0


# --- wiring -----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ownwords", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest", help="add writing samples to the corpus")
    i.add_argument("kind", choices=["claude", "file", "md", "mbox"])
    i.add_argument("path")
    i.add_argument("--register", default="chat", choices=REGISTERS)
    i.add_argument("--address", help="your email address, for mbox")
    i.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("status", help="what is in the corpus")
    s.set_defaults(fn=cmd_status)

    pr = sub.add_parser("profile", help="build a style profile for a register")
    pr.add_argument("--register", default="chat", choices=REGISTERS)
    pr.set_defaults(fn=cmd_profile)

    c = sub.add_parser("check", help="measure text, no model call")
    c.add_argument("path", help="file, or - for stdin")
    c.add_argument("--register", default="chat", choices=REGISTERS)
    c.set_defaults(fn=cmd_check)

    r = sub.add_parser("rewrite", help="rewrite text into your voice")
    r.add_argument("path", help="file, or - for stdin")
    r.add_argument("--register", default="chat", choices=REGISTERS)
    r.add_argument("--out", help="write here instead of stdout")
    r.add_argument("--backend", default="ollama", choices=["ollama", "anthropic"],
                   help="ollama (local, default) or anthropic (sends samples off-machine)")
    r.add_argument("--model", default=None)
    r.add_argument("--passes", type=int, default=3)
    r.add_argument("--exemplars", type=int, default=5)
    r.add_argument("--temperature", type=float, default=1.0)
    r.set_defaults(fn=cmd_rewrite)

    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
