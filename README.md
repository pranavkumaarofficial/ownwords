# ownwords

Rewrites a draft into your own writing style, using a profile measured from your own past writing, and tells you whether the result actually matches.

It runs locally by default. Your writing samples stay on your machine.

## Why it exists

The usual way to make a model's draft sound like you is to paste a few of your messages into a chat window and ask for a rewrite. The output is better than the draft and still not yours, and nothing tells you which. Three things are missing from that approach.

**A measurement.** If you cannot score the output, you cannot tell whether a rewrite got better or only different. Here a deterministic function measures 24 style metrics on your corpus to build the profile, and the same function measures every rewrite. No model judges the output.

**Register.** One person writing a Slack message, a mail to a professor and a design doc is three different writers. A corpus mixed across those averages them into someone who does not exist. Every sample here carries a register (`chat`, `email`, `longform`, `notes`, `comment`) and they are never mixed.

**Privacy.** A style profile is built from private writing: sent mail, chat logs, notes. The rewrite prompt includes some of that writing as examples. That is a bad thing to send to a hosted API by default, so the default backend is a local model through Ollama.

Alternatives I looked at and did not use:

- **"Humanizer" tools.** They target AI detectors, not a particular person. Removing the tells gives you text with no voice at all, which reads as machine-written in its own way. This tool is not built to evade detectors and is not evaluated against them.
- **A LoRA fine-tune on the corpus.** It needs a GPU and has to be redone as the corpus grows. With only a handful of examples in the prompt, prompting was simpler to iterate on and to measure.
- **Embedding retrieval for the examples.** BM25 picked topically close samples well enough, and it needs no model and no network.

## Quickstart

Python 3.9+. No third-party packages.

```bash
git clone https://github.com/pranavkumaarofficial/ownwords
cd ownwords
pip install -e .

# measure a draft for AI tells; no corpus, no model needed
ownwords check examples/ai-draft.txt
```

To rewrite in your own style you need at least 15 samples of your writing, each 25+ words, for the register you want:

```bash
# sent mail pasted into one file, samples separated by lines of ---
ownwords ingest file my-emails.txt --register email
ownwords profile --register email          # writes the profile, prints what it measured

# rewrite with a local model (install Ollama first)
ollama pull qwen2.5:7b-instruct
ownwords rewrite draft.txt --register email
```

Other sources: `ownwords ingest mbox All-mail.mbox --address you@example.com` (Gmail Takeout; quoted replies and signatures are stripped), `ownwords ingest md ~/notes --register longform`, and `ownwords ingest claude conversations.json` (Claude data export, your own messages only). `ownwords status` shows what is stored.

The corpus and profiles live in `%APPDATA%\ownwords` on Windows and `~/.local/share/ownwords` elsewhere, never inside the checkout. Set `OWNWORDS_HOME` to move them.

`--backend anthropic` uses Claude instead (needs `ANTHROPIC_API_KEY`). It prints a warning, because the draft and up to five of your samples are sent with every call.

## How it works

```
ingest    your writing -> samples, tagged by register
profile   samples -> 10th-90th percentile band per metric, from your own spread
rewrite   draft + 5 nearest samples (BM25) + measured constraints -> model -> candidate
verify    candidate -> AI-tell lint + per-metric deviations
repair    if it misses: "your sentences average 24 words, you run 9-18" -> model again
```

Only the rewrite step calls a model. Everything that judges the output is deterministic and free, which is why the loop can afford three passes. It returns the best candidate it saw, not the last one, because a repair aimed at one metric can push another out of range.

## Design notes

**The style score does not work as a verdict, and the tool says so.** The first version scored rewrites 0 to 100 against the profile bands. Then I held out 80 of my own messages and scored them against AI-written passages. The AI passages landed inside my own score distribution at both band widths I tried. A tool that printed "92% match" would have been measuring nothing. The score is still used, because the per-metric deviations make the repair prompt specific, and specific numbers change the next draft where "be more natural" does not. But it is labelled as a steering signal everywhere it appears.

**What does separate the two is a small lint.** There are 13 rules for constructions models overuse (significance inflation, trailing "-ing" clauses, "not just X but Y", em dash asides, and so on). On my corpus:

| | my own writing (80 held out) | AI passages (4) | other human writing (2) |
| --- | --- | --- | --- |
| AI-tell hits per 100 words | mean 0.03, 76 of 80 at zero | 2.94 to 18.75 | 0.00, 0.00 |
| style band score, 0-100 | mean 89.2, 10th percentile 72.4 | 83.5 to 99.5 | not measured |

At a threshold of 1.0 hits per 100 words, the lint flags all 4 AI passages and 1 of my 80 messages. A rewrite is accepted on that threshold.

To reproduce on your own corpus: `python -m tools.evaluate` (needs 160+ chat samples of 40+ words). My corpus is not in this repo, so the exact numbers above are only reproducible by me. They came from 782 messages, 608 of them long enough to profile.

**Typos are measured, not copied.** A chat corpus is full of lowercase sentence starts and missing apostrophes. Those come from typing speed, not style. The pipeline measures them so it can tell the model to ignore them, and the output check skips them. What gets copied is sentence length and its spread, punctuation habits, hedging, and how directly a point gets made.

**What I would do differently:** build the evaluation before the scoring function. The band score took a full iteration before the eval showed it did not discriminate.

## Limitations

- **Nothing checks that the rewrite still says what the draft said.** The prompt forbids adding or dropping facts, but no code verifies it. Read the diff on anything that matters. This is the next thing to build.
- **The evaluation is small and partly circular.** The 4 AI passages and 2 human controls are hand-written, and the lint rules were written by the same person. "4 of 4 caught" says the rules catch what they were designed to catch. A real test needs drafts produced by several models from neutral prompts.
- **The lint is a fixed list.** It catches today's model vocabulary and will go stale. It cannot see a sentence that is bland rather than overwritten.
- **Only one register has been measured on real data**, my `chat` register. Email, longform, notes and comment work mechanically but have no real corpus behind them yet.
- **Local rewrite speed and quality have not been measured.** On a laptop CPU a 7B model is slow. There are no numbers here until there is a command that produces them.
- **The 15-sample floor is a judgement call**, not a measured threshold.

## Layout

```
ownwords/
  features.py   measure() -> 24 metrics; profile and check both use it
  profile.py    per-metric bands, constraints for the model
  lint.py       the 13 AI-tell rules, each with a span and a fix
  retrieve.py   BM25, about 20 lines, no dependency
  rewrite.py    rewrite -> measure -> repair loop
  llm.py        Ollama or Anthropic over urllib
  store.py      SQLite corpus; re-ingest is idempotent
  paths.py      where the corpus lives
  ingest/       Claude export, mbox, text files, markdown
tools/evaluate.py   the numbers above
tests/              41 tests, no network
```

```bash
python -m unittest discover -s tests
```

## License

MIT
