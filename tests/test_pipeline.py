"""Tests for the parts that must not drift.

The deterministic half of this pipeline is the half worth testing: if measure()
changes meaning, every stored profile silently becomes wrong, and the rewrite
loop starts optimising towards a target nobody chose.

Run: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ownwords import features as F
from ownwords import lint as L
from ownwords.models import Sample
from ownwords.profile import Band, Profile
from ownwords.retrieve import Retriever
from ownwords.store import Store


def mk_samples(n=30, register="chat"):
    """Samples with deliberate, consistent style: short sentences, heavy first
    person, contractions, no semicolons."""
    out = []
    for i in range(n):
        out.append(Sample(
            text=(
                "i want the parser to stop guessing. it keeps picking the wrong "
                "branch when the input is empty, and i don't think a bigger "
                "lookahead fixes it. the real problem is that run %d of the test "
                "suite never covered the empty case at all. can you add a test "
                "first and then we look at the grammar." % i
            ),
            register=register, source="test",
        ))
    return out


class TestFeatures(unittest.TestCase):
    def test_empty_text_returns_zeros_not_crash(self):
        v = F.measure("")
        self.assertEqual(set(v), set(F.METRIC_KEYS))
        self.assertTrue(all(x == 0.0 for x in v.values()))

    def test_rates_are_length_invariant(self):
        """A rate per 100 words must not change when the text is duplicated,
        otherwise long drafts would always look different from short samples."""
        one = "I think this is probably fine, but I really can't tell yet."
        v1 = F.measure(one)
        v2 = F.measure(" ".join([one] * 8))
        for k in ("comma_rate", "contraction_rate", "first_person_rate",
                  "hedge_rate", "intensifier_rate"):
            self.assertAlmostEqual(v1[k], v2[k], delta=0.6, msg=k)

    def test_contraction_counted_without_apostrophe(self):
        """Missing apostrophes are a typing artefact; the style is the contraction."""
        with_ap = F.measure("I don't think it isn't working, you're wrong.")
        without = F.measure("I dont think it isnt working, youre wrong.")
        self.assertAlmostEqual(
            with_ap["contraction_rate"], without["contraction_rate"], delta=0.01
        )
        self.assertGreater(without["contraction_rate"], 0)

    def test_ordinary_words_are_not_contractions(self):
        """'its', 'were', 'lets' and 'ill' are words, not contractions."""
        v = F.measure("The were ill lets its own thing were fine and its ill.")
        self.assertEqual(v["contraction_rate"], 0.0)

    def test_lowercase_start_detection_ignores_bullets(self):
        v = F.measure("- Add the test.\n- Fix the grammar.")
        self.assertEqual(v["lower_start_rate"], 0.0)
        v2 = F.measure("add the test. fix the grammar.")
        self.assertEqual(v2["lower_start_rate"], 100.0)

    def test_sentence_length_variation_is_measured(self):
        flat = F.measure(". ".join(["one two three four five"] * 8) + ".")
        varied = F.measure(
            "Short. This one runs considerably longer than the previous sentence "
            "did, by a wide margin and with several clauses hanging off it. Short "
            "again. Then another long one that keeps going for a while yet."
        )
        self.assertLess(flat["sent_len_sd"], varied["sent_len_sd"])

    def test_function_ngrams_mask_content(self):
        a = F.function_ngrams("i want the parser to stop guessing")
        b = F.function_ngrams("i want the kernel to stop panicking")
        self.assertEqual(a, b)  # same skeleton, different topic

    def test_mattr_insensitive_to_length(self):
        base = ("alpha beta gamma delta epsilon zeta eta theta iota kappa "
                "lambda mu nu xi omicron pi rho sigma tau upsilon ") * 10
        short = F.measure(base)["mattr"]
        long_ = F.measure(base * 4)["mattr"]
        self.assertAlmostEqual(short, long_, delta=0.05)


class TestLint(unittest.TestCase):
    def test_catches_the_obvious_tells(self):
        bad = (
            "This serves as a testament to our robust approach. It's not just "
            "fast, it's seamless. In order to delve into the landscape, we "
            "leverage a myriad of tools, highlighting the intricate interplay. "
            "I hope this helps! \U0001F680"
        )
        rules = {h.rule for h in L.lint(bad)}
        for expected in ("significance-inflation", "promo-adjective", "ai-vocab",
                         "negative-parallelism", "ing-tail", "filler-opener",
                         "chatbot-artifact", "emoji"):
            self.assertIn(expected, rules)

    def test_clean_human_text_is_quiet(self):
        """The lint must not fire on ordinary writing, or it is useless."""
        good = (
            "The parser picks the wrong branch on empty input. I added a test for "
            "the empty case before touching the grammar, because I wasn't sure the "
            "grammar was the problem. It was: the rule had no epsilon production."
        )
        self.assertEqual(L.lint(good), [], L.summary(L.lint(good)))

    def test_summary_of_nothing(self):
        self.assertEqual(L.summary([]), "clean")


class TestBand(unittest.TestCase):
    def test_inside_band_is_zero_miss(self):
        b = Band(p10=2.0, p50=4.0, p90=6.0, floor=0.1)
        self.assertEqual(b.miss(4.0), 0.0)
        self.assertEqual(b.miss(2.0), 0.0)
        self.assertEqual(b.miss(6.0), 0.0)

    def test_degenerate_band_does_not_divide_by_zero(self):
        """The author never uses semicolons: p10 == p90 == 0. One semicolon must
        be a large finite miss, not an infinite one."""
        b = Band(p10=0.0, p50=0.0, p90=0.0, floor=1e-3)
        m = b.miss(1.0)
        self.assertGreater(m, 1.0)
        self.assertLess(m, 1e6)

    def test_miss_direction(self):
        b = Band(p10=10.0, p50=14.0, p90=18.0, floor=1.0)
        self.assertGreater(b.miss(30.0), 0)
        self.assertGreater(b.miss(1.0), 0)


class TestProfile(unittest.TestCase):
    def test_refuses_to_build_from_too_few_samples(self):
        with self.assertRaises(ValueError):
            Profile.build(mk_samples(5), "chat")

    def test_build_and_roundtrip(self):
        p = Profile.build(mk_samples(40), "chat")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chat.json"
            p.save(path)
            q = Profile.load(path)
        self.assertEqual(p.register, q.register)
        self.assertEqual(p.n_samples, q.n_samples)
        self.assertEqual(
            {k: vars(v) for k, v in p.bands.items()},
            {k: vars(v) for k, v in q.bands.items()},
        )
        self.assertEqual(p.overall, q.overall)

    def test_own_corpus_scores_high_against_itself(self):
        """Sanity check on the whole scoring idea: text drawn from the corpus
        must score well, or the metric is measuring the wrong thing."""
        samples = mk_samples(40)
        p = Profile.build(samples, "chat")
        self.assertGreater(p.score(samples[0].text), 85.0)

    def test_ai_prose_scores_lower_than_own_prose(self):
        samples = mk_samples(40)
        p = Profile.build(samples, "chat")
        ai = (
            "In today's rapidly evolving technological landscape, the parser "
            "represents a pivotal component of the broader system architecture, "
            "underscoring the intricate interplay between lexical analysis and "
            "syntactic interpretation; moreover, its robust design facilitates "
            "comprehensive coverage across a myriad of edge cases, thereby "
            "ensuring seamless operation throughout the entirety of the pipeline."
        )
        self.assertLess(p.score(ai), p.score(samples[0].text))
        self.assertTrue(p.check(ai))

    def test_surface_metric_excluded_by_default(self):
        """Corpus is all-lowercase; correctly capitalised text must not be
        flagged for it unless surface checking is asked for."""
        p = Profile.build(mk_samples(40), "chat")
        capitalised = (
            "I want the parser to stop guessing. It keeps picking the wrong "
            "branch when the input is empty, and I don't think a bigger "
            "lookahead fixes it. Can you add a test first."
        )
        self.assertNotIn(
            "lower_start_rate", {d.metric for d in p.check(capitalised)}
        )
        self.assertIn(
            "lower_start_rate",
            {d.metric for d in p.check(capitalised, surface=True)},
        )

    def test_constraints_tell_the_model_not_to_copy_typos(self):
        c = Profile.build(mk_samples(40), "chat").as_constraints()
        self.assertIn("not style", c)

    def test_sparse_habit_is_not_reported_as_never(self):
        """One sample in ten hedges -> median is 0, corpus rate is not. The
        constraint must come from the corpus rate."""
        plain = ("the parser picks the wrong branch on empty input and the test "
                 "suite never covered that case at all, so i added one for it "
                 "first and then went back to look at the grammar rule itself")
        hedged = ("i think the parser is probably picking the wrong branch, maybe "
                  "because the empty case was never covered at all, but i am not "
                  "sure that the grammar rule is really what needs changing here")
        samples = (
            [Sample(text=plain + " run %d" % i, register="chat", source="p%d" % i)
             for i in range(27)]
            + [Sample(text=hedged + " run %d" % i, register="chat", source="h%d" % i)
               for i in range(3)]
        )
        p = Profile.build(samples, "chat")
        self.assertEqual(p.bands["hedge_rate"].p50, 0.0)
        self.assertGreater(p.overall["hedge_rate"], 0.0)
        self.assertNotIn("Hedging (maybe, probably, i think, basically): never",
                         p.as_constraints())


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.samples = [
            Sample(text="the postgres migration keeps failing on the index rebuild "
                        "because the lock is held by the analytics job overnight",
                   register="chat", source="a"),
            Sample(text="the parser picks the wrong branch when the grammar has no "
                        "epsilon production and the input happens to be empty",
                   register="chat", source="b"),
            Sample(text="can you check whether the kubernetes readiness probe is "
                        "actually hitting the health endpoint or just the port",
                   register="chat", source="c"),
        ] * 4

    def test_retrieves_topically_nearest(self):
        r = Retriever(self.samples, min_words=10)
        top = r.top("the grammar and the parser disagree on empty input", k=1)
        self.assertIn("parser", top[0].text)

    def test_zero_overlap_still_returns_samples(self):
        r = Retriever(self.samples, min_words=10)
        self.assertTrue(r.top("zzzz qqqq vvvv", k=3))

    def test_respects_word_budget(self):
        r = Retriever(self.samples, min_words=10)
        picked = r.top("parser grammar", k=5, max_words=30)
        self.assertLessEqual(sum(s.words for s in picked), 30)

    def test_empty_corpus_raises(self):
        with self.assertRaises(ValueError):
            Retriever([], min_words=10)


class TestStore(unittest.TestCase):
    def db_path(self) -> Path:
        """A scratch db. shutil.rmtree with ignore_errors because Windows can
        still hold the sqlite file for a moment after close(), and a failed
        tempdir teardown should not be reported as a failing test."""
        import shutil
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        return Path(d) / "c.db"

    def test_reingest_is_idempotent(self):
        """Re-running ingest on the same export must not duplicate the corpus,
        because duplicates silently reweight the profile."""
        with Store(self.db_path()) as s:
            samples = mk_samples(20)
            self.assertEqual(s.add(samples), 20)
            self.assertEqual(s.add(samples), 0)
            self.assertEqual(s.count(), 20)

    def test_registers_are_kept_apart(self):
        with Store(self.db_path()) as s:
            s.add(mk_samples(20, "chat"))
            s.add(mk_samples(20, "email"))
            self.assertEqual(s.count("chat"), 20)
            self.assertEqual(len(s.get(register="email")), 20)
            self.assertEqual(s.count("notes"), 0)

    def test_rejects_unknown_register(self):
        with self.assertRaises(ValueError):
            Sample(text="x", register="tweet", source="t")


class TestRewriteLoop(unittest.TestCase):
    """The loop itself, with a scripted model so no network is involved."""

    class FakeLLM:
        def __init__(self, replies):
            self.replies = list(replies)
            self.calls = []

        def complete(self, system, user):
            self.calls.append(user)
            return self.replies.pop(0) if self.replies else "fallback text here now"

    def setUp(self):
        from ownwords.rewrite import rewrite
        self.rewrite = rewrite
        self.samples = mk_samples(40)
        self.profile = Profile.build(self.samples, "chat")
        self.retriever = Retriever(self.samples, min_words=10)

    def test_stops_early_when_converged(self):
        good = self.samples[0].text
        llm = self.FakeLLM([good, "should not be reached"])
        r = self.rewrite("Leverage robust synergy.", self.profile,
                         self.retriever, llm, max_passes=3)
        self.assertEqual(r.passes, 1)
        self.assertEqual(len(llm.calls), 1)

    def test_repair_pass_receives_the_numbers(self):
        bad = ("The system serves as a robust testament to comprehensive "
               "architectural paradigms; moreover, it facilitates seamless "
               "operation throughout the entirety of the processing pipeline.")
        llm = self.FakeLLM([bad, self.samples[0].text])
        r = self.rewrite("x", self.profile, self.retriever, llm, max_passes=3)
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("you run", llm.calls[1])   # numeric feedback, not adjectives
        self.assertEqual(r.text, self.samples[0].text)

    def test_keeps_best_candidate_not_last(self):
        """A later pass can be worse. The loop must return the winner."""
        good = self.samples[0].text
        bad = ("Moreover, the comprehensive paradigm underscores a myriad of "
               "intricate, robust, seamless architectural considerations; "
               "furthermore it leverages holistic synergy throughout.")
        llm = self.FakeLLM([good + " ;", bad, bad])
        r = self.rewrite("x", self.profile, self.retriever, llm, max_passes=3)
        self.assertNotEqual(r.text, bad)

    def test_notes_section_is_stripped_from_measured_text(self):
        llm = self.FakeLLM([
            self.samples[0].text + "\n\nNOTES\n- the draft claims 40% faster "
            "with no baseline given."
        ])
        r = self.rewrite("x", self.profile, self.retriever, llm, max_passes=1)
        self.assertNotIn("NOTES", r.text)

    def test_respects_max_passes(self):
        bad = "Moreover, it leverages a myriad of robust, seamless paradigms."
        llm = self.FakeLLM([bad] * 5)
        r = self.rewrite("x", self.profile, self.retriever, llm, max_passes=2)
        self.assertEqual(r.passes, 2)
        self.assertFalse(r.converged)

    def test_system_prompt_agrees_with_profile_on_typos(self):
        """The profile tells the model not to copy typing artefacts. The system
        prompt once told it the opposite, and the model got both every call."""
        from ownwords.rewrite import SYSTEM
        self.assertNotIn("Those are the signal", SYSTEM)
        self.assertIn("Do not copy their typing errors", SYSTEM)


class TestPaths(unittest.TestCase):
    """The corpus must never default to a location inside the source tree."""

    def test_env_override(self):
        import os
        from ownwords import paths
        old = os.environ.get("OWNWORDS_HOME")
        os.environ["OWNWORDS_HOME"] = "/tmp/ow-test"
        try:
            self.assertEqual(paths.corpus_db(), Path("/tmp/ow-test") / "corpus.db")
        finally:
            if old is None:
                os.environ.pop("OWNWORDS_HOME")
            else:
                os.environ["OWNWORDS_HOME"] = old

    def test_default_is_outside_the_checkout(self):
        import os
        from ownwords import paths
        old = os.environ.pop("OWNWORDS_HOME", None)
        try:
            repo = Path(__file__).resolve().parent.parent
            self.assertNotIn(repo, paths.data_dir().resolve().parents)
        finally:
            if old is not None:
                os.environ["OWNWORDS_HOME"] = old

    def test_default_backend_is_local(self):
        from ownwords.llm import LLM
        from ownwords.cli import build_parser
        self.assertEqual(LLM().backend, "ollama")
        a = build_parser().parse_args(["rewrite", "x.txt"])
        self.assertEqual(a.backend, "ollama")


class TestLLMTransport(unittest.TestCase):
    """llm.py over real HTTP against a stub, because a JSON shape mistake in the
    transport is invisible until the first live call, and the first live call is
    the one you are paying for."""

    def setUp(self):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers["content-length"])
                received.append(json.loads(self.rfile.read(n)))
                out = json.dumps(
                    {"message": {"role": "assistant", "content": "rewritten text"}}
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        self.received = received
        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.host = "http://127.0.0.1:%d" % self.server.server_port
        t = threading.Thread(target=self.server.serve_forever, daemon=True)
        t.start()
        self.addCleanup(self.server.shutdown)

    def test_ollama_roundtrip(self):
        from ownwords.llm import LLM
        llm = LLM(backend="ollama", model="stub", ollama_host=self.host)
        self.assertEqual(llm.complete("sys", "user text"), "rewritten text")
        sent = self.received[0]
        self.assertEqual(sent["model"], "stub")
        self.assertFalse(sent["stream"])
        self.assertEqual(sent["messages"][0]["role"], "system")
        self.assertEqual(sent["messages"][1]["content"], "user text")

    def test_unreachable_host_raises_llmerror(self):
        from ownwords.llm import LLM, LLMError
        llm = LLM(backend="ollama", ollama_host="http://127.0.0.1:1", timeout=2)
        with self.assertRaises(LLMError):
            llm.complete("s", "u")

    def test_unknown_backend_raises(self):
        from ownwords.llm import LLM, LLMError
        with self.assertRaises(LLMError):
            LLM(backend="nope").complete("s", "u")

    def test_missing_api_key_raises_before_any_request(self):
        import os
        from ownwords.llm import LLM, LLMError
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with self.assertRaises(LLMError):
                LLM(backend="anthropic").complete("s", "u")
        finally:
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old

if __name__ == "__main__":
    unittest.main()
