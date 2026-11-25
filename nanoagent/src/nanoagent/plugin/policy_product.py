"""``sketch_bm25``, the product selector: Okapi BM25 over the shared chunks, every count read from a sketch.

Q = the distinct terms (``chunking.tokens``) of the task minus its "Rules not in scope" section; d_i = the term list
of chunk i (shared chunker), N chunks, avgdl = mean |d_i| (lengths are exact counts, not sketched).
  presence  one Bloom filter per chunk of its distinct terms, p = 0.01; with ``sketch_opts`` probe_sizing (VS1)
            p = min(0.01, 0.1 / (|Q| N)), so the expected number of false hits over all |Q| N probes is at most 0.1.
  df        a count-min sketch, 1024 x 5, of one increment per (chunk, distinct term); conservative update with
            conservative_update (VS2). df^ = min(N, query): df <= df^, and df^ <= df + (e / 1024) T with probability
            at least 1 - e^-5 (T = the sum of the chunks' distinct-term counts).
            idf(t) = ln(1 + (N - df^ + 0.5) / (df^ + 0.5)), so idf^ <= idf (idf falls as df rises).
  tf        one count-min sketch per chunk, width 8 |d_i| (8 counters per token), depth 4, of every token of d_i.
            tf <= tf^, and a row errs only if a collision adds >= 1, which by Markov has probability <= |d_i| / w
            = 1/8, so P[tf^ != tf] <= 8^-4 (about 2.4e-4) per query.
  score(i) = sum over t in Q reported by chunk i's filter, with tf^ > 0, of
            idf(t) tf^ (k1 + 1) / (tf^ + k1 (1 - b + b |d_i| / avgdl)),  k1 = 1.2, b = 0.75 (a heuristic: no guarantee).
            Guarantee: every query term a chunk holds is scored (no Bloom false negatives, tf^ >= tf >= 1);
            ``twin="exact"`` (bm25x) swaps each sketch for its exact twin, and the score is policy_select's BM25.
  dedup     ``"minhash"``: a chunk whose estimated Jaccard with a kept chunk (word 3-shingles, K = 128, LSH 32 x 4)
            is >= 0.8 is skipped; one_permutation (VS3) signs with densified one-permutation hashing.
VS1-VS3 are :mod:`sketches_variant`'s, the steps bloom_chunk takes too.
Budget B = floor(f x W), W = the handbook's word count (whitespace-split, before any strip), f = ``fraction``, or
``above``'s fraction g when ``above`` = (n, g) and W >= n; then capped at ``max_words`` when set (:func:`budget_words`).
The product's ``budget: frac:0.35,above:5000:0.60`` is B = floor(0.35 W) for W < 5,000 and floor(0.60 W) otherwise,
with no cap; the side rung ``frac:0.35,words:1500`` is the planned min(floor(0.35 W), 1,500). Chunks
scoring > 0 are taken best first (ties: handbook order) while the word count stays within 1.05 B, stopping once B
is reached, and shown as policy_select.bm25 shows them: handbook order after the title line, a later piece of a
section without its title line when the piece before it is shown. ``stopwords: light`` then strips 27 function
words from unguarded lines (:func:`strip_light`). With the same budget, dedup none and no strip, bm25x shows
exactly the text of policy_select.bm25_frac.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from nanoagent.plugin import chunking, policy_select
from nanoagent.plugin import sketches_variant as variant
from nanoagent.plugin.bloom_filter import lookup_key
from nanoagent.plugin.chunking import tokens
from nanoagent.plugin.policy_select import B_LEN, K1, SLACK, first_line, in_scope
from nanoagent.plugin.policy_vanilla import LSH_BANDS, MINHASH_K, NEAR_DUP, shingles
from nanoagent.plugin.sketches import BloomFilter, CountMinSketch, ExactCounter, ExactNearDuplicates, ExactSet, NearDuplicates

if TYPE_CHECKING:
    from nanoagent.plugin.bots import Components

DF_WIDTH, DF_DEPTH = 1024, 5
TF_PER_TOKEN, TF_DEPTH = 8, 4
P_TEXTBOOK = 0.01
OPTIONS = ("probe_sizing", "conservative_update", "one_permutation")  # VS1, VS2, VS3 (sketches_variant); N1 runs on bloom_chunk only
# The light list: hh_boilerplate_v2's Misra-Gries heavy hitters of the 9 handbooks that are also among general
# English's 100 most frequent words (efficient_corporate_travel_agent pilot/fleet/notes/stop_stack.md), fixed before any run.
LIGHT = frozenset("a an are as at be by for from have in is it of on other that the their this time to when will with you your".split())
HEADING_WORDS = 12
DIGIT = re.compile(r"\d")


def count_min(width: int, depth: int, salt: str, conservative: bool = False) -> CountMinSketch:
    """A count-min sketch given its width and depth (eps = e / w, delta = e^-d), with conservative update (VS2,
    :class:`sketches_variant.ConservativeCountMin`) when ``conservative``."""
    cms = (variant.ConservativeCountMin if conservative else CountMinSketch)(salt=salt)
    cms.width, cms.depth, cms.rows = width, depth, [[0] * width for _ in range(depth)]
    return cms


def units(handbook: str, chunker: str) -> list[tuple[str, int, int]]:
    """(chunk text, section index, piece index) for each chunk of ``chunking.CHUNKERS[chunker]``, in order."""
    if chunker == "plain":
        return [(c, s, 0) for s, c in enumerate(chunking.plain_chunks(handbook))]
    return [(piece, s, p) for s, section in enumerate(chunking._sections(handbook)) for p, piece in enumerate(chunking._pieces(section))]


def document_frequency(query: set[str], docs: list[list[str]], idf: str, exact: bool, opts: tuple[str, ...], seed: int) -> dict[str, int]:
    """df^(t) = min(N, the sketch's count) for t in the query, from one increment per (chunk, distinct term)."""
    df = ExactCounter() if exact or idf == "exact" else count_min(DF_WIDTH, DF_DEPTH, f"{seed}:df", "conservative_update" in opts)
    for d in docs:
        for t in sorted(set(d)):
            df.add(t)
    return {t: min(len(docs), df.query(t)) for t in query}


def term_frequency(query: set[str], doc: list[str], i: int, p: float, exact: bool, conservative: bool, seed: int) -> dict[str, int]:
    """tf^(t) for each query term chunk i's Bloom filter reports, read from its count-min sketch (0 when absent)."""
    present = ExactSet() if exact else BloomFilter(len(set(doc)), p, salt=f"{seed}:chunk{i}")
    tf = ExactCounter() if exact else count_min(max(TF_PER_TOKEN, TF_PER_TOKEN * len(doc)), TF_DEPTH, f"{seed}:tf{i}", conservative)
    for t in sorted(set(doc)):
        present.add(t)
    for t in doc:
        tf.add(t)
    return {t: tf.query(t) for t in sorted(query) if present.contains(t)}


def scores(query: set[str], docs: list[list[str]], idf: str, exact: bool, opts: tuple[str, ...], seed: int) -> list[float]:
    """score(i), the BM25 formula of the module docstring over the sketched df, presence and tf."""
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / max(1, n)
    df = document_frequency(query, docs, idf, exact, opts, seed)
    weight = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in query}
    p = variant.probe_p(len(query), n, P_TEXTBOOK) if "probe_sizing" in opts else P_TEXTBOOK
    out = []
    for i, d in enumerate(docs):
        norm = K1 * (1 - B_LEN + B_LEN * len(d) / max(1e-9, avgdl))
        tf = term_frequency(query, d, i, p, exact, "conservative_update" in opts, seed)
        out.append(math.fsum(weight[t] * f * (K1 + 1) / (f + norm) for t, f in tf.items() if f > 0))
    return out


def near_duplicates(exact: bool, opts: tuple[str, ...], seed: int) -> NearDuplicates | ExactNearDuplicates:
    """The dedup index: MinHash + LSH (VS3's one-permutation signatures with one_permutation), or its exact twin."""
    return ExactNearDuplicates() if exact else variant.kit(opts)[2](MINHASH_K, LSH_BANDS, salt=f"{seed}:mh")


def pick(texts: list[str], score: list[float], budget: int, near: NearDuplicates | ExactNearDuplicates | None) -> list[int]:
    """Indices kept: best first within 1.05 B until B is reached, skipping near-copies of kept chunks (``near``)."""
    total, kept = 0, []
    for i in sorted((i for i, s in enumerate(score) if s > 0), key=lambda i: (-score[i], i)):
        if total >= budget:
            break
        words = len(texts[i].split())
        sh = shingles(tokens(texts[i])) if near is not None else set()
        if total + words > SLACK * budget or (near is not None and near.similar(sh, NEAR_DUP)):
            continue
        if near is not None:
            near.add(i, sh)
        kept.append(i)
        total += words
    return sorted(kept)


def budget_words(words: int, fraction: float, max_words: int | None = None, above: tuple[int, float] | None = None) -> int:
    """B for a handbook of ``words`` words: floor(f x words), f = ``above``'s fraction when ``above`` = (n, g) and
    words >= n, else ``fraction``; at most ``max_words`` when it is set."""
    if above is not None and words >= above[0]:
        fraction = above[1]
    budget = int(fraction * words)
    return budget if max_words is None else min(budget, max_words)


def chosen(handbook: str, task_text: str, *, idf: str = "cms", dedup: str = "minhash", twin: str = "sketch", chunker: str = "structured",
           fraction: float = 0.35, max_words: int | None = None, above: tuple[int, float] | None = None, sketch_opts: tuple[str, ...] = (),
           seed: int = 580) -> tuple[list[tuple[str, int, int]], list[int]]:
    """(the units, the indices kept), within :func:`budget_words` (``fraction``, ``max_words``, ``above``)."""
    exact, parts = twin == "exact", units(handbook, chunker)
    docs = [tokens(text) for text, _, _ in parts]
    score = scores(set(tokens(in_scope(task_text))), docs, idf, exact, tuple(sketch_opts), seed)
    budget = budget_words(len(handbook.split()), fraction, max_words, above)
    near = near_duplicates(exact, tuple(sketch_opts), seed) if dedup == "minhash" else None
    return parts, pick([text for text, _, _ in parts], score, budget, near)


def select(handbook: str, task_text: str, **options) -> str:
    """The policy text shown: the title line, then the kept chunks as policy_select.bm25 shows them (``options``: see :func:`chosen`)."""
    parts, kept = chosen(handbook, task_text, **options)
    shown, out = set(kept), []
    for i in kept:
        text, section, piece = parts[i]
        continued = piece > 0 and i - 1 in shown and parts[i - 1][1] == section
        out.append(text.split("\n", 1)[1] if continued else text)
    if not out or not out[0].startswith(first_line(handbook)):
        out.insert(0, first_line(handbook))
    return "\n\n".join(out)


def strip_light(text: str) -> str:
    """``text`` without the words of :data:`LIGHT` (matched by ``bloom_filter.lookup_key``), except on guarded lines:
    a line with a digit (rates, amounts, day counts, rule numbers) or a heading or list label (<= 12 words, not ending
    in '.') is kept whole. Guarantee: no token with a digit, no negation and no logic word (not, no, must, only,
    unless, if, or, and, ...) is ever removed, since none is in the list; other lines' words are joined by one space."""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if not s or DIGIT.search(s) or (len(s.split()) <= HEADING_WORDS and not s.endswith(".")):
            out.append(line)
        else:
            out.append(" ".join(tok for tok in line.split() if lookup_key(tok) not in LIGHT))
    return "\n".join(out)


def stripped(inner: Callable[[str, str], str], handbook: str, task_text: str) -> str:
    """``inner``'s policy text after :func:`strip_light` (``stopwords: light`` on any sketch selector)."""
    return strip_light(inner(handbook, task_text))


def bm25_budget(handbook: str, task_text: str, fraction: float, max_words: int) -> str:
    """policy_select.bm25 with B = min(floor(fraction x handbook words), max_words) (``bm25`` + ``budget: frac:f,words:n``)."""
    return policy_select.bm25(handbook, task_text, budget=min(int(fraction * len(handbook.split())), max_words))


def check(c: Components) -> None:
    """Refuse a ``sketch_bm25`` configuration it cannot run (each component it reads must be set)."""
    unset = [name for name in ("idf", "dedup", "twin", "chunker", "budget", "seed", "stopwords") if getattr(c, name) is None]
    if unset:
        raise ValueError(f"selector 'sketch_bm25' needs {', '.join(unset)}: set harness.bot=product (or optimal), or set each")
    if c.idf == "none":
        raise ValueError("selector 'sketch_bm25' needs a document frequency: harness.idf cms or exact")
    if c.backup not in (None, "none") or any(o not in OPTIONS for o in c.sketch_opts or ()):
        raise NotImplementedError(f"harness.backup={c.backup!r}, sketch_opts={c.sketch_opts!r}: only {list(OPTIONS)} run on sketch_bm25 (N1 and VS4 run on bloom_chunk)")


def method(c: Components, fraction: float, max_words: int | None, above: tuple[int, float] | None = None) -> Callable[[str, str], str]:
    """The ``sketch_bm25`` selector :func:`nanoagent.plugin.policy.with_policy` runs for ``c``."""
    options = {"idf": c.idf, "dedup": c.dedup, "twin": c.twin, "chunker": c.chunker, "sketch_opts": tuple(c.sketch_opts or ())}
    assert c.seed is not None  # check
    return functools.partial(select, **options, fraction=fraction, max_words=max_words, above=above, seed=c.seed)
