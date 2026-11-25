"""``bloom_chunk``, the vanilla randomized selector: keep the handbook chunks that best match the task.

Q = the distinct terms (``chunking.tokens``) of the task minus its "Rules not in scope" section; chunks come
from the shared chunker; ``twin="exact"`` swaps every sketch for its exact twin, which bounds each sketch error.
  S1  one Bloom filter (p = 0.01) per chunk of its distinct terms: no false negatives, false positives about p.
  S2  ``idf="none"``: weight(t) = 1 (a hit count, rung V-a). ``cms``: a count-min sketch (272 x 5) of document
      frequency, one increment per (chunk, distinct term); weight(t) = ln(N / min(N, df^)), and 0 when df^ = 0
      (count-min never under-counts, so df^ = 0 proves t is in no chunk). ``exact``: df counted exactly.
  S4  score(i) = sum of weight(t) over the t in Q that chunk i's filter reports (a heuristic: no guarantee).
  S3  ``dedup="minhash"``: a chunk whose estimated Jaccard with a kept chunk (word 3-shingles, K = 128,
      LSH 32 x 4) is >= 0.8 is skipped. Budget B = floor(fraction x handbook words), capped at ``max_words``.
Chunks scoring > 0 are taken best first (ties: handbook order) while the word count stays within 1.05 B,
stopping once B is reached; they are shown in handbook order after the handbook's title line.
``sketch_opts`` and ``backup="learned"`` (the variant steps VS1-VS4, N1) are in :mod:`sketches_variant`.
"""

from __future__ import annotations

import math

from nanoagent.plugin import sketches_variant as variant
from nanoagent.plugin.chunking import CHUNKERS, tokens
from nanoagent.plugin.policy_select import first_line, in_scope
from nanoagent.plugin.sketches import BloomFilter, CountMinSketch, ExactCounter, ExactNearDuplicates, ExactSet, NearDuplicates

P_BLOOM, SLACK = 0.01, 1.05
CMS_EPS, CMS_DELTA = 0.01, 0.01  # width ceil(e / 0.01) = 272, depth ceil(ln 100) = 5
MINHASH_K, LSH_BANDS, NEAR_DUP, SHINGLE = 128, 32, 0.8, 3


def shingles(terms: list[str], k: int = SHINGLE) -> set[str]:
    """The word k-shingles of ``terms`` (one shingle when there are fewer than k terms)."""
    return {" ".join(terms[i : i + k]) for i in range(max(1, len(terms) - k + 1))}


def weights(query: set[str], chunk_terms: list[set[str]], idf: str, exact: bool, seed: int, cms: type = CountMinSketch) -> dict[str, float]:
    """weight(t) for t in the query: 1 for ``idf="none"``, else ln(N / min(N, df^)) and 0 when df^ = 0, where
    df^ comes from one increment per (chunk, distinct term) pair."""
    if idf == "none":
        return {t: 1.0 for t in query}
    df = ExactCounter() if exact or idf == "exact" else cms(CMS_EPS, CMS_DELTA, salt=f"{seed}:df")
    for t in (t for terms in chunk_terms for t in sorted(terms)):  # sorted: conservative update is order-sensitive
        df.add(t)
    n = len(chunk_terms)
    return {t: math.log(n / min(n, df.query(t))) if df.query(t) else 0.0 for t in query}


def scores(query: set[str], chunk_terms: list[set[str]], weight: dict[str, float], exact: bool, seed: int, p: float,
           bloom: type = BloomFilter) -> list[float]:
    """score(i): the summed weights of the query terms chunk i's Bloom filter (salted by i) reports."""
    out = []
    for i, terms in enumerate(chunk_terms):
        f = ExactSet() if exact else bloom(len(terms), p, salt=f"{seed}:chunk{i}")
        for t in terms:
            f.add(t)
        out.append(math.fsum(weight[t] for t in sorted(query) if f.contains(t)))
    return out


def pick(chunks: list[str], score: list[float], budget: int, dedup: str, exact: bool, seed: int, near: type = NearDuplicates) -> list[int]:
    """Indices kept: best first within 1.05 B until B is reached, skipping near-copies of kept chunks."""
    found = ExactNearDuplicates() if exact else near(MINHASH_K, LSH_BANDS, salt=f"{seed}:mh")
    total, kept = 0, []
    for i in sorted((i for i, s in enumerate(score) if s > 0), key=lambda i: (-score[i], i)):
        if total >= budget:
            break
        words, sh = len(chunks[i].split()), shingles(tokens(chunks[i]))
        if total + words > SLACK * budget or (dedup == "minhash" and found.similar(sh, NEAR_DUP)):
            continue
        found.add(i, sh)
        kept.append(i)
        total += words
    return sorted(kept)


def chosen(handbook: str, task_text: str, *, idf: str = "cms", dedup: str = "minhash", twin: str = "sketch", chunker: str = "structured",
           fraction: float = 0.35, max_words: int | None = None, seed: int = 580, p: float = P_BLOOM,
           sketch_opts: tuple[str, ...] = (), backup: str = "none") -> tuple[list[str], list[int]]:
    """(the chunks, the indices kept). ``backup="learned"`` (VS4) cuts the scorer at SCORER_FRACTION, then admits."""
    exact, chunks = twin == "exact", CHUNKERS[chunker](handbook)
    chunk_terms = [set(tokens(c)) for c in chunks]
    query = set(tokens(in_scope(task_text)))
    bloom, cms, near = variant.kit(sketch_opts)
    p = variant.probe_p(len(query), len(chunks), p) if "probe_sizing" in sketch_opts else p
    score = scores(query, chunk_terms, weights(query, chunk_terms, idf, exact, seed, cms), exact, seed, p, bloom)
    budget = int((variant.SCORER_FRACTION if backup == "learned" else fraction) * len(handbook.split()))
    kept = pick(chunks, score, budget if max_words is None else min(budget, max_words), dedup, exact, seed, near)
    return chunks, variant.backup_admit(chunks, chunk_terms, query, kept, exact, seed) if backup == "learned" else kept


def select(handbook: str, task_text: str, **options) -> str:
    """The policy text shown: the title line, then the kept chunks in handbook order (``options``: see :func:`chosen`)."""
    chunks, kept = chosen(handbook, task_text, **options)
    parts = [chunks[i] for i in kept]
    if not parts or not parts[0].startswith(first_line(handbook)):
        parts.insert(0, first_line(handbook))
    return "\n\n".join(parts)
