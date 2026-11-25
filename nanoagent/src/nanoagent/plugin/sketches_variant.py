"""The variant steps over the textbook sketches of :mod:`nanoagent.plugin.sketches` (``harness.sketch_opts``).

Standard library only; each step is separable and keeps its parent's interface and exact twin. VS1 probe_sizing
is ported from the author's COMP 480/580 Assignment 1 (Efficient_LLM_With_Bloom_Filter, report.md section 3,
the resolution-limit rule eps <= 1 / m). VS2 conservative_update: C. Estan and G. Varghese, SIGCOMM 2002. VS3
one_permutation: P. Li, A. Owen and C.-H. Zhang, NIPS 2012; A. Shrivastava, ICML 2017 (optimal densification).
N1 double_hash: A. Kirsch and M. Mitzenmacher, ESA 2006 (a negative). VS4 is ``harness.backup: learned``.
"""

from __future__ import annotations

import itertools
import math
import re

from nanoagent.plugin.sketches import BloomFilter, CountMinSketch, ExactSet, MinHash, NearDuplicates, seeded_hash

PROBE_FALSE_HITS = 0.1  # VS1: expected false hits over the whole handbook
SCORER_FRACTION, BACKUP_P, KEY_HITS = 0.15, 0.01, 3  # VS4: scorer budget, backup filter rate, query hits of a key


def probe_p(query_terms: int, chunks: int, p: float = 0.01) -> float:
    """VS1: p = min(p, 0.1 / (|Q| N)). |Q| N probes at rate p give E[false hits] = |Q| N p <= 0.1, so with
    probability >= 0.9 no chunk gains a false hit (Markov); the cost is log2(1 / p) / ln 2 bits per item."""
    return min(p, PROBE_FALSE_HITS / max(1, query_terms * chunks))


class DoubleHashBloom(BloomFilter):
    """N1: g_i(x) = h1(x) + i h2(x) mod m, m prime, h2 in [1, m - 1], so two hashes give k distinct positions.
    Guarantee: no false negatives; the false-positive rate tends to (1 - e^(-kn/m))^k as m grows (Kirsch and
    Mitzenmacher 2006)."""

    def __init__(self, n: int, p: float = 0.01, salt: str = "bloom") -> None:
        super().__init__(n, p, salt)
        self.m = next(x for x in itertools.count(self.m) if x > 1 and all(x % d for d in range(2, math.isqrt(x) + 1)))
        self.bits = bytearray((self.m + 7) // 8)

    def _positions(self, item: str) -> list[int]:
        h1, h2 = seeded_hash(item, self.salt, 0), 1 + seeded_hash(item, self.salt, 1) % (self.m - 1)
        return [(h1 + i * h2) % self.m for i in range(self.k)]


class ConservativeCountMin(CountMinSketch):
    """VS2: add(x, c) sets each of x's counters to max(counter, min over x's counters + c). Guarantee: still
    f(x) <= query(x), and query(x) is never above the plain count-min's (same w, d, hashes), so the eps T bound
    holds with probability >= 1 - delta. Needs non-negative increments."""

    def add(self, item: str, count: int = 1) -> None:
        self.total += count
        cols = [seeded_hash(item, self.salt, j) % self.width for j in range(self.depth)]
        target = min(row[c] for row, c in zip(self.rows, cols)) + count
        for row, c in zip(self.rows, cols):
            row[c] = max(row[c], target)


class OnePermutationMinHash(MinHash):
    """VS3: one hash v(x); bin j holds min v(x) // K over x with v(x) mod K = j. An empty bin j copies bin
    b(j, t) for t = 0, 1, ... until a non-empty bin is found (b a seeded hash shared by every set). Guarantee
    (Shrivastava 2017): P[sig(A)_j = sig(B)_j] = J(A, B) for every j, so the estimate is unbiased, from one
    hash per item instead of K."""

    def signature(self, items: set[str]) -> tuple[int, ...]:
        if not items:
            return (self.EMPTY,) * self.K
        bins: list[int | None] = [None] * self.K
        for x in items:
            h = seeded_hash(x, self.salt, 0)
            j, v = h % self.K, h // self.K
            bins[j] = v if bins[j] is None else min(bins[j], v)
        return tuple(self._fill(bins, j) for j in range(self.K))

    def _fill(self, bins: list[int | None], j: int) -> int:
        t = 0
        while bins[j] is None:
            j, t = seeded_hash(f"{j}:{t}", self.salt + ":dense", 0) % self.K, t + 1
        return bins[j]


class OnePermutationNearDuplicates(NearDuplicates):
    """:class:`NearDuplicates` on VS3 signatures: the same LSH S-curve 1 - (1 - J^r)^b, since each bin matches
    with probability J."""

    def __init__(self, K: int = 128, bands: int = 32, salt: str = "minhash") -> None:
        super().__init__(K, bands, salt)
        self.minhash = OnePermutationMinHash(K, salt)


def kit(sketch_opts: tuple[str, ...] = ()) -> tuple[type, type, type]:
    """(Bloom, count-min, near-duplicate) classes for ``sketch_opts`` (VS1 changes p, not a class: :func:`probe_p`)."""
    return (
        DoubleHashBloom if "double_hash" in sketch_opts else BloomFilter,
        ConservativeCountMin if "conservative_update" in sketch_opts else CountMinSketch,
        OnePermutationNearDuplicates if "one_permutation" in sketch_opts else NearDuplicates,
    )


LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[a-z][.)])\s+")
NAME = re.compile(r"\b[A-Z][a-zA-Z&'-]+")
MONEY_OR_UNIT = re.compile(
    r"(?:\$|USD|€|£)\s?\d|\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent|miles?|km|days?|hours?|hrs?|minutes?|nights?|weeks?|months?|years?)\b", re.I
)


def is_key(chunk: str, terms: set[str], query: set[str]) -> bool:
    """VS4's label-free detector: >= KEY_HITS query terms and a name list (>= 3 list lines with >= 2 capitalised
    words each), an amount of money or a number with a unit."""
    names = sum(bool(LIST_LINE.match(line)) and len(NAME.findall(line)) >= 2 for line in chunk.splitlines())
    return len(terms & query) >= KEY_HITS and (names >= 3 or bool(MONEY_OR_UNIT.search(chunk)))


def backup_admit(chunks: list[str], chunk_terms: list[set[str]], query: set[str], kept: list[int], exact: bool, seed: int) -> list[int]:
    """VS4 (Kraska et al. 2018; Mitzenmacher 2018): the scorer's pick plus every chunk a backup Bloom filter
    (p = 0.01) over the key chunks the scorer missed reports. Guarantee: no key chunk is lost (no false
    negatives); each other chunk is admitted with probability about p."""
    missed = [c for i, c in enumerate(chunks) if i not in kept and is_key(c, chunk_terms[i], query)]
    backup = ExactSet() if exact else BloomFilter(len(missed), BACKUP_P, salt=f"{seed}:backup")
    for chunk in missed:
        backup.add(chunk)
    return sorted(set(kept) | {i for i, chunk in enumerate(chunks) if i not in kept and backup.contains(chunk)})
