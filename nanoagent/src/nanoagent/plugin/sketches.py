"""Textbook randomized sketches, each followed by its exact twin with the same interface (standard library only).

Every hash is ``hashlib.blake2b`` keyed by (salt, i): the i-th hash function is an independent seeded function,
and nothing depends on Python's ``hash()`` randomisation (PYTHONHASHSEED). ``harness.twin: exact`` swaps each
sketch for its twin. References: B. H. Bloom, CACM 1970; G. Cormode and S. Muthukrishnan, "An improved data
stream summary: the count-min sketch", J. Algorithms 2005; A. Broder 1997; P. Indyk and R. Motwani, STOC 1998;
Leskovec, Rajaraman and Ullman, Mining of Massive Datasets, ch. 3. The Bloom sizing and the MinHash + LSH banding
are ported from the author's COMP 480/580 Assignment 1 (Bloom filters) and Assignment 2 (MinHash and LSH).
"""

from __future__ import annotations

import hashlib
import math


def seeded_hash(item: str, salt: str, i: int) -> int:
    """The i-th member of a family of independent 64-bit hash functions selected by ``salt``."""
    key = f"{salt}:{i}".encode()[:64]
    return int.from_bytes(hashlib.blake2b(item.encode(), digest_size=8, key=key).digest(), "little")


class BloomFilter:
    """Set membership with no false negatives and a false-positive rate of about p.

    Sizing for n items at target rate p:  m = ceil(-n ln p / (ln 2)^2) bits,  k = round((m / n) ln 2) hashes,
    i.e. m / n = log2(1 / p) / ln 2 bits per item. Guarantee: every added item is reported; after n inserts
    a non-member is reported with probability (1 - e^(-kn/m))^k, which is about p.
    """

    def __init__(self, n: int, p: float = 0.01, salt: str = "bloom") -> None:
        n = max(1, n)
        self.m = math.ceil(-n * math.log(p) / math.log(2) ** 2)
        self.k = max(1, round(self.m / n * math.log(2)))
        self.salt = salt
        self.bits = bytearray((self.m + 7) // 8)

    def _positions(self, item: str) -> list[int]:
        return [seeded_hash(item, self.salt, i) % self.m for i in range(self.k)]

    def add(self, item: str) -> None:
        for pos in self._positions(item):
            self.bits[pos // 8] |= 1 << (pos % 8)

    def contains(self, item: str) -> bool:
        return all(self.bits[pos // 8] >> (pos % 8) & 1 for pos in self._positions(item))


class ExactSet:
    """Exact twin of :class:`BloomFilter`: a Python set (no false positives, no false negatives)."""

    def __init__(self, n: int = 0, p: float = 0.0, salt: str = "") -> None:
        self.items: set[str] = set()

    def add(self, item: str) -> None:
        self.items.add(item)

    def contains(self, item: str) -> bool:
        return item in self.items


class CountMinSketch:
    """Frequency estimates that never under-count.

    Width w = ceil(e / eps), depth d = ceil(ln(1 / delta)) (eps = delta = 0.01 gives 272 x 5); row j adds the
    count at column h_j(x) mod w and query(x) is the minimum over the rows. Guarantee: f(x) <= query(x) always,
    and query(x) <= f(x) + eps * T with probability at least 1 - delta, where T is the total of all counts.
    """

    def __init__(self, eps: float = 0.01, delta: float = 0.01, salt: str = "cms") -> None:
        self.width = math.ceil(math.e / eps)
        self.depth = math.ceil(math.log(1 / delta))
        self.salt = salt
        self.rows = [[0] * self.width for _ in range(self.depth)]
        self.total = 0

    def add(self, item: str, count: int = 1) -> None:
        self.total += count
        for j, row in enumerate(self.rows):
            row[seeded_hash(item, self.salt, j) % self.width] += count

    def query(self, item: str) -> int:
        return min(row[seeded_hash(item, self.salt, j) % self.width] for j, row in enumerate(self.rows))


class ExactCounter:
    """Exact twin of :class:`CountMinSketch`: a dict of counts (query(x) = f(x))."""

    def __init__(self, eps: float = 0.0, delta: float = 0.0, salt: str = "") -> None:
        self.counts: dict[str, int] = {}
        self.total = 0

    def add(self, item: str, count: int = 1) -> None:
        self.total += count
        self.counts[item] = self.counts.get(item, 0) + count

    def query(self, item: str) -> int:
        return self.counts.get(item, 0)


def jaccard(a: set, b: set) -> float:
    """Exact Jaccard similarity J(A, B) = |A n B| / |A u B| (0 when both sets are empty)."""
    return len(a & b) / len(a | b) if a or b else 0.0


class MinHash:
    """Jaccard estimates from K independent min-hashes.

    sig(A)_i = min over x in A of h_i(x). P[sig(A)_i = sig(B)_i] = J(A, B), so the fraction of equal positions
    is an unbiased estimate of J with variance J(1 - J) / K (standard error at most 1 / (2 sqrt K)).
    """

    EMPTY = 2**64  # larger than any 64-bit hash, so an empty set matches nothing

    def __init__(self, K: int = 128, salt: str = "minhash") -> None:
        self.K, self.salt = K, salt

    def signature(self, items: set[str]) -> tuple[int, ...]:
        if not items:
            return (self.EMPTY,) * self.K
        return tuple(min(seeded_hash(x, self.salt, i) for x in items) for i in range(self.K))

    @staticmethod
    def estimate(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
        return sum(x == y != MinHash.EMPTY for x, y in zip(sig_a, sig_b)) / len(sig_a)


class LSH:
    """Banding: a K = b * r signature is cut into b bands of r rows; two keys are candidates when some band
    is identical. A pair with Jaccard J is a candidate with probability 1 - (1 - J^r)^b, an S-curve with its
    threshold near (1 / b)^(1 / r) (32 x 4: about 0.42; a pair at J = 0.8 is found with probability > 0.999)."""

    def __init__(self, bands: int = 32, rows: int = 4) -> None:
        self.bands, self.rows = bands, rows
        self.tables: list[dict[tuple[int, ...], list]] = [{} for _ in range(bands)]

    def _bands(self, sig: tuple[int, ...]) -> list[tuple[int, ...]]:
        assert len(sig) == self.bands * self.rows, "the signature length must be bands * rows"
        return [sig[j * self.rows : (j + 1) * self.rows] for j in range(self.bands)]

    def add(self, key, sig: tuple[int, ...]) -> None:
        for table, band in zip(self.tables, self._bands(sig)):
            table.setdefault(band, []).append(key)

    def candidates(self, sig: tuple[int, ...]) -> set:
        return {key for table, band in zip(self.tables, self._bands(sig)) for key in table.get(band, [])}


class NearDuplicates:
    """MinHash + LSH near-duplicate finder: LSH proposes candidates, the MinHash estimate confirms (>= threshold).
    Guarantee: the two error sources above; a stored set is missed only if LSH misses it or the estimate is low."""

    def __init__(self, K: int = 128, bands: int = 32, salt: str = "minhash") -> None:
        self.minhash, self.lsh, self.sigs = MinHash(K, salt), LSH(bands, K // bands), {}

    def add(self, key, items: set[str]) -> None:
        self.sigs[key] = self.minhash.signature(items)
        self.lsh.add(key, self.sigs[key])

    def similar(self, items: set[str], threshold: float) -> list:
        sig = self.minhash.signature(items)
        return [k for k in sorted(self.lsh.candidates(sig)) if MinHash.estimate(sig, self.sigs[k]) >= threshold]


class ExactNearDuplicates:
    """Exact twin of :class:`NearDuplicates`: every stored set is a candidate and similarity is exact Jaccard."""

    def __init__(self) -> None:
        self.sets: dict = {}

    def add(self, key, items: set[str]) -> None:
        self.sets[key] = items

    def similar(self, items: set[str], threshold: float) -> list:
        return [k for k, s in self.sets.items() if jaccard(items, s) >= threshold]
