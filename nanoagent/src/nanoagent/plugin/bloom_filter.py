"""Stage 1 of policy parsing: compress a travel-policy handbook with two Bloom filters.

A compression **level** is ``<relevance tier>+<word tier>``:

* the **relevance tier** keeps only the units (sentences or paragraphs) that contain a flight term,
  looked up in a Bloom filter of about 50 generic air-travel words;
* the **word tier** then drops every word that a second Bloom filter of low-meaning words matches.

``none+none`` is the uncompressed handbook. Every compression also runs the same steps with exact
``set``s, so the damage done by false positives alone can be reported for each filter.

Use from Python::

    import bloom_filter
    text, stats = bloom_filter.compress(handbook, "flight_paragraphs+curated")

This module knows nothing about LLMs.
"""

from __future__ import annotations

import hashlib
import math
import re
import string
import time
from dataclasses import dataclass
from functools import lru_cache

SEED = 580
HASH_KEY = SEED.to_bytes(8, "little")
WORD_TARGET_FPR = 0.01  # a word-filter false positive deletes one word everywhere it appears
RELEVANCE_TARGET_FPR = 0.001  # a relevance false positive keeps a whole unit, so it must be rarer

WORD_TIERS = ("none", "curated", "common1000")
RELEVANCE_TIERS = ("none", "flight_sentences", "flight_sentences_ctx", "flight_paragraphs")
DEFAULT_LEVEL = "flight_paragraphs+curated"
LEVELS = tuple(f"{r}+{w}" for r in RELEVANCE_TIERS for w in WORD_TIERS)
# Opt-in tiers outside the 12-level grid: the relevance filter also holds LODGING_TERMS, for tasks whose scope
# covers hotels as well as flights (e.g. Gates' lodging caps and non-reimbursable properties).
EXTRA_RELEVANCE_TIERS = ("travel_paragraphs",)

# --------------------------------------------------------------------------- #
# Vocabularies
# --------------------------------------------------------------------------- #

# NLTK's English stopword list (179 words), copied verbatim so no download is needed.
NLTK_STOPWORDS = frozenset("""
i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself yourselves
he him his himself she she's her hers herself it it's its itself they them their theirs themselves
what which who whom this that that'll these those am is are was were be been being have has had
having do does did doing a an the and but if or because as until while of at by for with about
against between into through during before after above below to from up down in out on off over
under again further then once here there when where why how all any both each few more most other
some such no nor not only own same so than too very s t can will just don don't should should've
now d ll m o re ve y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn
hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't shouldn
shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
""".split())

# Words never dropped, because deleting them can change a rule. Generic English, not tuned to any policy.
PROTECTED = frozenset("""
no nor not never without don don't ain aren aren't couldn couldn't didn didn't doesn doesn't hadn
hadn't hasn hasn't haven haven't isn isn't mightn mightn't mustn mustn't needn needn't shan shan't
shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
more most less least few only same very than all any each both some
over under above below before after during until again
s t d m o y ll re ve ma
zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen
sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety
hundred thousand million billion trillion half dozen
first second third fourth fifth sixth seventh eighth ninth tenth
january february march april may june july august september october november december
monday tuesday wednesday thursday friday saturday sunday
""".split())

# Markdown and list decoration. The table separator "|" is deliberately absent, so rows keep their columns.
DECORATIVE = frozenset(["#", "##", "###", "####", "*", "**", "-", "--", "---", ">", "•", "_"])

CURATED = NLTK_STOPWORDS - PROTECTED
COMMON_WORDS = 1000  # common1000: CURATED plus this many of wordfreq's most frequent English words

# Generic air-travel terms, plural forms written out. Deliberately left out because they are too common
# in any travel handbook and would keep almost every unit: "travel", "trip", "business", "first".
FLIGHT_TERMS = frozenset("""
air aircraft airfare airfares airline airlines airplane airplanes airport airports
flight flights fly flying flown flyer flyers
fare fares ticket tickets ticketed ticketing
economy coach cabin cabins class seat seats seating upgrade upgrades upgraded
baggage bag bags luggage layover layovers connecting nonstop non-stop red-eye
itinerary itineraries charter charters carrier carriers
""".split())

# Generic lodging terms, plural forms written out. "night" catches nightly rate caps ("per night").
LODGING_TERMS = frozenset("""
lodging hotel hotels motel motels accommodation accommodations room rooms resort resorts
property properties night nights nightly inn inns
""".split())


def word_vocabulary(tier: str) -> list[str]:
    """The sorted words the word filter drops for ``tier`` (empty for ``none``)."""
    if tier not in WORD_TIERS:
        raise ValueError(f"Unknown word tier {tier!r}; choose from {', '.join(WORD_TIERS)}")
    if tier == "none":
        return []
    words = set(CURATED | DECORATIVE)
    if tier == "common1000":
        from wordfreq import top_n_list  # loaded only when needed

        words.update(w for w in top_n_list("en", COMMON_WORDS)
                     if w.isascii() and w.isalpha() and w not in PROTECTED)
    return sorted(words)


def relevance_vocabulary(tier: str) -> list[str]:
    """The sorted flight terms the relevance filter holds for ``tier`` (empty for ``none``)."""
    if tier not in RELEVANCE_TIERS + EXTRA_RELEVANCE_TIERS:
        raise ValueError(f"Unknown relevance tier {tier!r}; choose from "
                         f"{', '.join(RELEVANCE_TIERS + EXTRA_RELEVANCE_TIERS)}")
    if tier == "none":
        return []
    return sorted(FLIGHT_TERMS | LODGING_TERMS) if tier.startswith("travel_") else sorted(FLIGHT_TERMS)


def parse_level(level: str) -> tuple[str, str]:
    """Split ``"<relevance>+<word>"`` into its two tiers, checking both."""
    relevance, sep, word = level.partition("+")
    if not sep or relevance not in RELEVANCE_TIERS + EXTRA_RELEVANCE_TIERS or word not in WORD_TIERS:
        raise ValueError(
            f"Unknown level {level!r}: expected <relevance>+<word> with relevance in "
            f"{', '.join(RELEVANCE_TIERS)} and word in {', '.join(WORD_TIERS)}"
        )
    return relevance, word


# --------------------------------------------------------------------------- #
# Bloom filter
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1 << 17)
def _hash_pair(word: str) -> tuple[int, int]:
    """Two 64-bit hashes of ``word`` from one keyed blake2b call; the second is forced odd.

    They don't depend on the filter's size or hash count, so they are cached and shared by every
    filter in a sweep.
    """
    digest = hashlib.blake2b(word.encode("utf-8"), digest_size=16, key=HASH_KEY).digest()
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:], "little") | 1


class BloomFilter:
    """``size`` bits (R) in a ``bytearray``, probed at ``hash_count`` (k) positions per word.

    Positions come from Kirsch–Mitzenmacher double hashing, ``(h1 + i * h2) mod R``, where ``h1``
    and ``h2`` come from one keyed blake2b digest. The bits are identical across processes and
    machines (Python's salted ``hash()`` is never used).
    """

    def __init__(self, size: int, hash_count: int):
        if size < 1 or hash_count < 1:
            raise ValueError(f"size and hash_count must be positive, got {size} and {hash_count}")
        self.size = size
        self.hash_count = hash_count
        self.bits = bytearray((size + 7) // 8)

    @classmethod
    def from_words(cls, words, size: int, hash_count: int) -> BloomFilter:
        bf = cls(size, hash_count)
        for word in words:
            bf.add(word)
        return bf

    def positions(self, word: str) -> list[int]:
        h1, h2 = _hash_pair(word)
        return [(h1 + i * h2) % self.size for i in range(self.hash_count)]

    def add(self, word: str) -> None:
        for p in self.positions(word):
            self.bits[p >> 3] |= 1 << (p & 7)

    def __contains__(self, word: str) -> bool:
        return all(self.bits[p >> 3] >> (p & 7) & 1 for p in self.positions(word))

    @property
    def memory_bits(self) -> int:
        return self.size

    @staticmethod
    def theoretical_fpr(n: int, size: int, hash_count: int) -> float:
        """The standard approximation ``(1 - e^{-kN/R})^k`` for N inserted words."""
        return (1.0 - math.exp(-hash_count * n / size)) ** hash_count

    @staticmethod
    def optimal_k(n: int, size: int) -> int:
        """``round((R/N) ln 2)``, at least 1."""
        return max(1, round(size / n * math.log(2)))


def smallest_size_for_fpr(n: int, target: float) -> int:
    """The smallest power-of-two R whose theoretical FPR at k_opt is at most ``target``."""
    size = 1
    while BloomFilter.theoretical_fpr(n, size, BloomFilter.optimal_k(n, size)) > target:
        size *= 2
    return size


def size_for_fpr(n: int, target: float) -> int:
    """R giving about ``target`` FPR at the optimal k: ``R = -N ln p / (ln 2)^2`` (for FPR sweeps)."""
    return math.ceil(-n * math.log(target) / math.log(2) ** 2)


def default_filter_params(n: int, target: float, size: int | None = None,
                          hash_count: int | None = None) -> tuple[int, int]:
    """Fill in a missing R (smallest power of two reaching ``target``) and k (k_opt for that R)."""
    if size is None:
        size = smallest_size_for_fpr(n, target)
    if hash_count is None:
        hash_count = BloomFilter.optimal_k(n, size)
    return size, hash_count


# --------------------------------------------------------------------------- #
# Text units
# --------------------------------------------------------------------------- #

_EDGE_PUNCTUATION = string.punctuation + "“”‘’—–…•"
_BULLET = re.compile(r"^\s*(?:[-*•◦▪●]|\d+[.)]|[a-zA-Z][.)])\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])")
_LIST_MARKER = re.compile(r"\(?(?:\d+|[a-zA-Z])[.)]")


def lookup_key(token: str) -> str:
    """The form of a whitespace token looked up in a filter.

    Lowercased, curly apostrophes made straight, and punctuation stripped from both ends, so "The",
    "the," and "(the" share the key "the". A token that is only punctuation ("##", "•") keeps its raw
    form, so decoration can still match.
    """
    key = token.lower().replace("’", "'").strip(_EDGE_PUNCTUATION)
    return key or token


def split_paragraphs(text: str) -> list[str]:
    """Blocks of text between blank lines, with surrounding whitespace removed."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_sentences(paragraph: str) -> list[str]:
    """A paragraph's sentences. Each bullet or list line starts a new item; other lines continue the
    previous one. Items are then split where ``.``, ``!`` or ``?`` is followed by whitespace and a
    capital, digit, quote or bracket."""
    items: list[str] = []
    for line in paragraph.splitlines():
        line = line.strip()
        if not line:
            continue
        if _BULLET.match(line) or not items:
            items.append(line)
        else:
            items[-1] += " " + line
    sentences: list[str] = []
    for item in items:
        pieces = [s.strip() for s in _SENTENCE_END.split(item) if s.strip()]
        for piece in pieces:
            if sentences and _LIST_MARKER.fullmatch(sentences[-1]):  # "1." is a list marker, not a sentence
                sentences[-1] += " " + piece
            else:
                sentences.append(piece)
    return sentences


# --------------------------------------------------------------------------- #
# Compression
# --------------------------------------------------------------------------- #


@dataclass
class _Filter:
    """One Bloom filter and the exact set of the same words, used to measure the filter's damage."""

    tier: str
    words: list[str]
    bf: BloomFilter
    exact: frozenset[str]

    @classmethod
    def build(cls, tier: str, words: list[str], target: float, size: int | None,
              hash_count: int | None) -> _Filter:
        size, hash_count = default_filter_params(len(words), target, size, hash_count)
        return cls(tier, words, BloomFilter.from_words(words, size, hash_count), frozenset(words))

    def base_stats(self, keys: set[str], memory: bool) -> dict:
        """Size, theoretical FPR, and the empirical FPR over the distinct non-member ``keys`` queried."""
        nonmembers = keys - self.exact
        false_positives = sum(key in self.bf for key in nonmembers)
        stats = {
            "tier": self.tier,
            "n_words": len(self.words),
            "size": self.bf.size,
            "hash_count": self.bf.hash_count,
            "memory_bits": self.bf.memory_bits,
            "theoretical_fpr": BloomFilter.theoretical_fpr(len(self.words), self.bf.size, self.bf.hash_count),
            "distinct_nonmembers": len(nonmembers),
            "false_positives": false_positives,
            "empirical_fpr": false_positives / len(nonmembers) if nonmembers else 0.0,
        }
        if memory:
            from pympler.asizeof import asizeof  # only needed for the memory comparison

            stats["bf_bytes"] = asizeof(self.bf)
            stats["set_bytes"] = asizeof(self.exact)
        return stats


class Compressor:
    """The two filters of one compression level, built once and reusable on many handbooks.

    Args:
        level: ``"<relevance tier>+<word tier>"``.
        relevance_size, relevance_hash_count: R and k of the relevance filter. Default: the smallest
            power of two reaching ``RELEVANCE_TARGET_FPR`` at k_opt.
        word_size, word_hash_count: R and k of the word filter. Default: the smallest power of two
            reaching ``WORD_TARGET_FPR`` at k_opt.
    """

    def __init__(self, level: str = DEFAULT_LEVEL, relevance_size: int | None = None,
                 relevance_hash_count: int | None = None, word_size: int | None = None,
                 word_hash_count: int | None = None):
        self.level = level
        self.relevance_tier, self.word_tier = parse_level(level)
        rel_words, word_words = relevance_vocabulary(self.relevance_tier), word_vocabulary(self.word_tier)
        self.relevance = (_Filter.build(self.relevance_tier, rel_words, RELEVANCE_TARGET_FPR,
                                        relevance_size, relevance_hash_count) if rel_words else None)
        self.word = (_Filter.build(self.word_tier, word_words, WORD_TARGET_FPR, word_size, word_hash_count)
                     if word_words else None)

    def config(self) -> dict:
        """The level and each filter's R and k (``None`` for a tier of ``none``)."""
        def params(f: _Filter | None) -> dict | None:
            return None if f is None else {"size": f.bf.size, "hash_count": f.bf.hash_count}

        return {"level": self.level, "relevance_filter": params(self.relevance), "word_filter": params(self.word)}

    def drops_word(self, token: str) -> bool:
        """Whether the word filter drops ``token`` (used to match quotes against the original text)."""
        return self.word is not None and lookup_key(token) in self.word.bf

    def compress(self, text: str, memory: bool = False) -> tuple[str, dict]:
        """Compress ``text``: relevance tier first, then the word tier on what is left.

        Args:
            text: The handbook.
            memory: Also measure each filter's and exact set's size in bytes (``pympler``; slower).

        Returns:
            ``(compressed_text, stats)``. ``stats`` has the level, ``relevance_filter`` and
            ``word_filter`` (each ``None`` for a tier of ``none``) with their FPR and damage, and the
            overall words, characters and time.
        """
        start = time.perf_counter()
        kept, relevance_stats = self._relevance_pass(text, memory)
        compressed, word_stats = self._word_pass(kept, memory)
        words_in, words_out = len(text.split()), len(compressed.split())
        return compressed, {
            "level": self.level,
            "relevance_filter": relevance_stats,
            "word_filter": word_stats,
            "words_in": words_in,
            "words_out": words_out,
            "retained_frac": words_out / words_in if words_in else 1.0,
            "chars_in": len(text),
            "chars_out": len(compressed),
            "compress_s": time.perf_counter() - start,
        }

    # -- relevance tier ------------------------------------------------------ #

    def _relevance_pass(self, text: str, memory: bool) -> tuple[str, dict | None]:
        if self.relevance is None:
            return text, None
        f = self.relevance
        unit = self.relevance_tier.split("_", 1)[1]  # flight_paragraphs / travel_paragraphs -> paragraphs
        paragraphs = split_paragraphs(text)
        if unit == "paragraphs":
            groups = [[p] for p in paragraphs]
        else:
            groups = [split_sentences(p) for p in paragraphs]

        def matches(unit_text: str, members) -> bool:
            return any(lookup_key(token) in members for token in unit_text.split())

        out_paragraphs, units_total, units_kept, fp_units, fp_words = [], 0, 0, 0, 0
        for units in groups:
            hit_bf = [matches(u, f.bf) for u in units]
            hit_exact = [matches(u, f.exact) for u in units]
            keep_bf, keep_exact = _with_context(hit_bf, unit), _with_context(hit_exact, unit)
            kept = [u for u, keep in zip(units, keep_bf) if keep]
            units_total += len(units)
            units_kept += len(kept)
            for u, by_bf, by_exact in zip(units, keep_bf, keep_exact):
                if by_bf and not by_exact:  # kept only because of a false positive
                    fp_units += 1
                    fp_words += len(u.split())
            if kept:
                out_paragraphs.append(kept[0] if unit == "paragraphs" else "\n".join(kept))

        keys = {lookup_key(token) for token in text.split()}
        stats = f.base_stats(keys, memory)
        stats.update({
            "unit": unit,
            "units_total": units_total,
            "units_kept": units_kept,
            "fp_units_kept": fp_units,
            "fp_words_added": fp_words,
        })
        return "\n\n".join(out_paragraphs), stats

    # -- word tier ------------------------------------------------------------ #

    def _word_pass(self, text: str, memory: bool) -> tuple[str, dict | None]:
        if self.word is None:
            return text, None
        f = self.word
        drop: dict[str, bool] = {}  # memo: key -> filter answer
        fp_tokens, fp_words = 0, set()
        lines, previous_blank = [], False
        for line in text.split("\n"):
            kept = []
            for token in line.split():
                key = lookup_key(token)
                if key not in drop:
                    drop[key] = key in f.bf
                if not drop[key]:
                    kept.append(token)
                elif key not in f.exact:  # dropped only because of a false positive
                    fp_tokens += 1
                    fp_words.add(key)
            new_line = " ".join(kept)
            if not new_line and previous_blank:
                continue
            previous_blank = not new_line
            lines.append(new_line)

        stats = f.base_stats(set(drop), memory)
        stats.update({"fp_tokens_removed": fp_tokens, "fp_words": sorted(fp_words)})
        return "\n".join(lines).strip(), stats


def _with_context(hits: list[bool], unit: str) -> list[bool]:
    """Which units are kept: the hits, plus one neighbor on each side for ``sentences_ctx``."""
    if unit != "sentences_ctx":
        return hits
    n = len(hits)
    return [any(hits[j] for j in range(max(0, i - 1), min(n, i + 2))) for i in range(n)]


def compress(text: str, level: str = DEFAULT_LEVEL, memory: bool = False, **params) -> tuple[str, dict]:
    """Compress ``text`` at ``level``; ``params`` are :class:`Compressor`'s filter sizes."""
    return Compressor(level, **params).compress(text, memory=memory)
