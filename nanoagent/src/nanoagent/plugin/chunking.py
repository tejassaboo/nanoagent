"""The handbook chunkers and the term tokenizer that every sketch selector shares (standard library only).

Deterministic and shared, so a bot comparison changes one selector field and never the units it ranks.

plain       blank-line paragraphs, nothing else (side rung C0).
structured  the pieces :func:`nanoagent.plugin.policy_select.bm25` scores (``chunks_of`` + ``split_chunk``):
            a one-line paragraph of at most 12 words not ending in '.' is a heading and is joined to the
            paragraph after it; a chunk of more than 150 words is cut before outline sub-heading lines
            ("b. Air", "iv Lodging", "3.2 Rates") of at most 7 words not ending in '.'; every later piece
            starts with its section's first line, so it is scored (and shown) under its title.
            Guarantee: on any handbook the pieces are byte-identical to the ones bm25 scores.
tokens      ``policy_select.tokens``: [a-z0-9]+ runs, lower-cased, NLTK English stopwords and tokens shorter
            than 3 characters dropped, then one suffix of ing/ed/es/s stripped when 3 or more characters remain.
"""

from __future__ import annotations

import re

from nanoagent.plugin.bloom_filter import NLTK_STOPWORDS as STOPWORDS

MIN_TOKEN_LEN = 3
SUFFIXES = ("ing", "ed", "es", "s")
HEADING_WORDS = 12  # a heading paragraph has at most this many words
SPLIT_OVER = 150  # a structured chunk with more words is cut at its sub-headings
SUBHEAD_WORDS = 7
SUBHEAD = re.compile(r"^(?:[a-z]|[ivx]+|\d+(?:\.\d+)*)[.)]?\s+\S")


def stem(token: str) -> str:
    """``token`` without its first matching suffix of ing/ed/es/s, when at least 3 characters remain."""
    for suffix in SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= MIN_TOKEN_LEN:
            return token[: -len(suffix)]
    return token


def tokens(text: str) -> list[str]:
    """The stemmed terms of ``text``, in order (repeats kept)."""
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) >= MIN_TOKEN_LEN]


def plain_chunks(handbook: str) -> list[str]:
    """Blocks of text between blank lines, stripped."""
    return [p.strip() for p in re.split(r"\n\s*\n", handbook) if p.strip()]


def _sections(handbook: str) -> list[str]:
    """Paragraphs, each run of heading paragraphs joined (blank line between) to the paragraph after it."""
    out: list[str] = []
    pending: list[str] = []
    for par in plain_chunks(handbook):
        if "\n" not in par and len(par.split()) <= HEADING_WORDS and not par.endswith("."):
            pending.append(par)
            continue
        out.append("\n\n".join(pending + [par]))
        pending = []
    return out + (["\n\n".join(pending)] if pending else [])


def _is_subhead(line: str) -> bool:
    s = line.strip()
    return bool(SUBHEAD.match(s)) and len(s.split()) <= SUBHEAD_WORDS and not s.endswith(".")


def _pieces(section: str) -> list[str]:
    """``section`` cut before its sub-heading lines when it has more than SPLIT_OVER words; later pieces get its title."""
    lines = section.splitlines()
    if len(section.split()) <= SPLIT_OVER:
        return [section]
    groups: list[list[str]] = [[]]
    for k, line in enumerate(lines):
        if k > 0 and groups[-1] and _is_subhead(line):
            groups.append([])
        groups[-1].append(line)
    if len(groups) == 1:
        return [section]
    return ["\n".join(g) if j == 0 else lines[0] + "\n" + "\n".join(g) for j, g in enumerate(groups)]


def structured_chunks(handbook: str) -> list[str]:
    """Heading-joined sections, long ones cut at their outline sub-headings (see the module docstring)."""
    return [piece for section in _sections(handbook) for piece in _pieces(section)]


CHUNKERS = {"plain": plain_chunks, "structured": structured_chunks}
