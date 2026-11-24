"""Task-conditioned policy selectors for the ``harness.policy`` switch: ``rule_quota`` and ``bm25``.

Both keep whole chunks of the handbook, verbatim and in document order, chosen by how well they match the
task. They are deterministic, read nothing but their two arguments and use only the standard library. They
are ports of the travel-agent fleet's methods ``rule_quota_v2`` (with ``rule_quota`` and the chunker of
``cms_idf_rank_v2`` it builds on) and ``bm25_topk_v2`` (with ``bm25_topk``), flattened into one module. The
fleet's diagnostics files are not written, so nothing here touches the file system.

The task text
-------------
Each selector takes ``task_text``: the agent's system prompt (what :func:`nanoagent.plugin.policy.with_policy`
passes) or a bare task text. :func:`task_portion` reduces a NanoAgent plugin prompt (see
``nanoagent.plugin.tasks``) to the task's instructions, dropping the plugin and workspace sentence, the
``Task: <description>.`` line and the closing submit sentence, so the query is the task's own text. Its
``Rules not in scope`` section (from that heading to the next one) is removed too, so terms the task says to
ignore do not pull in paragraphs.

Shared pieces
-------------
Chunks   blank-line paragraphs; a one-line paragraph of <= 12 words not ending in '.' is a heading and is
         joined (with a blank line) to the next paragraph, so a section title travels with its body.
Terms    [a-z0-9]+ lower-cased, NLTK stopwords and tokens shorter than 3 characters dropped, one suffix of
         ing/ed/es/s stripped when >= 3 characters remain.
Title    the handbook's first line (its title) always leads the output.

rule_quota (budget-free)
------------------------
Chunks further split at lettered/roman/dotted-number sub-heading lines ("a. Lodging", "3.2.1 ...") of <= 12
words not ending in '.'; a chunk ending in ':' then absorbs the list items that follow it. A rule is a task
line whose first token is a label such as C9, G10, T4 or R5a, plus its more-indented sub-bullets (with fewer
than 3 such rules, each sentence mentioning "polic" is a rule). Each rule's terms go into their own Bloom
filter (k independent keyed blake2b hashes, sized for 0.1 expected false hits over the handbook's
vocabulary). score(chunk, rule) = IDF mass of the chunk's terms that hit the rule's filter / the rule's IDF
mass over terms in the handbook, IDF = ln((N + 1) / (df + 1)) over this handbook's chunks only. Rules in
task order: a rule whose best score s* is >= 0.15 protects up to 2 chunks it ranks, not yet protected, each
scoring >= 0.5 s*. The output is the protected chunks; a sub-chunk is preceded by its section's title line
unless that title is already shown. There is no word budget.

bm25 (budgeted)
---------------
Chunks of more than 150 words are split at outline sub-heading lines ("b. ", "3.2 ", "iv. ") of <= 7 words
not ending in '.'; a later piece is scored with its section title in front. Okapi BM25 (k1 = 1.2, b = 0.75,
IDF = ln(1 + (N - df + 0.5) / (df + 0.5)) over this handbook's chunks) ranks the pieces against the distinct
query terms. Pieces are taken best first while the running word count stays <= 1.05 B, stopping once B is
reached, where the budget B is the word count of ``bloom_filter.compress(handbook,
"flight_paragraphs+none")`` (the bloom method's relevance pass alone) clamped to 20%-50% of the handbook's
words, so bm25 is compared with bloom at a shared size. A kept piece whose preceding piece is not kept
carries its section title.
"""

from __future__ import annotations

import hashlib
import math
import re

from nanoagent.plugin import bloom_filter

MIN_TOKEN_LEN = 3
HEADING_WORDS = 12
STOPWORDS = bloom_filter.NLTK_STOPWORDS

# A NanoAgent plugin prompt (tasks.tools_prompt / task_prompt / decide_prompt): its opening sentence,
# "Task: <description>." and the instructions, then (agent tasks) the submit sentence.
PLUGIN_PROMPT = re.compile(
    r"\AYou (?:work|decide) one task .*?\n\nTask: [^\n]*\n(?P<task>.*?)(?:\n\nWhen the task is done, call \S+ with its result\.)?\Z", re.S
)
NOT_IN_SCOPE = re.compile(r"(?ims)^#+\s*Rules not in scope.*?(?=^#+\s|\Z)")


def task_portion(task_text: str) -> str:
    """The task's instructions inside a NanoAgent plugin prompt, or ``task_text`` itself when it is a bare task."""
    m = PLUGIN_PROMPT.match(task_text)
    return m.group("task") if m else task_text


def in_scope(task_text: str) -> str:
    """The task without its 'Rules not in scope' section."""
    return NOT_IN_SCOPE.sub("", task_portion(task_text))


def stem(tok: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def tokens(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) >= MIN_TOKEN_LEN]


def chunks_of(handbook: str) -> list[str]:
    """Blank-line paragraphs, each heading paragraph joined to the paragraph after it."""
    out, pending = [], []
    for par in bloom_filter.split_paragraphs(handbook):
        if "\n" not in par and len(par.split()) <= HEADING_WORDS and not par.rstrip().endswith("."):
            pending.append(par)
            continue
        out.append("\n\n".join(pending + [par]))
        pending = []
    if pending:
        out.append("\n\n".join(pending))
    return out


def first_line(handbook: str) -> str:
    return handbook.strip().splitlines()[0]


# -- rule_quota ------------------------------------------------------------------------------------------ #

FLOOR = 0.15  # a rule whose best score is below this protects nothing
RATIO = 0.5  # a protected chunk scores at least this fraction of the rule's best score
SLOTS = 2  # chunks a rule may protect
MIN_RULES = 3
EXPECTED_FALSE_HITS = 0.1  # per rule filter, over the handbook's whole vocabulary
SUBHEADING = re.compile(r"^(?:(?:[a-z]|[ivx]+)[.)]|\d+(?:\.\d+)+\.?)\s+\S")
LIST_ITEM = re.compile("^(?:[\u2022\\-*\uf0a7\u25aa]\\s+|\\d+[.)]\\s+)")  # bullet, -, *, Wingdings bullet, small square; or "1." "2)"
RULE_LABEL = re.compile(r"^\s*(?:[-*•]\s+)?\**([A-Z]{1,3}\d+[a-z]?)\b")


def subsplit(chunk: str) -> list[str]:
    """``chunk`` cut before each sub-heading line after its first line."""
    parts: list[list[str]] = [[]]
    for line in chunk.split("\n"):
        s = line.strip()
        if parts[-1] and SUBHEADING.match(s) and len(s.split()) <= HEADING_WORDS and not s.endswith("."):
            parts.append([])
        parts[-1].append(line)
    return ["\n".join(p).strip() for p in parts if "\n".join(p).strip()]


def rule_chunks(handbook: str) -> tuple[list[str], list[str | None]]:
    """Sub-split chunks, each list glued to its lead-in; parent[i] = title of the section chunk i was cut from."""
    raw: list[str] = []
    parent: list[str | None] = []
    for chunk in chunks_of(handbook):
        parts = subsplit(chunk)
        title = parts[0].strip().splitlines()[0].strip()
        for j, part in enumerate(parts):
            raw.append(part)
            parent.append(title if j > 0 else None)
    out: list[str] = []
    out_parent: list[str | None] = []
    i = 0
    while i < len(raw):
        cur, par = raw[i], parent[i]
        i += 1
        if cur.rstrip().endswith(":"):
            while i < len(raw) and LIST_ITEM.match(raw[i].lstrip()) and not SUBHEADING.match(raw[i].lstrip()):
                cur += "\n" + raw[i]
                i += 1
        out.append(cur)
        out_parent.append(par)
    return out, out_parent


def rules_of(task_text: str) -> list[str]:
    """The task's labelled rules in order (each with its sub-bullets), else its sentences mentioning "polic"."""
    task = in_scope(task_text)
    rules: list[str] = []
    open_rule, indent = False, 0
    for line in task.splitlines():
        depth = len(line) - len(line.lstrip())
        if RULE_LABEL.match(line):
            rules.append(line.strip())
            open_rule, indent = True, depth
        elif open_rule and re.match(r"^\s*[-*•]\s", line) and depth > indent:
            rules[-1] += "\n" + line.strip()
        elif line.strip():
            open_rule = False
    if len(rules) >= MIN_RULES:
        return rules
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", task) if "polic" in s.lower()]


class Bloom:
    """A Bloom filter of ``size`` bits whose ``hash_count`` positions come from independent keyed blake2b hashes."""

    def __init__(self, words: set[str], false_positive_rate: float) -> None:
        n = max(1, len(words))
        self.size = math.ceil(-n * math.log(false_positive_rate) / math.log(2) ** 2)
        self.hash_count = max(1, round(self.size / n * math.log(2)))
        self.bits = bytearray((self.size + 7) // 8)
        for word in sorted(words):
            for p in self.positions(word):
                self.bits[p >> 3] |= 1 << (p & 7)

    def positions(self, word: str) -> list[int]:
        return [
            int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8, key=b"rq%d" % i).digest(), "little") % self.size
            for i in range(self.hash_count)
        ]

    def __contains__(self, word: str) -> bool:
        return all(self.bits[p >> 3] >> (p & 7) & 1 for p in self.positions(word))


def rule_quota(handbook: str, task_text: str) -> str:
    chunks, parent = rule_chunks(handbook)
    cterms = [set(tokens(c)) for c in chunks]
    n = len(chunks)
    df: dict[str, int] = {}
    for ts in cterms:
        for t in ts:
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log((n + 1) / (d + 1)) for t, d in df.items()}
    false_positive_rate = EXPECTED_FALSE_HITS / max(1, len(df))

    protected: set[int] = set()
    for rule in rules_of(task_text):
        rterms = set(tokens(rule))
        mass = math.fsum(idf[t] for t in rterms if t in idf)
        if mass <= 0:
            continue
        bloom = Bloom(rterms, false_positive_rate)
        scores = [math.fsum(idf[t] for t in ts if t in bloom) / mass for ts in cterms]
        order = sorted((i for i in range(n) if scores[i] > 0), key=lambda i: (-scores[i], i))
        if order and scores[order[0]] >= FLOOR:
            best = scores[order[0]]
            protected.update([i for i in order if i not in protected and scores[i] >= RATIO * best][:SLOTS])
    chosen = sorted(protected)

    title = first_line(handbook)
    parts: list[str] = []
    shown: set[str] = set()
    if not chosen or not chunks[chosen[0]].startswith(title):
        parts.append(title)
        shown.add(title.strip())
    for i in chosen:
        section = parent[i]
        # A sub-chunk names its section unless that title is shown or the section's own first chunk is kept.
        if section and section not in shown and not any(parent[j] is None and chunks[j].strip().splitlines()[0][:70] == section[:70] for j in chosen):
            parts.append(section)
        if section:
            shown.add(section)
        shown.add(chunks[i].strip().splitlines()[0].strip())
        parts.append(chunks[i])
    return "\n\n".join(parts)


# -- bm25 ------------------------------------------------------------------------------------------------ #

K1, B_LEN = 1.2, 0.75
SPLIT_OVER = 150  # chunks with more words are split at their outline sub-headings
SUBHEAD_WORDS = 7
SUBHEAD = re.compile(r"^(?:[a-z]|[ivx]+|\d+(?:\.\d+)*)[.)]?\s+\S")
BUDGET_LEVEL = "flight_paragraphs+none"
BUDGET_LO, BUDGET_HI = 0.20, 0.50
SLACK = 1.05


def word_budget(handbook: str) -> int:
    """B: the words bloom's relevance pass keeps, clamped to 20%-50% of the handbook's words."""
    total = len(handbook.split())
    kept, _ = bloom_filter.compress(handbook, BUDGET_LEVEL)
    return int(min(max(len(kept.split()), BUDGET_LO * total), BUDGET_HI * total))


def split_chunk(chunk: str) -> list[str]:
    if len(chunk.split()) <= SPLIT_OVER:
        return [chunk]
    pieces: list[str] = []
    cur: list[str] = []
    for k, line in enumerate(chunk.splitlines()):
        s = line.strip()
        if k > 0 and cur and SUBHEAD.match(s) and len(s.split()) <= SUBHEAD_WORDS and not s.endswith("."):
            pieces.append("\n".join(cur))
            cur = []
        cur.append(line)
    pieces.append("\n".join(cur))
    return pieces if len(pieces) > 1 else [chunk]


def bm25_scores(query: set[str], docs: list[list[str]]) -> list[float]:
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / max(1, n)
    df = {t: sum(1 for d in docs if t in d) for t in query}
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in query}
    scores = []
    for d in docs:
        tf: dict[str, int] = {}
        for t in d:
            if t in query:
                tf[t] = tf.get(t, 0) + 1
        norm = K1 * (1 - B_LEN + B_LEN * len(d) / max(1e-9, avgdl))
        scores.append(sum(idf[t] * f * (K1 + 1) / (f + norm) for t, f in tf.items()))
    return scores


BUDGET_FRACTION = 0.35  # bm25_frac: the midpoint of bm25's 20%-50% clamp, chosen before any run (not tuned)


def bm25_frac(handbook: str, task_text: str, fraction: float = BUDGET_FRACTION) -> str:
    """bm25 with a vocabulary-free budget: B = ``fraction`` of the handbook's words (an ablation of B)."""
    return bm25(handbook, task_text, budget=int(fraction * len(handbook.split())))


def bm25(handbook: str, task_text: str, budget: int | None = None) -> str:
    units = []  # (piece, chunk index, piece index, chunk title)
    for ci, chunk in enumerate(chunks_of(handbook)):
        title = chunk.strip().splitlines()[0]
        units.extend((piece, ci, pi, title) for pi, piece in enumerate(split_chunk(chunk)))
    scored = [piece if pi == 0 else f"{title}\n{piece}" for piece, _, pi, title in units]
    scores = bm25_scores(set(tokens(in_scope(task_text))), [tokens(t) for t in scored])

    budget = word_budget(handbook) if budget is None else budget
    chosen, total = [], 0
    for i in sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i)):
        if total >= budget:
            break
        words = len(scored[i].split())
        if total + words <= SLACK * budget:
            chosen.append(i)
            total += words
    kept = set(chosen)

    parts = []
    for i in sorted(chosen):
        piece, ci, pi, title = units[i]
        previous_kept = i - 1 in kept and units[i - 1][1] == ci
        parts.append(piece if pi == 0 or previous_kept else f"{title}\n{piece}")
    if not parts or not parts[0].startswith(first_line(handbook)):
        parts.insert(0, first_line(handbook))
    return "\n\n".join(parts)
