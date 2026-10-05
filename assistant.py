"""A deliberately simple stand-in for a policy-question assistant: the system under test.

It is NOT a real RAG system. Retrieval is BM25 keyword matching and "generation"
is quoting the top-ranked sections with citations, so the harness runs anywhere with no API key.
The point is the interface: the harness only calls `Assistant.ask()`, so a real
LLM-backed assistant can be swapped in without touching the eval.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it
its me my of on or our should so that the their them these they this to under was we what
when which who will with you your any all not""".split())

# Scope control: topics the assistant must hand to a human, never answer itself.
ESCALATE_TERMS = ["discriminat", "national origin", "race", "lawsuit", "sued", "legal advice"]


def stem(word):
    """Crude suffix stripping so 'waive', 'waiver' and 'waivers' match."""
    for suffix in ("s", "ing", "ed", "er", "e"):
        if len(word) > 4 and word.endswith(suffix):
            word = word[: -len(suffix)]
    return word


def tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS]


@dataclass
class Chunk:
    doc_id: str
    title: str
    version: str
    effective: str
    status: str
    section: str
    text: str


@dataclass
class Response:
    behaviour: str                      # ANSWER | REFUSE | ESCALATE
    text: str
    citations: list = field(default_factory=list)   # chunks the answer quotes
    retrieved: list = field(default_factory=list)   # [(score, Chunk)], ranked


def load_corpus(corpus_dir, include_superseded=False):
    chunks = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        raw = path.read_text()
        _, front, body = raw.split("---", 2)
        meta = dict(line.split(": ", 1) for line in front.strip().splitlines())
        if meta["status"] != "current" and not include_superseded:
            continue                    # corpus control: superseded docs are not indexed
        for block in re.split(r"^## ", body, flags=re.M)[1:]:
            heading, _, text = block.partition("\n")
            chunks.append(Chunk(meta["doc_id"], meta["title"], meta["version"],
                                meta["effective"], meta["status"],
                                heading.split(" ", 1)[0], text.strip()))
    return chunks


class Assistant:
    def __init__(self, corpus_dir, k=3, refuse_below=1.0, answer_from=2, max_sentences=3,
                 include_superseded=False):
        self.k, self.refuse_below = k, refuse_below
        self.answer_from, self.max_sentences = answer_from, max_sentences
        self.chunks = load_corpus(corpus_dir, include_superseded)
        self.docs = [Counter(tokens(c.title + " " + c.text)) for c in self.chunks]
        self.avg_len = sum(sum(d.values()) for d in self.docs) / len(self.docs)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def retrieve(self, question, k1=1.2, b=0.75):
        """BM25: the standard keyword ranking function."""
        terms = set(tokens(question))
        scored = []
        for chunk, doc in zip(self.chunks, self.docs):
            norm = k1 * (1 - b + b * sum(doc.values()) / self.avg_len)
            score = sum(self.idf[t] * doc[t] * (k1 + 1) / (doc[t] + norm)
                        for t in terms if t in doc)
            scored.append((round(score, 2), chunk))
        scored.sort(key=lambda s: -s[0])
        return scored[: self.k]

    def ask(self, question):
        retrieved = self.retrieve(question)
        if any(term in question.lower() for term in ESCALATE_TERMS):
            return Response("ESCALATE", "This needs a human: route it to Compliance today.",
                            [], retrieved)
        if not retrieved or retrieved[0][0] < self.refuse_below:
            return Response("REFUSE", "I can't answer that from the policy library. "
                            "Please contact Legal or Compliance.", [], retrieved)
        used = [c for s, c in retrieved[: self.answer_from] if s >= self.refuse_below]
        quoted = [" ".join(re.split(r"(?<=\.)\s+", c.text)[: self.max_sentences]) +
                  f" [{c.doc_id} v{c.version} §{c.section}]" for c in used]
        return Response("ANSWER", " ".join(quoted), used, retrieved)
