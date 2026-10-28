"""
Knowledge base retrieval (the "R" in RAG) over SOPs and past incident reports.

Pipeline
--------
1. Load      every markdown file under knowledge/ (sops/ and incidents/).
2. Chunk     SOPs are split at "## " headings (one chunk per section);
             incident reports are short, so each one stays a single chunk.
             Every chunk starts with its document title ("contextual header"), so a
             section like "## Actions" still says WHICH procedure it belongs to.
3. Index     a) BM25 keyword index   -> exact terms: equipment codes (P-301), tag names.
             b) Embedding index      -> meaning: "brightness fell" ~ "pulp looked darker".
4. Search    run both, then merge the two rankings with Reciprocal Rank Fusion.

Why hybrid?
  Embeddings are weak at exact identifiers: "P-301" and "P-310" look almost the same
  to an embedding model, but to an engineer they are different pumps. BM25 is weak at
  paraphrase: "holes and spots" will not match a query about "WIS defects" unless the
  words overlap. Industrial text is full of both, so we use both.

Why Reciprocal Rank Fusion (RRF)?
  BM25 scores (0-20+) and cosine similarities (0-1) are on different scales, so adding
  them is meaningless. RRF ignores raw scores and only uses each result's RANK:
      rrf_score(doc) = sum over retrievers of 1 / (k + rank)      (k = 60 by convention)
  A document ranked well by both retrievers rises to the top.

Embeddings are optional and free by default: `fastembed` runs a small embedding model
(BAAI/bge-small-en-v1.5) locally on your laptop. OpenAI embeddings can be used instead
(EMBEDDINGS=openai). With EMBEDDINGS=none the search runs on BM25 alone.
"""
from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from rca.config import embeddings_provider

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "knowledge"
EMBED_CACHE = ROOT / "data" / "embedding_cache.json"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RRF_K = 60

STOPWORDS = set("""a an and are as at be by for from has have if in into is it its of on or
that the their then there these this to was were when which with without not no do does""".split())


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: str          # "sop" or "incident"
    title: str
    section: str
    text: str              # what gets indexed and shown to the LLM (includes the title)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1-2. Load and chunk
# ---------------------------------------------------------------------------
def load_chunks(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(knowledge_dir.rglob("*.md")):
        doc_type = "sop" if path.parent.name == "sops" else "incident"
        raw = path.read_text(encoding="utf-8").strip()
        title_line, _, body = raw.partition("\n")
        title = title_line.lstrip("# ").strip()
        doc_id = title.split(":")[0].strip()          # "SOP-BL-01", "INC-2025-014"

        if doc_type == "incident":
            chunks.append(Chunk(f"{doc_id}#0", doc_id, doc_type, title, "full report", f"{title}\n{body.strip()}"))
            continue

        # Split SOPs at level-2 headings. re.split keeps the heading text as separate items.
        parts = re.split(r"^## +(.+)$", body, flags=re.MULTILINE)
        for i in range(1, len(parts), 2):
            section, content = parts[i].strip(), parts[i + 1].strip()
            chunks.append(Chunk(f"{doc_id}#{i // 2}", doc_id, doc_type, title, section,
                                f"{title} | {section}\n{content}"))
    return chunks


def tokenize(text: str) -> list[str]:
    """Lowercase words, keeping identifiers intact: 'P-301', 'bleach_temp', 'h2o2'."""
    tokens = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


# ---------------------------------------------------------------------------
# 3b. Embeddings (optional, cached)
# ---------------------------------------------------------------------------
class Embedder:
    """Turns text into unit-length vectors, with an on-disk cache so each document is
    embedded once, not on every run. Subclasses implement `_embed_raw`."""
    model = "base"

    def __init__(self, cache_path: Path = EMBED_CACHE):
        self.cache_path = cache_path
        self.cache: dict[str, list[float]] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def _key(self, text: str, is_query: bool) -> str:
        return hashlib.sha256(f"{self.model}::{is_query}::{text}".encode()).hexdigest()

    def _embed_raw(self, texts: list[str], is_query: bool) -> list[list[float]]:
        raise NotImplementedError

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        missing = [t for t in texts if self._key(t, is_query) not in self.cache]
        if missing:
            for t, vec in zip(missing, self._embed_raw(missing, is_query)):
                self.cache[self._key(t, is_query)] = [float(v) for v in vec]
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))
        vecs = np.array([self.cache[self._key(t, is_query)] for t in texts], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)     # unit length -> dot product = cosine


class LocalEmbedder(Embedder):
    """Free, offline embeddings with fastembed (downloads a ~70 MB model on first use).
    bge models embed queries and passages slightly differently, hence `is_query`."""
    model = LOCAL_EMBED_MODEL

    def __init__(self, **kw):
        super().__init__(**kw)
        from fastembed import TextEmbedding
        self.engine = TextEmbedding(self.model)

    def _embed_raw(self, texts, is_query):
        fn = self.engine.query_embed if is_query else self.engine.passage_embed
        return list(fn(texts))


class OpenAIEmbedder(Embedder):
    model = OPENAI_EMBED_MODEL

    def __init__(self, **kw):
        super().__init__(**kw)
        from openai import OpenAI
        self.client = OpenAI(default_headers={"Accept-Encoding": "gzip, deflate"})

    def _embed_raw(self, texts, is_query):
        return [d.embedding for d in self.client.embeddings.create(model=self.model, input=texts).data]


def make_embedder(provider: str) -> Embedder | None:
    """Create the embedder, or return None (BM25-only) if it cannot load - e.g. no internet
    for the first model download. Degrading gracefully beats crashing the whole assistant."""
    cls = {"fastembed": LocalEmbedder, "openai": OpenAIEmbedder}.get(provider)
    if cls is None:
        return None
    try:
        return cls()
    except Exception as err:                      # noqa: BLE001
        warnings.warn(f"Embeddings ({provider}) unavailable, using BM25 only: {err}")
        return None


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """rankings: one list of chunk indices per retriever, best first."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


# ---------------------------------------------------------------------------
# 4. The knowledge base
# ---------------------------------------------------------------------------
class KnowledgeBase:
    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR, use_embeddings: bool | None = None):
        """use_embeddings: None = decide from settings, False = BM25 only."""
        self.chunks = load_chunks(knowledge_dir)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        provider = embeddings_provider() if use_embeddings is not False else "none"
        self.embedder = make_embedder(provider)
        self.vectors = self.embedder.embed([c.text for c in self.chunks]) if self.embedder else None

    @property
    def mode(self) -> str:
        return "hybrid" if self.embedder else "bm25"

    def search(self, query: str, k: int = 4, doc_type: str | None = None, candidates: int = 10) -> list[dict]:
        """Return the top-k chunks for `query`. Each retriever proposes `candidates`
        results; RRF merges them. Optionally restrict to doc_type 'sop' or 'incident'."""
        allowed = [i for i, c in enumerate(self.chunks) if doc_type in (None, c.doc_type)]

        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_rank = sorted(allowed, key=lambda i: bm25_scores[i], reverse=True)[:candidates]
        rankings = [bm25_rank]

        vec_rank: list[int] = []
        if self.embedder is not None:
            q = self.embedder.embed([query], is_query=True)[0]
            sims = self.vectors @ q
            vec_rank = sorted(allowed, key=lambda i: sims[i], reverse=True)[:candidates]
            rankings.append(vec_rank)

        fused = reciprocal_rank_fusion(rankings)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{
            "chunk_id": self.chunks[i].chunk_id,
            "doc_id": self.chunks[i].doc_id,
            "doc_type": self.chunks[i].doc_type,
            "title": self.chunks[i].title,
            "section": self.chunks[i].section,
            "text": self.chunks[i].text,
            "score": round(fused[i], 4),
            "bm25_rank": bm25_rank.index(i) + 1 if i in bm25_rank else None,
            "vector_rank": vec_rank.index(i) + 1 if i in vec_rank else None,
        } for i in top]
