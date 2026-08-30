# /// zerodep
# version = "0.4.0"
# deps = []
# tier = "medium"
# category = "text"
# note = "Install/update via: https://zerodep.readthedocs.io/en/latest/guide/cli/"
# ///
"""Zero-dependency sparse text search with BM25 family and TF-IDF ranking.

Part of zerodep: https://github.com/Oaklight/zerodep
Copyright (c) 2026 Peng Ding. MIT License.

Full-text search engine with inverted index, supporting BM25/BM25+/BM25L/BM25F
and TF-IDF+Cosine similarity ranking. Designed for use in LLM/Agent/RAG pipelines.

Basic usage::

    index = SparseIndex()
    index.add("doc1", "the quick brown fox")
    index.add("doc2", "the lazy dog")
    results = index.search("quick fox")

Multi-field (BM25F) usage::

    index = SparseIndex(field_weights={"title": 2.0, "body": 1.0})
    index.add("doc1", {"title": "Python Guide", "body": "Learn Python basics"})
    results = index.search("python")

Custom tokenizer::

    import jieba
    index = SparseIndex(tokenize=jieba.lcut)

Persistence::

    index.save("index.json")
    index = SparseIndex.load("index.json")
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "Result",
    "SparseIndex",
    "jaccard_similarity",
    "rrf",
    "mmr",
]

_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Default tokenizer
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"[\w]+", re.UNICODE)

# ---------------------------------------------------------------------------
# Bayesian probability utilities
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _logit(p: float) -> float:
    """Logit (inverse sigmoid) with epsilon clamping."""
    p = max(1e-10, min(1.0 - 1e-10, p))
    return math.log(p / (1.0 - p))


def _bayesian_likelihood(score: float, alpha: float, beta: float) -> float:
    """Sigmoid likelihood: L(s) = sigmoid(alpha * (s - beta))."""
    return _sigmoid(alpha * (score - beta))


def _tf_prior(tf: float) -> float:
    """Term-frequency prior: P_tf = 0.2 + 0.7 * min(1, tf/10)."""
    return 0.2 + 0.7 * min(1.0, tf / 10.0)


def _norm_prior(doc_len_ratio: float) -> float:
    """Doc-length normalization prior (bell curve peaking at ratio=0.5)."""
    return 0.3 + 0.6 * (1.0 - min(1.0, abs(doc_len_ratio - 0.5) * 2.0))


def _composite_prior(tf: float, doc_len_ratio: float) -> float:
    """Composite prior: 0.7*P_tf + 0.3*P_norm, clamped to [0.1, 0.9]."""
    return max(0.1, min(0.9, 0.7 * _tf_prior(tf) + 0.3 * _norm_prior(doc_len_ratio)))


def _bayesian_posterior(
    likelihood: float,
    prior: float,
    base_rate: float | None = None,
) -> float:
    """Two-step Bayesian posterior update."""
    # Step 1: standard Bayes
    p = likelihood * prior / (likelihood * prior + (1.0 - likelihood) * (1.0 - prior))
    # Step 2: optional base_rate correction
    if base_rate is not None:
        p = p * base_rate / (p * base_rate + (1.0 - p) * (1.0 - base_rate))
    return p


def _score_to_probability(
    score: float,
    tf: float,
    doc_len_ratio: float,
    alpha: float,
    beta: float,
    base_rate: float | None = None,
) -> float:
    """Full pipeline: BM25 score -> calibrated probability."""
    L = _bayesian_likelihood(score, alpha, beta)
    prior = _composite_prior(tf, doc_len_ratio)
    return _bayesian_posterior(L, prior, base_rate)


def _prob_or(probs: list[float]) -> float:
    """P(A or B or ...) = 1 - prod(1 - p_i), in log-space for stability."""
    log_complement = sum(math.log(max(1e-10, 1.0 - p)) for p in probs)
    return 1.0 - math.exp(log_complement)


def _log_odds_conjunction(probs: list[float], alpha: float = 0.5) -> float:
    """Log-odds conjunction: sigmoid(mean(logit(p_i)) * n^alpha)."""
    n = len(probs)
    if n == 0:
        return 0.0
    mean_logit = sum(_logit(p) for p in probs) / n
    return _sigmoid(mean_logit * (n**alpha))


def _default_tokenize(text: str) -> list[str]:
    """Simple whitespace/punctuation tokenizer with lowercasing.

    Splits on non-word characters and lowercases all tokens.
    For better results (stemming, stop-word removal, CJK segmentation),
    pass a custom tokenizer to SparseIndex.
    """
    return [tok.lower() for tok in _SPLIT_RE.findall(text)]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """A single search result."""

    doc_id: str
    score: float
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Internal document record
# ---------------------------------------------------------------------------


@dataclass
class _DocRecord:
    """Internal storage for a single document."""

    doc_id: str
    field_lengths: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# SparseIndex
# ---------------------------------------------------------------------------


class SparseIndex:
    """Sparse text search index with BM25 family and TF-IDF ranking.

    Args:
        variant: Ranking algorithm. One of "bm25", "bm25l", "tfidf".
        k1: BM25 term-frequency saturation parameter (ignored for tfidf).
        b: BM25 document-length normalization parameter (ignored for tfidf).
        delta: BM25+ lower-bound correction. Set to 0 for classic BM25.
        field_weights: Per-field boost weights for BM25F multi-field search.
            When ``None``, documents are treated as single-field text.
        tokenize: Tokenization function ``str -> list[str]``.
            Defaults to a simple Unicode word splitter with lowercasing.
        calibrated: If ``True``, BM25 scores are converted to calibrated
            probabilities in [0, 1] via Bayesian BM25. Call :meth:`calibrate`
            to estimate or provide the calibration parameters.
    """

    _SQLITE_MAGIC = b"SQLite format 3\x00"

    def __init__(
        self,
        variant: str = "bm25",
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        field_weights: dict[str, float] | None = None,
        tokenize: Callable[[str], list[str]] | None = None,
        calibrated: bool = False,
    ) -> None:
        if variant not in ("bm25", "bm25l", "tfidf"):
            raise ValueError(
                f"Unknown variant {variant!r}, expected 'bm25', 'bm25l', or 'tfidf'"
            )

        self.variant = variant
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.field_weights = field_weights
        self._tokenize = tokenize or _default_tokenize
        self.calibrated = calibrated

        # Bayesian BM25 calibration parameters
        self._alpha: float | None = None
        self._beta: float | None = None
        self._base_rate: float | None = None

        # Document storage: doc_id -> _DocRecord
        self._docs: dict[str, _DocRecord] = {}

        # Inverted index: term -> {doc_id -> {field -> term_freq}}
        self._index: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        # Document frequency: term -> number of documents containing it
        self._df: dict[str, int] = defaultdict(int)

        # Total field lengths for average calculation
        self._total_field_lengths: dict[str, float] = defaultdict(float)

        # Reverse index: doc_id -> set of terms (for fast delete)
        self._doc_terms: dict[str, set[str]] = defaultdict(set)

    # -- properties ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._docs)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._docs

    @property
    def doc_count(self) -> int:
        """Number of documents in the index."""
        return len(self._docs)

    @property
    def vocab_size(self) -> int:
        """Number of unique terms in the index."""
        return len(self._index)

    # -- document management -------------------------------------------------

    def add(
        self,
        doc_id: str,
        content: str | dict[str, str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a document to the index.

        Args:
            doc_id: Unique document identifier.
            content: Document text (single-field) or ``{field: text}`` dict
                (multi-field for BM25F).
            metadata: Arbitrary JSON-serializable metadata attached to the doc.

        Raises:
            ValueError: If ``doc_id`` already exists (use ``update`` instead).
        """
        if doc_id in self._docs:
            raise ValueError(
                f"Document {doc_id!r} already exists, use update() instead"
            )
        self._insert(doc_id, content, metadata)

    def remove(self, doc_id: str) -> None:
        """Remove a document from the index.

        Args:
            doc_id: Document identifier to remove.

        Raises:
            KeyError: If ``doc_id`` does not exist.
        """
        if doc_id not in self._docs:
            raise KeyError(f"Document {doc_id!r} not found")
        self._delete(doc_id)

    def update(
        self,
        doc_id: str,
        content: str | dict[str, str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update (replace) a document in the index.

        Args:
            doc_id: Document identifier to update.
            content: New document text or ``{field: text}`` dict.
            metadata: New metadata (replaces old metadata entirely).

        Raises:
            KeyError: If ``doc_id`` does not exist (use ``add`` instead).
        """
        if doc_id not in self._docs:
            raise KeyError(f"Document {doc_id!r} not found, use add() instead")
        self._delete(doc_id)
        self._insert(doc_id, content, metadata)

    # -- search --------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Result]:
        """Search for documents matching the query.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            filters: Metadata filters. Each key-value pair must match exactly
                in the document's metadata. Use a callable value for custom
                filter logic (e.g. ``{"year": lambda y: y > 2023}``).

        Returns:
            List of :class:`Result` objects sorted by descending score.
        """
        if not self._docs:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        if self.variant == "tfidf":
            scores = self._score_tfidf(query_tokens)
        else:
            scores = self._score_bm25(query_tokens)

        # Apply metadata filters
        if filters:
            scores = {
                doc_id: score
                for doc_id, score in scores.items()
                if self._match_filters(doc_id, filters)
            }

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            Result(
                doc_id=doc_id,
                score=score,
                metadata=self._docs[doc_id].metadata,
            )
            for doc_id, score in ranked
            if score > 0
        ]

    # -- calibration ---------------------------------------------------------

    def calibrate(
        self,
        *,
        alpha: float | None = None,
        beta: float | None = None,
        base_rate: float | None = None,
        n_samples: int = 50,
    ) -> None:
        """Estimate Bayesian calibration parameters from corpus statistics.

        When both ``alpha`` and ``beta`` are provided they are used directly.
        Otherwise they are auto-estimated by sampling pseudo-queries from the
        corpus and collecting raw BM25 scores:

        * ``beta = median(scores)`` -- the score midpoint.
        * ``alpha = 1 / stdev(scores)`` -- controls sigmoid steepness.

        After calling this method, :meth:`search` with a BM25 variant will
        return calibrated probabilities in [0, 1] instead of raw scores.

        Args:
            alpha: Sigmoid steepness parameter. Auto-estimated when ``None``.
            beta: Sigmoid midpoint parameter. Auto-estimated when ``None``.
            base_rate: Optional base-rate for a second Bayesian correction
                step. ``None`` skips the correction (the default).
            n_samples: Number of pseudo-queries for auto-estimation.

        Raises:
            RuntimeError: If the index is empty or auto-estimation fails.
        """
        if alpha is not None and beta is not None:
            self._alpha = alpha
            self._beta = beta
            self._base_rate = base_rate
            self.calibrated = True
            return

        if not self._docs:
            raise RuntimeError("Cannot auto-calibrate on an empty index")

        # Sample documents and use their first tokens as pseudo-queries
        doc_ids = list(self._docs.keys())
        step = max(1, len(doc_ids) // n_samples)
        sampled_ids = doc_ids[::step][:n_samples]

        all_scores: list[float] = []
        for doc_id in sampled_ids:
            # Collect up to 5 terms from this document
            terms = list(self._doc_terms.get(doc_id, set()))[:5]
            if not terms:
                continue
            scores = self._score_bm25(terms)
            all_scores.extend(s for s in scores.values() if s > 0)

        if len(all_scores) < 2:
            raise RuntimeError(
                "Not enough score samples for auto-calibration "
                f"(got {len(all_scores)}, need >= 2)"
            )

        self._beta = statistics.median(all_scores)
        stdev = statistics.stdev(all_scores)
        self._alpha = 1.0 / stdev if stdev > 0 else 1.0
        self._base_rate = base_rate
        self.calibrated = True

    # -- persistence ---------------------------------------------------------

    def save(self, path: str, format: str | None = None) -> None:
        """Save the index to disk.

        Args:
            path: File path to save to.
            format: ``"json"`` or ``"sqlite"``. If ``None``, inferred from
                file extension (``.db`` -> sqlite, otherwise json).
        """
        fmt = self._resolve_format(path, format)
        if fmt == "json":
            self._save_json(path)
        else:
            self._save_sqlite(path)

    @classmethod
    def load(cls, path: str) -> SparseIndex:
        """Load an index from disk.

        Auto-detects format by inspecting the file header.

        Args:
            path: File path to load from.

        Returns:
            A new :class:`SparseIndex` instance with restored state.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        with open(p, "rb") as f:
            header = f.read(16)

        if header.startswith(cls._SQLITE_MAGIC):
            return cls._load_sqlite(path)
        return cls._load_json(path)

    # -- internal: insert / delete -------------------------------------------

    def _normalize_content(self, content: str | dict[str, str]) -> dict[str, str]:
        """Convert content to {field: text} dict."""
        if isinstance(content, str):
            return {"_default": content}
        return dict(content)

    def _insert(
        self,
        doc_id: str,
        content: str | dict[str, str],
        metadata: dict[str, Any] | None,
    ) -> None:
        fields = self._normalize_content(content)
        field_lengths: dict[str, int] = {}
        seen_terms: set[str] = set()

        for field_name, text in fields.items():
            tokens = self._tokenize(text)
            field_lengths[field_name] = len(tokens)
            self._total_field_lengths[field_name] += len(tokens)

            for token in tokens:
                self._index[token][doc_id][field_name] += 1
                seen_terms.add(token)

        for term in seen_terms:
            self._df[term] += 1

        self._doc_terms[doc_id] = seen_terms

        self._docs[doc_id] = _DocRecord(
            doc_id=doc_id,
            field_lengths=field_lengths,
            metadata=metadata,
        )

    def _delete(self, doc_id: str) -> None:
        doc = self._docs[doc_id]

        # Update total field lengths
        for field_name, length in doc.field_lengths.items():
            self._total_field_lengths[field_name] -= length

        # Remove from inverted index using reverse index (O(terms_in_doc)
        # instead of O(vocab_size))
        empty_terms: list[str] = []
        for term in self._doc_terms[doc_id]:
            postings = self._index.get(term)
            if postings is not None and doc_id in postings:
                del postings[doc_id]
                self._df[term] -= 1
                if self._df[term] <= 0:
                    empty_terms.append(term)

        # Clean up empty terms
        for term in empty_terms:
            del self._index[term]
            del self._df[term]

        del self._doc_terms[doc_id]
        del self._docs[doc_id]

    # -- internal: BM25 scoring ----------------------------------------------

    def _avg_field_length(self, field_name: str) -> float:
        n = len(self._docs)
        if n == 0:
            return 0.0
        return self._total_field_lengths[field_name] / n

    def _idf(self, term: str) -> float:
        """Compute IDF for a term using the standard BM25 IDF formula."""
        n = len(self._docs)
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def _weighted_term_freq(
        self,
        field_tfs: dict[str, int],
        field_weights: dict[str, float],
    ) -> float:
        """Compute BM25F weighted pseudo term frequency across fields."""
        return sum(field_weights.get(fn, 1.0) * tf for fn, tf in field_tfs.items())

    def _weighted_doc_length(
        self,
        doc: _DocRecord,
        field_weights: dict[str, float],
    ) -> float:
        """Compute weighted document length across fields."""
        return sum(
            field_weights.get(fn, 1.0) * dl for fn, dl in doc.field_lengths.items()
        )

    def _weighted_avg_doc_length(self, field_weights: dict[str, float]) -> float:
        """Compute weighted average document length across fields."""
        avgdl = sum(w * self._avg_field_length(fn) for fn, w in field_weights.items())
        return avgdl if avgdl != 0 else 1.0

    def _bm25_tf_norm(
        self,
        weighted_tf: float,
        weighted_dl: float,
        weighted_avgdl: float,
    ) -> float:
        """Compute BM25/BM25+/BM25L TF normalization factor."""
        k1, b, delta = self.k1, self.b, self.delta
        if self.variant == "bm25l":
            ctf = weighted_tf / (1.0 - b + b * weighted_dl / weighted_avgdl)
            return ((k1 + 1.0) * (ctf + delta)) / (k1 + ctf + delta)
        tf_norm = (weighted_tf * (k1 + 1.0)) / (
            weighted_tf + k1 * (1.0 - b + b * weighted_dl / weighted_avgdl)
        )
        return tf_norm + delta

    def _calibrate_bm25_scores(
        self,
        raw_scores: dict[str, float],
        query_tokens: list[str],
        field_weights: dict[str, float],
    ) -> dict[str, float]:
        """Convert raw BM25 scores to calibrated probabilities via Bayesian BM25."""
        alpha = self._alpha
        beta = self._beta
        base_rate = self._base_rate
        weighted_avgdl = self._weighted_avg_doc_length(field_weights)

        calibrated: dict[str, float] = {}
        for doc_id, raw_score in raw_scores.items():
            doc = self._docs[doc_id]
            total_tf = self._total_query_tf(doc_id, query_tokens, field_weights)
            w_dl = self._weighted_doc_length(doc, field_weights)
            doc_len_ratio = w_dl / weighted_avgdl
            calibrated[doc_id] = _score_to_probability(
                raw_score,
                total_tf,
                doc_len_ratio,
                alpha,
                beta,
                base_rate,  # type: ignore[arg-type]
            )
        return calibrated

    def _total_query_tf(
        self,
        doc_id: str,
        query_tokens: list[str],
        field_weights: dict[str, float],
    ) -> float:
        """Sum weighted term frequencies for all query tokens in a document."""
        total = 0.0
        for token in query_tokens:
            postings = self._index.get(token)
            if postings is not None and doc_id in postings:
                total += self._weighted_term_freq(postings[doc_id], field_weights)
        return total

    def _score_bm25(self, query_tokens: list[str]) -> dict[str, float]:
        """Score documents using BM25 / BM25+ / BM25L / BM25F."""
        scores: dict[str, float] = defaultdict(float)
        field_weights = self.field_weights or {"_default": 1.0}

        for token in query_tokens:
            if token not in self._index:
                continue

            idf = self._idf(token)
            weighted_avgdl = self._weighted_avg_doc_length(field_weights)

            for doc_id, field_tfs in self._index[token].items():
                doc = self._docs[doc_id]
                w_tf = self._weighted_term_freq(field_tfs, field_weights)
                w_dl = self._weighted_doc_length(doc, field_weights)
                scores[doc_id] += idf * self._bm25_tf_norm(w_tf, w_dl, weighted_avgdl)

        if not self.calibrated or self._alpha is None or self._beta is None:
            return dict(scores)

        return self._calibrate_bm25_scores(dict(scores), query_tokens, field_weights)

    # -- internal: TF-IDF scoring --------------------------------------------

    def _tfidf_idf(self, term: str, n: int) -> float:
        """Compute smoothed IDF for TF-IDF scoring."""
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(n / df) + 1.0

    def _build_query_tfidf_vec(
        self, query_tokens: list[str], n: int
    ) -> dict[str, float]:
        """Build TF-IDF weighted vector for query terms."""
        query_tf: dict[str, int] = defaultdict(int)
        for token in query_tokens:
            query_tf[token] += 1

        vec: dict[str, float] = {}
        for term, tf in query_tf.items():
            idf = self._tfidf_idf(term, n)
            if idf > 0:
                vec[term] = (1.0 + math.log(tf)) * idf
        return vec

    def _doc_tfidf_vec(
        self,
        doc_id: str,
        n: int,
        field_weights: dict[str, float],
    ) -> dict[str, float]:
        """Build TF-IDF weighted vector for a document."""
        vec: dict[str, float] = {}
        for term in self._doc_terms.get(doc_id, ()):
            postings = self._index.get(term)
            if postings is None or doc_id not in postings:
                continue
            w_tf = self._weighted_term_freq(postings[doc_id], field_weights)
            if w_tf <= 0:
                continue
            idf = self._tfidf_idf(term, n)
            if idf > 0:
                vec[term] = (1.0 + math.log(w_tf)) * idf
        return vec

    def _score_tfidf(self, query_tokens: list[str]) -> dict[str, float]:
        """Score documents using TF-IDF with cosine similarity."""
        n = len(self._docs)
        if n == 0:
            return {}

        field_weights = self.field_weights or {"_default": 1.0}
        query_vec = self._build_query_tfidf_vec(query_tokens, n)
        if not query_vec:
            return {}

        query_norm = math.sqrt(sum(v * v for v in query_vec.values()))

        # Collect candidate documents that contain at least one query term
        candidates: set[str] = set()
        for term in query_vec:
            if term in self._index:
                candidates.update(self._index[term].keys())

        scores: dict[str, float] = {}
        for doc_id in candidates:
            doc_vec = self._doc_tfidf_vec(doc_id, n, field_weights)
            dot_product = sum(
                w * query_vec[t] for t, w in doc_vec.items() if t in query_vec
            )
            doc_norm_sq = sum(w * w for w in doc_vec.values())
            if doc_norm_sq > 0 and dot_product > 0:
                scores[doc_id] = dot_product / (query_norm * math.sqrt(doc_norm_sq))

        return scores

    # -- internal: metadata filtering ----------------------------------------

    def _match_filters(self, doc_id: str, filters: dict[str, Any]) -> bool:
        """Check if a document's metadata matches all filters."""
        doc = self._docs[doc_id]
        if doc.metadata is None:
            return False

        for key, expected in filters.items():
            actual = doc.metadata.get(key)
            if callable(expected):
                if not expected(actual):
                    return False
            elif actual != expected:
                return False

        return True

    # -- internal: persistence (JSON) ----------------------------------------

    def _to_dict(self) -> dict[str, Any]:
        """Serialize index state to a JSON-compatible dict."""
        # Convert inverted index: nested defaultdicts -> plain dicts
        index_data: dict[str, dict[str, dict[str, int]]] = {}
        for term, postings in self._index.items():
            index_data[term] = {}
            for doc_id, field_tfs in postings.items():
                index_data[term][doc_id] = dict(field_tfs)

        docs_data: dict[str, dict[str, Any]] = {}
        for doc_id, doc in self._docs.items():
            docs_data[doc_id] = asdict(doc)

        doc_terms_data: dict[str, list[str]] = {
            doc_id: sorted(terms) for doc_id, terms in self._doc_terms.items()
        }

        return {
            "version": _VERSION,
            "config": {
                "variant": self.variant,
                "k1": self.k1,
                "b": self.b,
                "delta": self.delta,
                "field_weights": self.field_weights,
                "calibrated": self.calibrated,
                "alpha": self._alpha,
                "beta": self._beta,
                "base_rate": self._base_rate,
            },
            "docs": docs_data,
            "index": index_data,
            "df": dict(self._df),
            "total_field_lengths": dict(self._total_field_lengths),
            "doc_terms": doc_terms_data,
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> SparseIndex:
        """Restore index state from a deserialized dict."""
        config = data["config"]
        instance = cls(
            variant=config["variant"],
            k1=config["k1"],
            b=config["b"],
            delta=config["delta"],
            field_weights=config.get("field_weights"),
            calibrated=config.get("calibrated", False),
        )
        instance._alpha = config.get("alpha")
        instance._beta = config.get("beta")
        instance._base_rate = config.get("base_rate")

        # Restore docs
        for doc_id, doc_data in data["docs"].items():
            instance._docs[doc_id] = _DocRecord(
                doc_id=doc_data["doc_id"],
                field_lengths=doc_data["field_lengths"],
                metadata=doc_data.get("metadata"),
            )

        # Restore inverted index as nested defaultdicts
        for term, postings in data["index"].items():
            for doc_id, field_tfs in postings.items():
                for field_name, tf in field_tfs.items():
                    instance._index[term][doc_id][field_name] = tf

        # Restore df
        for term, count in data["df"].items():
            instance._df[term] = count

        # Restore total field lengths
        for field_name, total in data["total_field_lengths"].items():
            instance._total_field_lengths[field_name] = total

        # Restore reverse index
        if "doc_terms" in data:
            for doc_id, terms in data["doc_terms"].items():
                instance._doc_terms[doc_id] = set(terms)
        else:
            # Rebuild from inverted index if loading old format
            for term, postings in instance._index.items():
                for doc_id in postings:
                    instance._doc_terms[doc_id].add(term)

        return instance

    def _save_json(self, path: str) -> None:
        data = self._to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def _load_json(cls, path: str) -> SparseIndex:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls._from_dict(data)

    # -- internal: persistence (SQLite) --------------------------------------

    def _save_sqlite(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            p.unlink()

        conn = sqlite3.connect(path)
        try:
            cur = conn.cursor()
            cur.executescript("""
                CREATE TABLE config (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE docs (
                    doc_id        TEXT PRIMARY KEY,
                    field_lengths TEXT NOT NULL,
                    metadata      TEXT
                );
                CREATE TABLE inverted_index (
                    term     TEXT NOT NULL,
                    doc_id   TEXT NOT NULL,
                    field_tfs TEXT NOT NULL,
                    PRIMARY KEY (term, doc_id)
                );
                CREATE TABLE df (
                    term  TEXT PRIMARY KEY,
                    count INTEGER NOT NULL
                );
                CREATE TABLE field_lengths_total (
                    field_name TEXT PRIMARY KEY,
                    total      REAL NOT NULL
                );
            """)

            # Config
            config = {
                "version": _VERSION,
                "variant": self.variant,
                "k1": self.k1,
                "b": self.b,
                "delta": self.delta,
                "field_weights": self.field_weights,
                "calibrated": self.calibrated,
                "alpha": self._alpha,
                "beta": self._beta,
                "base_rate": self._base_rate,
            }
            for key, value in config.items():
                cur.execute(
                    "INSERT INTO config (key, value) VALUES (?, ?)",
                    (key, json.dumps(value)),
                )

            # Docs
            for doc_id, doc in self._docs.items():
                cur.execute(
                    "INSERT INTO docs (doc_id, field_lengths, metadata)"
                    " VALUES (?, ?, ?)",
                    (
                        doc_id,
                        json.dumps(doc.field_lengths),
                        json.dumps(doc.metadata) if doc.metadata is not None else None,
                    ),
                )

            # Inverted index
            for term, postings in self._index.items():
                for doc_id, field_tfs in postings.items():
                    cur.execute(
                        "INSERT INTO inverted_index"
                        " (term, doc_id, field_tfs)"
                        " VALUES (?, ?, ?)",
                        (term, doc_id, json.dumps(dict(field_tfs))),
                    )

            # DF
            for term, count in self._df.items():
                cur.execute(
                    "INSERT INTO df (term, count) VALUES (?, ?)",
                    (term, count),
                )

            # Total field lengths
            for field_name, total in self._total_field_lengths.items():
                cur.execute(
                    "INSERT INTO field_lengths_total (field_name, total) VALUES (?, ?)",
                    (field_name, total),
                )

            conn.commit()
        finally:
            conn.close()

    @classmethod
    def _load_sqlite(cls, path: str) -> SparseIndex:
        conn = sqlite3.connect(path)
        try:
            cur = conn.cursor()

            # Load config
            config: dict[str, Any] = {}
            for key, value in cur.execute("SELECT key, value FROM config"):
                config[key] = json.loads(value)

            instance = cls(
                variant=config["variant"],
                k1=config["k1"],
                b=config["b"],
                delta=config["delta"],
                field_weights=config.get("field_weights"),
                calibrated=config.get("calibrated", False),
            )
            instance._alpha = config.get("alpha")
            instance._beta = config.get("beta")
            instance._base_rate = config.get("base_rate")

            # Load docs
            for doc_id, field_lengths_json, metadata_json in cur.execute(
                "SELECT doc_id, field_lengths, metadata FROM docs"
            ):
                instance._docs[doc_id] = _DocRecord(
                    doc_id=doc_id,
                    field_lengths=json.loads(field_lengths_json),
                    metadata=json.loads(metadata_json)
                    if metadata_json is not None
                    else None,
                )

            # Load inverted index and rebuild reverse index
            for term, doc_id, field_tfs_json in cur.execute(
                "SELECT term, doc_id, field_tfs FROM inverted_index"
            ):
                field_tfs = json.loads(field_tfs_json)
                for field_name, tf in field_tfs.items():
                    instance._index[term][doc_id][field_name] = tf
                instance._doc_terms[doc_id].add(term)

            # Load df
            for term, count in cur.execute("SELECT term, count FROM df"):
                instance._df[term] = count

            # Load total field lengths
            for field_name, total in cur.execute(
                "SELECT field_name, total FROM field_lengths_total"
            ):
                instance._total_field_lengths[field_name] = total

            return instance
        finally:
            conn.close()

    # -- internal: format resolution -----------------------------------------

    @staticmethod
    def _resolve_format(path: str, format: str | None) -> str:
        """Determine save format from explicit arg or file extension."""
        if format is not None:
            if format not in ("json", "sqlite"):
                raise ValueError(
                    f"Unknown format {format!r}, expected 'json' or 'sqlite'"
                )
            return format
        if Path(path).suffix in (".db", ".sqlite", ".sqlite3"):
            return "sqlite"
        return "json"


# ---------------------------------------------------------------------------
# Post-retrieval re-ranking utilities
# ---------------------------------------------------------------------------


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient between two sets.

    ``J(A, B) = |A ∩ B| / |A ∪ B|``

    Returns 0.0 when both sets are empty. This is a building-block for
    constructing similarity functions to pass to :func:`mmr`.

    Example::

        tokens = {}  # doc_id -> set of tokens
        sim_fn = lambda a, b: jaccard_similarity(tokens[a.doc_id], tokens[b.doc_id])
        diverse = mmr(results, sim_fn, lambda_=0.7)

    Args:
        set_a: First set.
        set_b: Second set.

    Returns:
        Similarity in [0.0, 1.0].
    """
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def rrf(
    *result_lists: list[Result],
    k: int = 60,
    top_k: int | None = None,
    weights: list[float] | None = None,
) -> list[Result]:
    """Reciprocal Rank Fusion of multiple ranked result lists.

    Combines results from different retrieval systems (e.g. BM25 sparse
    and dense vector search) using the formula::

        Score(d) = Σ weight_i / (k + rank_i)

    where *rank_i* is the 1-based position of document *d* in the *i*-th
    list (Cormack et al., SIGIR 2009).

    Example::

        sparse = index.search("quick fox", top_k=20)
        dense  = [Result("d1", 0.9), Result("d3", 0.7), ...]  # external
        fused  = rrf(sparse, dense, k=60, top_k=10)

    Args:
        *result_lists: One or more lists of :class:`Result` objects. Each
            list should be pre-sorted by descending relevance.
        k: RRF constant controlling rank sensitivity. Higher values flatten
            rank differences. Default 60 (per Cormack et al.).
        top_k: Maximum number of results to return. ``None`` returns all.
        weights: Per-list weights. When ``None``, all lists are weighted
            equally (1.0). Length must match the number of result lists.

    Returns:
        Fused list of :class:`Result` sorted by descending RRF score.
        Metadata is taken from the occurrence with the highest original
        score.

    Raises:
        ValueError: If no result lists are provided, ``weights`` length
            does not match, or ``k`` is not positive.
    """
    if not result_lists:
        raise ValueError("rrf() requires at least one result list")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if weights is not None and len(weights) != len(result_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match "
            f"number of result lists ({len(result_lists)})"
        )

    rrf_scores: dict[str, float] = {}
    best_meta: dict[str, dict[str, Any] | None] = {}
    best_original: dict[str, float] = {}

    for list_idx, results in enumerate(result_lists):
        w = weights[list_idx] if weights is not None else 1.0
        seen_in_list: set[str] = set()
        for rank_0, r in enumerate(results):
            if r.doc_id in seen_in_list:
                continue
            seen_in_list.add(r.doc_id)
            rrf_scores[r.doc_id] = rrf_scores.get(r.doc_id, 0.0) + w / (k + rank_0 + 1)
            if r.doc_id not in best_original or r.score > best_original[r.doc_id]:
                best_original[r.doc_id] = r.score
                best_meta[r.doc_id] = r.metadata

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]

    return [
        Result(doc_id=did, score=score, metadata=best_meta[did])
        for did, score in ranked
    ]


def mmr(
    results: list[Result],
    similarity_fn: Callable[[Result, Result], float],
    *,
    lambda_: float = 0.5,
    top_k: int | None = None,
) -> list[Result]:
    """Maximal Marginal Relevance re-ranking for result diversification.

    Greedily selects results that balance relevance with diversity::

        MMR(d) = argmax[λ · rel(d) − (1 − λ) · max sim(d, d_j)]

    where *d_j* are already-selected results. Relevance scores are min-max
    normalized to [0, 1] internally so they are comparable to similarity.

    Example::

        tokens = {r.doc_id: set(tokenize(texts[r.doc_id])) for r in results}
        sim = lambda a, b: jaccard_similarity(tokens[a.doc_id], tokens[b.doc_id])
        diverse = mmr(results, sim, lambda_=0.7, top_k=5)

    Args:
        results: Candidate results, typically from :meth:`SparseIndex.search`
            or :func:`rrf`.
        similarity_fn: A callable ``(Result, Result) -> float`` returning a
            similarity score in [0, 1]. See :func:`jaccard_similarity` for
            a set-based helper.
        lambda_: Trade-off between relevance (1.0) and diversity (0.0).
            Default 0.5.
        top_k: Number of results to select. ``None`` selects all (full
            re-ranking).

    Returns:
        Re-ranked list of :class:`Result` with scores set to their MMR
        scores.

    Raises:
        ValueError: If ``lambda_`` is not in [0, 1].
    """
    if not (0.0 <= lambda_ <= 1.0):
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")
    if not results:
        return []

    n = len(results)
    select_k = n if top_k is None else min(top_k, n)

    # Min-max normalize relevance scores to [0, 1]
    max_score = max(r.score for r in results)
    min_score = min(r.score for r in results)
    score_range = max_score - min_score
    if score_range > 0:
        norm = {r.doc_id: (r.score - min_score) / score_range for r in results}
    else:
        norm = {r.doc_id: (1.0 if max_score > 0 else 0.0) for r in results}

    candidates = list(results)
    selected: list[Result] = []

    while len(selected) < select_k and candidates:
        best_mmr = -math.inf
        best_idx = 0

        for idx, cand in enumerate(candidates):
            rel = norm[cand.doc_id]
            if selected:
                max_sim = max(similarity_fn(cand, s) for s in selected)
            else:
                max_sim = 0.0
            mmr_score = lambda_ * rel - (1.0 - lambda_) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        chosen = candidates.pop(best_idx)
        selected.append(
            Result(doc_id=chosen.doc_id, score=best_mmr, metadata=chosen.metadata)
        )

    return selected
