from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from campus_agent.models import SearchHit
from campus_agent.rag.index import IndexData, normalise_vector
from campus_agent.rag.tokenizer import (
    core_phrase,
    query_terms,
    semantic_group_coverage,
    token_counts,
)


_BM25_K1 = 1.5
_BM25_B = 0.75
_VECTOR_WEIGHT = 0.68
_BM25_WEIGHT = 0.22
_COVERAGE_WEIGHT = 0.10
_MIN_LEXICAL_COVERAGE = 0.24
_MIN_COMPOUND_CONCEPT_COVERAGE = 0.75


class QueryEmbeddings(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(first * second for first, second in zip(left, right, strict=True))


class HybridRetriever:
    """Combine cosine similarity with BM25 and exact-term coverage."""

    def __init__(self, index: IndexData, embeddings: QueryEmbeddings) -> None:
        self.index = index
        self.embeddings = embeddings
        self._document_texts: list[str] = []
        self._document_terms: list[dict[str, int]] = []
        self._document_lengths: list[int] = []
        document_frequencies: Counter[str] = Counter()
        for item in index.chunks:
            chunk = item.chunk
            searchable_text = (
                f"{chunk.title}\n{chunk.title}\n{chunk.section}\n{chunk.text}"
            )
            counts = token_counts(searchable_text)
            self._document_texts.append(searchable_text)
            self._document_terms.append(counts)
            self._document_lengths.append(sum(counts.values()))
            # BM25 needs the number of documents containing a term, not the
            # total number of times that term occurs across the corpus.
            document_frequencies.update(counts.keys())
        self._document_frequencies = dict(document_frequencies)
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 1.0
        )

    def _bm25_score(
        self,
        query_weights: dict[str, float],
        document_terms: dict[str, int],
        document_length: int,
    ) -> float:
        document_count = len(self._document_terms)
        if not document_count:
            return 0.0
        length_ratio = document_length / max(self._average_document_length, 1.0)
        score = 0.0
        for term, query_weight in query_weights.items():
            frequency = document_terms.get(term, 0)
            if not frequency:
                continue
            document_frequency = self._document_frequencies.get(term, 0)
            inverse_frequency = math.log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequency + _BM25_K1 * (
                1.0 - _BM25_B + _BM25_B * length_ratio
            )
            score += (
                query_weight
                * inverse_frequency
                * frequency
                * (_BM25_K1 + 1.0)
                / denominator
            )
        return score

    @staticmethod
    def _lexical_profile(
        query: str,
        document_text: str,
        document_terms: dict[str, int],
        primary_terms: dict[str, float],
        expanded_terms: dict[str, float],
    ) -> tuple[float, list[str], bool, float, int]:
        primary_total = sum(primary_terms.values()) or 1.0
        primary_matched = sum(
            weight
            for term, weight in primary_terms.items()
            if document_terms.get(term, 0)
        )
        primary_coverage = primary_matched / primary_total
        group_coverage, group_count = semantic_group_coverage(query, document_text)
        coverage = max(
            primary_coverage,
            group_coverage * 0.85 if group_count else 0.0,
        )
        matched = {
            term
            for term in (*primary_terms, *expanded_terms)
            if len(term) >= 2 and document_terms.get(term, 0)
        }
        ordered_matches = sorted(
            matched,
            key=lambda term: (
                -(primary_terms.get(term, expanded_terms.get(term, 0.0))),
                -len(term),
                term,
            ),
        )
        has_specific_signal = len(matched) >= 2 or any(
            len(term) >= 3 for term in matched
        )
        return (
            min(1.0, coverage),
            ordered_matches[:12],
            has_specific_signal,
            group_coverage,
            group_count,
        )

    def search(self, query: str, *, top_k: int = 4, min_score: float = 0.35) -> list[SearchHit]:
        query = query.strip()
        if not query or not self.index.chunks:
            return []
        query_vector = normalise_vector(
            self.embeddings.embed_query(query),
            expected_dimension=self.index.embedding_dimension,
        )
        lexical_query = core_phrase(query) or query
        primary_terms, expanded_terms = query_terms(lexical_query)
        query_weights = dict(primary_terms)
        for term, weight in expanded_terms.items():
            query_weights[term] = max(query_weights.get(term, 0.0), weight)

        candidates: list[tuple[float, float, float, list[str], bool, bool, object]] = []
        for item, document_text, document_terms, document_length in zip(
            self.index.chunks,
            self._document_texts,
            self._document_terms,
            self._document_lengths,
            strict=True,
        ):
            similarity = max(-1.0, min(1.0, _dot(query_vector, item.vector)))
            bm25_score = self._bm25_score(
                query_weights, document_terms, document_length
            )
            (
                coverage,
                matched_terms,
                has_specific_signal,
                group_coverage,
                group_count,
            ) = self._lexical_profile(
                query,
                document_text,
                document_terms,
                primary_terms,
                expanded_terms,
            )
            # When a question explicitly combines multiple known concepts
            # (for example "图书馆" + "校外访问"), matching only one concept
            # must not be treated as sufficient lexical evidence.
            concepts_satisfied = (
                group_count < 2
                or group_coverage >= _MIN_COMPOUND_CONCEPT_COVERAGE
            )
            candidates.append(
                (
                    similarity,
                    bm25_score,
                    coverage,
                    matched_terms,
                    has_specific_signal,
                    concepts_satisfied,
                    item,
                )
            )

        max_bm25 = max((candidate[1] for candidate in candidates), default=0.0)
        scored: list[SearchHit] = []
        for (
            similarity,
            raw_bm25,
            coverage,
            matched_terms,
            has_specific_signal,
            concepts_satisfied,
            item,
        ) in candidates:
            normalised_bm25 = raw_bm25 / max_bm25 if max_bm25 > 0 else 0.0
            vector_match = similarity >= min_score
            lexical_match = (
                raw_bm25 > 0
                and coverage >= _MIN_LEXICAL_COVERAGE
                and has_specific_signal
                and concepts_satisfied
            )
            if not vector_match and not lexical_match:
                continue
            fused_score = max(
                similarity,
                _VECTOR_WEIGHT * max(0.0, similarity)
                + _BM25_WEIGHT * normalised_bm25
                + _COVERAGE_WEIGHT * coverage,
            )
            scored.append(
                SearchHit(
                    chunk=item.chunk,
                    score=max(0.0, min(1.0, fused_score)),
                    bm25_score=normalised_bm25,
                    coverage=coverage,
                    matched_terms=matched_terms,
                )
            )
        scored.sort(
            key=lambda hit: (hit.score, hit.bm25_score, hit.coverage), reverse=True
        )
        if not scored:
            return []

        # Do not let weak tail matches enter the context merely because they
        # clear the absolute threshold. The source cap and near-duplicate
        # filter keep one long document from crowding out other evidence.
        relative_floor = scored[0].score * 0.72
        selected: list[SearchHit] = []
        seen_grams: list[set[str]] = []
        per_source: dict[str, int] = {}
        limit = max(1, int(top_k))
        for hit in scored:
            if hit.score < relative_floor:
                break
            source_count = per_source.get(hit.chunk.source_id, 0)
            if source_count >= 2:
                continue
            grams = set(token_counts(hit.chunk.text))
            if any(_jaccard(grams, previous) > 0.86 for previous in seen_grams):
                continue
            selected.append(hit)
            seen_grams.append(grams)
            per_source[hit.chunk.source_id] = source_count + 1
            if len(selected) >= limit:
                break
        return selected


# Keep the old import name available for callers outside the service module.
VectorRetriever = HybridRetriever
