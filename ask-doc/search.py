"""Hybrid semantic and exact-keyword search for private Ask Doc libraries."""
from __future__ import annotations

import hashlib
import logging
import re
import uuid

import rag_app as rag

logger = logging.getLogger("ask_doc.search")
QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "about", "can", "could", "did", "do", "does",
    "for", "from", "give", "how", "i", "in", "is", "it", "me", "of", "on",
    "please", "show", "tell", "that", "the", "their", "them", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "with", "would",
    "you", "your",
}
MAX_ANSWER_SOURCES = 18
MAX_SOURCES_PER_DOCUMENT = 2
MAX_KEYWORD_MATCHES = 80


def query_terms(question: str) -> list[str]:
    return list(dict.fromkeys(
        token
        for token in re.findall(r"[a-z0-9]+", question.casefold())
        if len(token) > 1 and token not in QUERY_STOP_WORDS
    ))


def exact_keyword_contexts(
    text: str, terms: list[str], radius: int = 100, per_term_limit: int = 3
) -> list[dict]:
    contexts = []
    for term in terms:
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        for match in list(pattern.finditer(text))[:per_term_limit]:
            start, end = match.span()
            prefix = text[max(0, start - radius):start].strip()
            suffix = text[end:min(len(text), end + radius)].strip()
            contexts.append({
                "prefix": f"...{prefix}" if start > radius else prefix,
                "keyword": match.group(0),
                "suffix": f"{suffix}..." if len(text) - end > radius else suffix,
            })
    return contexts


def source_from_point(point, semantic_score: float = 0.0) -> dict | None:
    payload = point.payload or {}
    section_id = int(payload.get("section_id", 0))
    if not section_id:
        return None
    return {
        "point_id": str(point.id),
        "section_id": section_id,
        "title": str(payload.get("title", "")),
        "doc": str(payload.get("doc_name", "")),
        "kind": str(payload.get("kind", "")),
        "location": str(payload.get("location", "")),
        "page_num": payload.get("page_num"),
        "excerpt": str(payload.get("text", "")),
        "score": semantic_score,
        "lexical_score": 0,
        "matched_keywords": [],
        "keyword_contexts": [],
    }


def select_answer_sources(candidates: list[dict]) -> list[dict]:
    ranked = sorted(
        candidates,
        key=lambda source: (
            source["lexical_score"],
            source["score"],
            source.get("good_votes", 0) - source.get("bad_votes", 0),
        ),
        reverse=True,
    )
    by_document: dict[str, list[dict]] = {}
    for source in ranked:
        by_document.setdefault(source["doc"], []).append(source)

    selected = []
    for rank in range(MAX_SOURCES_PER_DOCUMENT):
        for document_sources in by_document.values():
            if rank < len(document_sources):
                selected.append(document_sources[rank])
                if len(selected) == MAX_ANSWER_SOURCES:
                    return selected
    return selected


def feedback_point_id(uid: int, question: str, section_id: int) -> str:
    query_hash = hashlib.sha256(rag.norm(question).encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{uid}:{query_hash}:{section_id}"))


def feedback_votes(
    client, uid: int, question: str, section_ids: list[int]
) -> dict[int, tuple[int, int]]:
    if not section_ids:
        return {}
    point_ids = [
        feedback_point_id(uid, question, section_id) for section_id in section_ids
    ]
    points = client.retrieve(
        collection_name=rag.QDRANT_FEEDBACK_COLLECTION,
        ids=point_ids,
        with_payload=True,
        with_vectors=False,
    )
    votes = {}
    for point in points:
        payload = point.payload or {}
        section_id = int(payload.get("section_id", 0))
        if section_id:
            votes[section_id] = (
                int(payload.get("good", 0)),
                int(payload.get("bad", 0)),
            )
    return votes


def retrieve(uid: int, question: str) -> dict:
    client, embeddings, _ = rag.rag_components()
    terms = query_terms(question)
    user_filter = rag.models.Filter(must=[rag._match("user_id", uid)])
    candidates: dict[str, dict] = {}
    scanned_chunks = 0
    semantic_error = ""
    lexical_error = ""
    normalized_query = " ".join(re.findall(r"[a-z0-9]+", question.casefold()))

    try:
        vector = embeddings.embed_query(question)
        response = client.query_points(
            collection_name=rag.QDRANT_COLLECTION,
            query=vector,
            query_filter=user_filter,
            limit=48,
            with_payload=True,
        )
        for point in response.points:
            score = max(0.0, min(1.0, float(point.score or 0.0)))
            if score < rag.MIN_VECTOR_SIMILARITY:
                continue
            source = source_from_point(point, score)
            if source:
                candidates[source["point_id"]] = source
    except Exception as exc:
        semantic_error = rag.safe_error_detail(exc)
        logger.exception("Semantic document search failed; trying keyword search")

    keyword_matches = []
    offset = None
    try:
        while True:
            points, offset = client.scroll(
                collection_name=rag.QDRANT_COLLECTION,
                scroll_filter=user_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                scanned_chunks += 1
                source = source_from_point(point)
                if source is None:
                    continue
                text = source["excerpt"]
                searchable_name = re.sub(
                    r"[^a-z0-9]+",
                    " ",
                    " ".join(
                        (source["doc"], source["title"], source["location"])
                    ).casefold(),
                )
                normalized_text = re.sub(r"[^a-z0-9]+", " ", text.casefold())
                matching_text = [
                    term for term in terms
                    if re.search(
                        rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
                        normalized_text,
                    )
                ]
                matching_name = [
                    term for term in terms
                    if re.search(
                        rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
                        searchable_name,
                    )
                ]
                if not matching_text and not matching_name:
                    continue

                source["matched_keywords"] = matching_text
                source["keyword_contexts"] = exact_keyword_contexts(
                    text, matching_text
                )
                source["lexical_score"] = (
                    len(matching_text) * 3
                    + len(matching_name) * 2
                    + (4 if normalized_query in normalized_text else 0)
                )
                existing = candidates.get(source["point_id"])
                if existing:
                    source["score"] = existing["score"]
                candidates[source["point_id"]] = source
                keyword_matches.append(source)
            if offset is None:
                break
    except Exception as exc:
        lexical_error = rag.safe_error_detail(exc)
        logger.exception("Full-library keyword search failed")

    keyword_matches.sort(
        key=lambda source: (source["lexical_score"], source["score"]),
        reverse=True,
    )
    sources = select_answer_sources(list(candidates.values()))
    section_ids = list({source["section_id"] for source in sources})
    try:
        votes = feedback_votes(client, uid, question, section_ids)
    except Exception:
        logger.exception("Could not load feedback votes; continuing without them")
        votes = {}
    for source in candidates.values():
        source["good_votes"], source["bad_votes"] = votes.get(
            source["section_id"], (0, 0)
        )
    for index, source in enumerate(sources, start=1):
        source["id"] = f"S{index}"
    return {
        "sources": sources,
        "keyword_matches": keyword_matches[:MAX_KEYWORD_MATCHES],
        "keyword_match_count": len(keyword_matches),
        "keyword_match_documents": len({source["doc"] for source in keyword_matches}),
        "scanned_chunks": scanned_chunks,
        "query_terms": terms,
        "semantic_error": semantic_error,
        "lexical_error": lexical_error,
    }
