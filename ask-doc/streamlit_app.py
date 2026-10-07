"""Ask Doc hosted on Streamlit Community Cloud."""
from __future__ import annotations

import hashlib
import io
import os
import uuid

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

SECRET_ENV_KEYS = (
    "NVIDIA_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "NVIDIA_CHAT_MODEL",
    "NVIDIA_EMBEDDING_MODEL",
    "NVIDIA_EMBEDDING_DIMENSION",
    "NVIDIA_QDRANT_COLLECTION",
    "MAX_UPLOAD_MB",
    "MAX_CHUNKS_PER_FILE",
)


def load_secrets() -> None:
    try:
        for key in SECRET_ENV_KEYS:
            value = st.secrets.get(key)
            if value is not None:
                os.environ[key] = str(value)
    except StreamlitSecretNotFoundError:
        return


load_secrets()

import rag_app as rag


st.set_page_config(page_title="Ask Doc", page_icon=":material/search:", layout="wide")


def user_id_for(email: str) -> int:
    digest = hashlib.sha256(email.strip().lower().encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def require_login() -> tuple[str, int]:
    if not getattr(st.user, "is_logged_in", False):
        st.title("Ask Doc")
        st.write("Sign in with Google to access your private document library.")
        if st.button("Sign in with Google", type="primary"):
            st.login()
        st.stop()

    try:
        allowed_email = str(st.secrets["CLOUD_USER_EMAIL"]).strip().lower()
    except StreamlitSecretNotFoundError:
        allowed_email = ""
    email = str(getattr(st.user, "email", "") or "").strip().lower()
    if not allowed_email:
        st.error("Configure `CLOUD_USER_EMAIL` in app secrets before using the app.")
        st.stop()
    if not email or email != allowed_email:
        st.error("This Google account is not authorized to use this private app.")
        st.stop()
    return email, user_id_for(email)


def indexed_documents(uid: int) -> list[dict]:
    client, _, _ = rag.rag_components()
    results = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=rag.QDRANT_COLLECTION,
            scroll_filter=rag.models.Filter(must=[rag._match("user_id", uid)]),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        results.extend(points)
        if offset is None:
            break

    docs: dict[str, dict] = {}
    for point in results:
        payload = point.payload or {}
        name = str(payload.get("doc_name", ""))
        if not name:
            continue
        item = docs.setdefault(
            name,
            {"name": name, "kind": str(payload.get("kind", "")), "chunks": 0},
        )
        item["chunks"] += 1
    return sorted(docs.values(), key=lambda item: item["name"].casefold())


def ingest_file(uid: int, name: str, data: bytes) -> int:
    if not data:
        raise ValueError("The selected file is empty.")
    if len(data) > rag.MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File exceeds the {rag.MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    parsers = {"pdf": rag.parse_pdf, "docx": rag.parse_docx, "xlsx": rag.parse_xlsx}
    parser = parsers.get(ext)
    if parser is None:
        raise ValueError("Only PDF, DOCX, and XLSX files are supported.")

    sections = parser(io.BytesIO(data), name.rsplit(".", 1)[0])
    if not sections:
        raise ValueError("No readable text found. Scanned PDFs need OCR first.")

    client, embeddings, _ = rag.rag_components()
    batch_id = uuid.uuid4().hex
    section_ids = [
        int.from_bytes(
            hashlib.sha256(f"{uid}:{name}:{index}".encode("utf-8")).digest()[:8],
            "big",
        )
        & ((1 << 63) - 1)
        for index in range(len(sections))
    ]
    try:
        rag.index_document_sections(
            client, embeddings, uid, name, ext, sections, section_ids, batch_id
        )
    except Exception:
        try:
            rag.delete_vectors_for_batch(client, batch_id)
        except Exception:
            rag.logger.exception("Could not remove incomplete vectors for %s", name)
        raise

    try:
        rag.delete_vectors_except_batch(client, uid, name, batch_id)
    except Exception:
        rag.logger.warning(
            "Older vectors for %s were not removed; search ignores them by batch.",
            name,
            exc_info=True,
        )
    return len(sections)


def retrieve(uid: int, question: str) -> list[dict]:
    client, embeddings, _ = rag.rag_components()
    vector = embeddings.embed_query(question)
    response = client.query_points(
        collection_name=rag.QDRANT_COLLECTION,
        query=vector,
        query_filter=rag.models.Filter(must=[rag._match("user_id", uid)]),
        limit=24,
        with_payload=True,
    )
    best: dict[int, dict] = {}
    for point in response.points:
        payload = point.payload or {}
        section_id = int(payload.get("section_id", 0))
        if not section_id:
            continue
        score = max(0.0, min(1.0, float(point.score or 0.0)))
        if score < rag.MIN_VECTOR_SIMILARITY:
            continue
        source = {
            "section_id": section_id,
            "title": str(payload.get("title", "")),
            "doc": str(payload.get("doc_name", "")),
            "kind": str(payload.get("kind", "")),
            "location": str(payload.get("location", "")),
            "page_num": payload.get("page_num"),
            "excerpt": str(payload.get("text", "")),
            "score": score,
        }
        if section_id not in best or score > best[section_id]["score"]:
            best[section_id] = source
    sources = sorted(best.values(), key=lambda source: source["score"], reverse=True)[:6]
    for index, source in enumerate(sources, start=1):
        source["id"] = f"S{index}"
    return sources


def render_citations(citations: list[dict]) -> None:
    for citation in citations:
        location = citation.get("page_num") or citation.get("location") or ""
        st.markdown(f"**{citation['doc']}** — {citation['title']} · {location}")
        st.caption(citation["excerpt"])


email, uid = require_login()
st.title("Ask Doc")
st.caption(f"Signed in as {email}")
if st.button("Sign out"):
    st.logout()

with st.sidebar:
    st.subheader("Upload documents")
    uploads = st.file_uploader(
        "Choose PDF, Word, or Excel files",
        type=["pdf", "docx", "xlsx"],
        accept_multiple_files=True,
        max_upload_size=rag.MAX_UPLOAD_BYTES // (1024 * 1024),
    )
    if st.button("Upload selected files", type="primary", disabled=not uploads):
        if len(uploads) > rag.MAX_FILES_PER_UPLOAD:
            st.error(f"Upload at most {rag.MAX_FILES_PER_UPLOAD} files at a time.")
        for uploaded in uploads[:rag.MAX_FILES_PER_UPLOAD]:
            try:
                with st.spinner(f"Indexing {uploaded.name}..."):
                    section_count = ingest_file(uid, uploaded.name, uploaded.getvalue())
                st.success(f"{uploaded.name}: indexed {section_count} sections.")
            except Exception as exc:
                detail = rag.safe_error_detail(exc)
                st.error(f"{uploaded.name}: {type(exc).__name__}: {detail or 'Indexing failed.'}")

    st.subheader("Your documents")
    try:
        docs = indexed_documents(uid)
        if docs:
            for index, document in enumerate(docs):
                st.write(f"**{document['name']}**")
                st.caption(f"{document['kind'].upper()} · {document['chunks']} indexed chunks")
                if st.button("Delete", key=f"delete-{index}-{document['name']}"):
                    client, _, _ = rag.rag_components()
                    rag.delete_vectors_for_document(client, uid, document["name"])
                    st.rerun()
        else:
            st.caption("No documents uploaded yet.")
    except Exception as exc:
        st.error(f"Could not load documents: {rag.safe_error_detail(exc)}")

question = st.text_input(
    "Ask a question about your documents",
    max_chars=1200,
    placeholder="Ask a question about your documents...",
)
if st.button("Search documents", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Searching your documents and preparing an answer..."):
            sources = retrieve(uid, question.strip())
            answer = rag.answer_from_sources(question.strip(), sources)
        st.subheader("Answer")
        st.write(answer["answer"])
        if answer["citations"]:
            st.subheader("Sources")
            render_citations(answer["citations"])
        if answer.get("match_explanation"):
            st.caption(answer["match_explanation"])
    except Exception as exc:
        st.error(f"Search failed: {rag.safe_error_detail(exc)}")
