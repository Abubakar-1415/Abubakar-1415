"""Ask Doc hosted on Streamlit Community Cloud."""
from __future__ import annotations

import hashlib
import io
import logging
import os
import time
import uuid

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from qdrant_client import QdrantClient

import auth_store as auth
import search as document_search

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

SESSION_DURATION_SECONDS = auth.SESSION_DURATION_SECONDS
SESSION_STARTED_AT_KEY = "_ask_doc_session_started_at"
USER_ID_KEY = "_ask_doc_user_id"
USERNAME_KEY = "_ask_doc_username"
logger = logging.getLogger("ask_doc.streamlit")


@st.cache_resource
def initialize_user_store() -> QdrantClient:
    return auth.initialize_user_store(
        os.environ.get("QDRANT_URL", ""),
        os.environ.get("QDRANT_API_KEY", ""),
    )


def clear_authenticated_session() -> None:
    st.session_state.pop(USER_ID_KEY, None)
    st.session_state.pop(USERNAME_KEY, None)
    st.session_state.pop(SESSION_STARTED_AT_KEY, None)


def authenticate() -> tuple[int, str] | None:
    if not st.session_state.get(USER_ID_KEY):
        return None
    started_at = st.session_state.get(SESSION_STARTED_AT_KEY)
    if started_at is None or auth.session_expired(started_at, time.time()):
        clear_authenticated_session()
        st.session_state["_ask_doc_session_expired"] = True
        return None
    return int(st.session_state[USER_ID_KEY]), str(st.session_state[USERNAME_KEY])


def show_authentication() -> None:
    st.title("Ask Doc")
    st.caption("Create a private account to manage and search your documents.")
    if st.session_state.pop("_ask_doc_session_expired", False):
        st.warning("Your session expired. Log in again to continue.")

    mode = st.radio("Account", ["Log in", "Create account"], horizontal=True)
    with st.form("account-form"):
        username = st.text_input("Username", max_chars=32)
        password = st.text_input("Password", type="password", max_chars=1024)
        confirm_password = ""
        if mode == "Create account":
            confirm_password = st.text_input(
                "Confirm password", type="password", max_chars=1024
            )
        submitted = st.form_submit_button(mode, type="primary", width="stretch")

    if not submitted:
        return

    try:
        if mode == "Create account":
            if password != confirm_password:
                st.error("The passwords do not match.")
                return
            uid = auth.create_user(user_store, username, password)
            normalized_username = auth.normalize_username(username)
        else:
            account = auth.authenticate_user(user_store, username, password)
            if account is None:
                st.error("Invalid username or password.")
                return
            uid, normalized_username = account
    except auth.UsernameAlreadyExistsError:
        st.error("That username is already registered.")
        return
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception:
        logger.exception("Ask Doc account operation failed")
        st.error("The account operation failed. Check the database configuration.")
        return

    st.session_state[USER_ID_KEY] = uid
    st.session_state[USERNAME_KEY] = normalized_username
    st.session_state[SESSION_STARTED_AT_KEY] = time.time()
    st.rerun()


@st.fragment(run_every="1s")
def monitor_session_expiry() -> None:
    started_at = st.session_state.get(SESSION_STARTED_AT_KEY)
    if started_at is None:
        return
    now = time.time()
    remaining = SESSION_DURATION_SECONDS - (now - started_at)
    if auth.session_expired(started_at, now):
        clear_authenticated_session()
        st.session_state["_ask_doc_session_expired"] = True
        st.rerun()
    hours, remainder = divmod(int(remaining), 3600)
    minutes, seconds = divmod(remainder, 60)
    st.caption(f"Session expires in {hours:02d}:{minutes:02d}:{seconds:02d}")
    st.progress(max(0.0, min(1.0, remaining / SESSION_DURATION_SECONDS)))


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


def record_feedback(
    uid: int, question: str, citations: list[dict], helpful: bool
) -> None:
    client, _, _ = rag.rag_components()
    doc_names = {
        int(citation["section_id"]): str(citation["doc"])
        for citation in citations
    }
    section_ids = list(doc_names)
    point_ids = [
        document_search.feedback_point_id(uid, question, section_id)
        for section_id in section_ids
    ]
    existing = {
        int(point.payload.get("section_id", 0)): point
        for point in client.retrieve(
            collection_name=rag.QDRANT_FEEDBACK_COLLECTION,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        if point.payload
    }
    points = []
    for section_id, point_id in zip(section_ids, point_ids):
        payload = existing.get(section_id)
        good = int((payload.payload or {}).get("good", 0)) if payload else 0
        bad = int((payload.payload or {}).get("bad", 0)) if payload else 0
        if helpful:
            good += 1
        else:
            bad += 1
        points.append(
            rag.models.PointStruct(
                id=point_id,
                vector=[1.0],
                payload={
                    "user_id": uid,
                    "doc_name": doc_names[section_id],
                    "section_id": section_id,
                    "good": good,
                    "bad": bad,
                },
            )
        )
    client.upsert(
        collection_name=rag.QDRANT_FEEDBACK_COLLECTION,
        points=points,
        wait=True,
    )


try:
    user_store = initialize_user_store()
except Exception:
    logger.exception("Ask Doc account store initialization failed")
    st.title("Ask Doc")
    st.error(
        "The account store is not configured or unavailable. Check the Qdrant "
        "URL and API key in Streamlit app secrets."
    )
    st.stop()

authenticated_user = authenticate()
if authenticated_user is None:
    show_authentication()
    st.stop()

uid, username = authenticated_user
header_title, header_session, header_logout = st.columns(
    [3, 2, 1], vertical_alignment="center"
)
with header_title:
    st.title("Ask Doc")
    st.caption("Ask your knowledge. Find the answer.")
with header_session:
    st.caption(f"Private library for **{username}**")
    monitor_session_expiry()
with header_logout:
    if st.button("Log out", icon=":material/logout:", width="stretch"):
        clear_authenticated_session()
        st.rerun()

st.divider()

with st.sidebar:
    st.subheader("My private document library")
    st.caption(f"Logged in as {username}")
    st.divider()

    st.subheader("Upload your documents")
    uploads = st.file_uploader(
        "Drop PDF, Word, or Excel files here",
        type=["pdf", "docx", "xlsx"],
        accept_multiple_files=True,
        max_upload_size=rag.MAX_UPLOAD_BYTES // (1024 * 1024),
        help=f"Upload up to {rag.MAX_FILES_PER_UPLOAD} files, each up to "
        f"{rag.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
    )
    if st.button(
        "Upload selected files",
        type="primary",
        disabled=not uploads,
        icon=":material/upload_file:",
        width="stretch",
    ):
        if len(uploads) > rag.MAX_FILES_PER_UPLOAD:
            st.error(f"Upload at most {rag.MAX_FILES_PER_UPLOAD} files at a time.")
        for uploaded in uploads[:rag.MAX_FILES_PER_UPLOAD]:
            try:
                with st.spinner(f"Indexing {uploaded.name}..."):
                    section_count = ingest_file(uid, uploaded.name, uploaded.getvalue())
                st.success(f"{uploaded.name}: indexed {section_count} sections.")
            except Exception as exc:
                detail = rag.safe_error_detail(exc)
                st.error(
                    f"{uploaded.name}: {type(exc).__name__}: "
                    f"{detail or 'Indexing failed.'}"
                )

    st.divider()
    try:
        docs = indexed_documents(uid)
        st.subheader(f"Your documents · {len(docs)}")
        if docs:
            for index, document in enumerate(docs):
                with st.container(border=True):
                    st.markdown(f"**{document['name']}**")
                    st.caption(
                        f"{document['kind'].upper()} · "
                        f"{document['chunks']} indexed chunks"
                    )
                    with st.popover("Delete document"):
                        st.warning("This removes the document and its indexed content.")
                        if st.button(
                            "Confirm delete",
                            key=f"delete-{index}-{document['name']}",
                            type="primary",
                        ):
                            try:
                                client, _, _ = rag.rag_components()
                                rag.delete_feedback_for_document(
                                    client, uid, document["name"]
                                )
                                rag.delete_vectors_for_document(
                                    client, uid, document["name"]
                                )
                                st.rerun()
                            except Exception as exc:
                                st.error(
                                    f"Could not delete {document['name']}: "
                                    f"{rag.safe_error_detail(exc)}"
                                )
        else:
            st.info("Your library is empty. Upload a file to start searching.")
    except Exception as exc:
        st.error(f"Could not load documents: {rag.safe_error_detail(exc)}")

with st.container(border=True):
    st.subheader("Search your library")
    st.caption(
        "Answers are grounded in your uploaded documents and include source citations."
    )
    with st.form("document-search", clear_on_submit=False):
        question = st.text_input(
            "Ask a question about your documents",
            max_chars=1200,
            placeholder="What would you like to know?",
        )
        submitted = st.form_submit_button(
            "Search",
            type="primary",
            icon=":material/search:",
            width="stretch",
        )

if submitted:
    st.session_state.pop("ask_doc_search_result", None)
    st.session_state.pop("ask_doc_search_error", None)
    st.session_state.pop("ask_doc_feedback_message", None)
    if not question.strip():
        st.session_state["ask_doc_search_error"] = "Type a question."
    else:
        try:
            with st.spinner("Searching your documents and preparing an answer..."):
                evidence = document_search.retrieve(uid, question.strip())
                answer = None
                answer_error = ""
                if evidence["sources"]:
                    try:
                        answer = rag.answer_from_sources(
                            question.strip(), evidence["sources"]
                        )
                    except Exception as exc:
                        answer_error = rag.safe_error_detail(exc)
                        logger.exception("AI answer generation failed after retrieval")
            st.session_state["ask_doc_search_result"] = {
                "question": question.strip(),
                "answer": answer,
                "answer_error": answer_error,
                "evidence": evidence,
            }
        except Exception as exc:
            st.session_state["ask_doc_search_error"] = rag.safe_error_detail(exc)

if error := st.session_state.get("ask_doc_search_error"):
    st.error(f"Search failed: {error}")

result = st.session_state.get("ask_doc_search_result")
if result:
    answer = result["answer"]
    evidence = result["evidence"]
    citations = answer["citations"] if answer else []
    with st.container(border=True):
        st.subheader("Answer")
        if answer and not answer["not_found"]:
            st.markdown(answer["answer"])
            st.caption(
                f"Relevance estimate: {answer['match_percent']}% "
                "(not a calibrated probability of correctness)"
            )
        elif result["answer_error"]:
            st.warning(
                "I found matching document passages, but the AI summary failed. "
                "The exact matches are listed below."
            )
            st.caption(f"Answer service: {result['answer_error']}")
        elif answer:
            st.info(answer["answer"])
        else:
            st.info("No strong matches were found. Try an exact name or keyword.")
        if evidence["semantic_error"]:
            st.caption(
                "Semantic search was unavailable; exact keyword search was still "
                f"attempted. {evidence['semantic_error']}"
            )

        if citations:
            st.markdown("#### Sources")
            for index, citation in enumerate(citations, start=1):
                location = citation.get("page_num") or citation.get("location") or ""
                source_label = (
                    f"{index}. {citation['doc']} · {citation['title']}"
                )
                if location:
                    source_label += f" · {location}"
                with st.expander(source_label):
                    st.caption(citation["kind"].upper())
                    st.write(citation["excerpt"])

        matches = evidence["keyword_matches"]
        if matches:
            st.markdown(
                f"#### Exact keyword matches · {evidence['keyword_match_count']} "
                f"passages in {evidence['keyword_match_documents']} documents"
            )
            if evidence["lexical_error"]:
                st.warning(
                    f"Keyword scan stopped after {evidence['scanned_chunks']} "
                    f"passages: {evidence['lexical_error']}"
                )
            else:
                st.caption(
                    f"Checked all {evidence['scanned_chunks']} indexed passages. "
                    "Each match shows the exact keyword with its surrounding text."
                )
            for match_index, match in enumerate(matches, start=1):
                location = match.get("page_num") or match.get("location") or ""
                label = f"{match_index}. {match['doc']}"
                if location:
                    label += f" · {location}"
                with st.expander(label):
                    st.caption(
                        f"Matched terms: {', '.join(match['matched_keywords']) or 'document title'}"
                    )
                    for context in match["keyword_contexts"]:
                        st.write(
                            f"{context['prefix']} **{context['keyword']}** "
                            f"{context['suffix']}"
                        )
                    if not match["keyword_contexts"]:
                        st.write(match["excerpt"])
            if evidence["keyword_match_count"] > len(matches):
                st.caption(
                    f"Showing the top {len(matches)} of "
                    f"{evidence['keyword_match_count']} matching passages."
                )
        elif evidence["lexical_error"]:
            st.warning(
                "Full-library exact-keyword search failed: "
                f"{evidence['lexical_error']}"
            )
        elif evidence["query_terms"]:
            st.caption(
                f"Checked {evidence['scanned_chunks']} indexed passages; "
                "no exact keyword matches found."
            )

        if answer:
            with st.expander("View answer as JSON"):
                st.json(answer, expanded=True)

        if citations and answer and not answer["not_found"]:
            st.divider()
            st.markdown("**Was this answer useful?**")
            feedback_message = st.session_state.get("ask_doc_feedback_message")
            if feedback_message:
                st.success(feedback_message)
            else:
                yes_col, no_col = st.columns(2)
                with yes_col:
                    if st.button(
                        "Yes, helpful",
                        key="feedback-yes",
                        icon=":material/thumb_up:",
                        width="stretch",
                    ):
                        try:
                            record_feedback(
                                uid,
                                result["question"],
                                citations,
                                True,
                            )
                            st.session_state["ask_doc_feedback_message"] = (
                                "Thanks. Feedback saved for this question."
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not save feedback: {rag.safe_error_detail(exc)}")
                with no_col:
                    if st.button(
                        "No, not helpful",
                        key="feedback-no",
                        icon=":material/thumb_down:",
                        width="stretch",
                    ):
                        try:
                            record_feedback(
                                uid,
                                result["question"],
                                citations,
                                False,
                            )
                            st.session_state["ask_doc_feedback_message"] = (
                                "Thanks. Feedback saved for this question."
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not save feedback: {rag.safe_error_detail(exc)}")
elif not st.session_state.get("ask_doc_search_error"):
    st.info("Ask a question to search your private document library.")
