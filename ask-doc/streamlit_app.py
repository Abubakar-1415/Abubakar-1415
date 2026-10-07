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

import auth_store as auth

SECRET_ENV_KEYS = (
    "DATABASE_URL",
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

SESSION_DURATION_SECONDS = 4 * 60 * 60
SESSION_STARTED_AT_KEY = "_ask_doc_session_started_at"
USER_ID_KEY = "_ask_doc_user_id"
USERNAME_KEY = "_ask_doc_username"
logger = logging.getLogger("ask_doc.streamlit")


@st.cache_resource
def initialize_user_store() -> bool:
    auth.initialize_user_store(os.environ.get("DATABASE_URL", ""))
    return True


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
            uid = auth.create_user(
                os.environ.get("DATABASE_URL", ""), username, password
            )
            normalized_username = auth.normalize_username(username)
        else:
            account = auth.authenticate_user(
                os.environ.get("DATABASE_URL", ""), username, password
            )
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
    votes = feedback_votes(client, uid, question, list(best))
    for section_id, source in best.items():
        source["good_votes"], source["bad_votes"] = votes.get(section_id, (0, 0))
    sources = sorted(
        best.values(),
        key=lambda source: (
            source["good_votes"] - source["bad_votes"],
            source["score"],
        ),
        reverse=True,
    )[:6]
    for index, source in enumerate(sources, start=1):
        source["id"] = f"S{index}"
    return sources


def feedback_point_id(uid: int, question: str, section_id: int) -> str:
    query_hash = hashlib.sha256(rag.norm(question).encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{uid}:{query_hash}:{section_id}"))


def feedback_votes(client, uid: int, question: str, section_ids: list[int]) -> dict[int, tuple[int, int]]:
    if not section_ids:
        return {}
    point_ids = [feedback_point_id(uid, question, section_id) for section_id in section_ids]
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


def record_feedback(
    uid: int, question: str, citations: list[dict], helpful: bool
) -> None:
    client, _, _ = rag.rag_components()
    doc_names = {
        int(citation["section_id"]): str(citation["doc"])
        for citation in citations
    }
    section_ids = list(doc_names)
    point_ids = [feedback_point_id(uid, question, section_id) for section_id in section_ids]
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
    initialize_user_store()
except Exception:
    logger.exception("Ask Doc account database initialization failed")
    st.title("Ask Doc")
    st.error(
        "The account database is not configured or unavailable. Add a working "
        "PostgreSQL DATABASE_URL in the Streamlit app secrets."
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
                sources = retrieve(uid, question.strip())
                answer = rag.answer_from_sources(question.strip(), sources)
            st.session_state["ask_doc_search_result"] = {
                "question": question.strip(),
                "answer": answer,
            }
        except Exception as exc:
            st.session_state["ask_doc_search_error"] = rag.safe_error_detail(exc)

if error := st.session_state.get("ask_doc_search_error"):
    st.error(f"Search failed: {error}")

result = st.session_state.get("ask_doc_search_result")
if result:
    answer = result["answer"]
    citations = answer["citations"]
    with st.container(border=True):
        st.subheader("Answer (JSON)")
        st.json(answer, expanded=True)

        if citations and not answer["not_found"]:
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
