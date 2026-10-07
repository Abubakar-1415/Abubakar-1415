"""
Ask Doc - AI document question answering.

Each user signs in with their own password, uploads PDF/Word/Excel files, and
searches their private library. New accounts require owner approval. PDF results
include page numbers; Word and Excel results cite their section or sheet.

Run locally:  python rag_app.py     (Python 3.10+)
NVIDIA package: pip install -U langchain-nvidia-ai-endpoints
Required env: ADMIN_USERNAME, ADMIN_PASSWORD, QDRANT_URL, QDRANT_API_KEY, NVIDIA_API_KEY
              (or set NVIDIA_API_KEY in api_key.py)
Optional env: OWNER_EMAIL + SMTP_HOST (+ SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, MAIL_FROM)
              for new-request emails, HOST, PORT, DB_PATH, ALLOW_REGISTER (1/0),
              COOKIE_SECURE (1 behind HTTPS), MAX_UPLOAD_MB, SEARCH_LIMIT_PER_DAY,
              MIN_VECTOR_SIMILARITY, NVIDIA_CHAT_MODEL, NVIDIA_EMBEDDING_MODEL,
              NVIDIA_QDRANT_COLLECTION
Scanned PDFs need OCR before upload.
"""
import hashlib
import hmac
import io
import json
import logging
import os
import re
import secrets
import smtplib
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from email.message import EmailMessage
from pathlib import Path

import pymupdf
import uvicorn
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse
from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from api_key import get_nvidia_api_key
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openpyxl import load_workbook
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ask_doc")

BASE = Path(__file__).resolve().parent
DB_PATH = os.environ.get("DB_PATH", str(BASE / "AskDoc.db"))
OWNER_EMAIL = os.environ.get("OWNER_EMAIL", "")
ALLOW_REGISTER = os.environ.get("ALLOW_REGISTER", "1") == "1"
SESSION_LIMIT = 4 * 3600
MIN_VECTOR_SIMILARITY = float(os.environ.get("MIN_VECTOR_SIMILARITY", "0.3"))
NVIDIA_CHAT_MODEL = os.environ.get("NVIDIA_CHAT_MODEL", "nvidia/nemotron-3-super-120b-a12b")
_RETIRED_EMBEDDING_MODELS = {
    "nvidia/nv-embedqa-e5-v5",
    "nvidia/llama-nemotron-embed-1b-v2",
}
_NEMOTRON_EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"
_configured_embedding_model = os.environ.get("NVIDIA_EMBEDDING_MODEL", "").strip()
QDRANT_FEEDBACK_COLLECTION = "ask_doc_feedback_v1"
_using_current_embedding_model = (
    not _configured_embedding_model
    or _configured_embedding_model in _RETIRED_EMBEDDING_MODELS
    or _configured_embedding_model == _NEMOTRON_EMBEDDING_MODEL
)
if _using_current_embedding_model:
    NVIDIA_EMBEDDING_MODEL = _NEMOTRON_EMBEDDING_MODEL
    NVIDIA_EMBEDDING_DIMENSION = 2048
    QDRANT_COLLECTION = (
        f"ask_doc_documents_nemotron_3_embed_1b_{NVIDIA_EMBEDDING_DIMENSION}")
else:
    NVIDIA_EMBEDDING_MODEL = _configured_embedding_model
    NVIDIA_EMBEDDING_DIMENSION = int(os.environ.get("NVIDIA_EMBEDDING_DIMENSION", "1024"))
    QDRANT_COLLECTION = os.environ.get(
        "NVIDIA_QDRANT_COLLECTION",
        f"ask_doc_documents_{NVIDIA_EMBEDDING_MODEL.replace('/', '_')}_"
        f"{NVIDIA_EMBEDDING_DIMENSION}")
MAX_UPLOAD_BYTES = max(100, int(os.environ.get("MAX_UPLOAD_MB", "100"))) * 1024 * 1024
MAX_FILES_PER_UPLOAD = 10
MAX_CHUNKS_PER_FILE = int(os.environ.get("MAX_CHUNKS_PER_FILE", "1500"))
MAX_SHEET_ROWS = 5000
SEARCH_LIMIT_PER_DAY = int(os.environ.get("SEARCH_LIMIT_PER_DAY", "150"))

_qdrant: QdrantClient | None = None
_embeddings: NVIDIAEmbeddings | None = None
_llm: ChatNVIDIA | None = None
_vector_store_ready = False

SCHEMA = """
create table if not exists users(id integer primary key, username text unique not null,
  pw text not null, created real not null, role text not null default 'user',
  status text not null default 'pending');
create table if not exists sessions(id integer primary key,
  user_id integer not null references users(id) on delete cascade,
  token text unique not null, started real not null, last_seen real not null, ended real);
create table if not exists documents(id integer primary key autoincrement,
  user_id integer not null references users(id) on delete cascade,
  name text not null, kind text not null, uploaded real not null, upload_id text);
create table if not exists sections(id integer primary key autoincrement,
  doc_id integer not null references documents(id) on delete cascade,
  user_id integer not null, title text not null, content text not null,
  body text not null, codes text not null, preview text not null,
  page_num integer, location text not null default '');
create table if not exists memory(user_id integer not null, query text not null,
  section_id integer not null references sections(id) on delete cascade,
  good integer not null default 0, bad integer not null default 0,
  primary key(user_id, query, section_id));
create table if not exists searches(user_id integer not null, ts real not null);
create index if not exists ix_sec_user on sections(user_id);
create index if not exists ix_search_user on searches(user_id, ts);
"""


class UploadError(Exception):
    """A problem with one uploaded file that the user can understand and fix."""


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH, timeout=15)
    c.row_factory = sqlite3.Row
    c.execute("pragma foreign_keys=on")
    c.execute("pragma journal_mode=wal")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def ensure_column(c, table: str, column: str, declaration: str):
    columns = {r["name"] for r in c.execute(f"pragma table_info({table})")}
    if column not in columns:
        c.execute(f"alter table {table} add column {column} {declaration}")


def owner_configured() -> bool:
    return bool(os.environ.get("ADMIN_USERNAME") and os.environ.get("ADMIN_PASSWORD"))


def notify_owner(subject: str, body: str):
    """Best-effort email. Never blocks or fails the request if mail is not set up."""
    host = os.environ.get("SMTP_HOST")
    if not (host and OWNER_EMAIL):
        return
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = os.environ.get("MAIL_FROM", OWNER_EMAIL)
        msg["To"] = OWNER_EMAIL
        msg.set_content(body)
        with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587")), timeout=20) as smtp:
            smtp.starttls()
            username = os.environ.get("SMTP_USERNAME")
            if username:
                smtp.login(username, os.environ.get("SMTP_PASSWORD", ""))
            smtp.send_message(msg)
    except (OSError, smtplib.SMTPException, ValueError) as exc:
        logger.warning("Owner email could not be sent: %s", exc)


# ---------------------------------------------------------------- RAG services

class GeneratedAnswer(BaseModel):
    answer: str = Field(description="A concise answer supported only by the provided sources.")
    source_ids: list[str] = Field(description="Source labels such as S1 that directly support the answer.")
    not_found: bool = Field(description="True when the sources do not contain enough evidence to answer.")


def safe_error_detail(exc: Exception) -> str:
    detail = re.sub(r"\s+", " ", str(exc)).strip()
    detail = re.sub(r"\bnvapi-[A-Za-z0-9_-]+\b", "[redacted]", detail)
    detail = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [redacted]", detail)
    for secret in (get_nvidia_api_key(), os.environ.get("QDRANT_API_KEY", "")):
        if secret:
            detail = detail.replace(secret, "[redacted]")
    return detail[:240]


def rag_components():
    global _qdrant, _embeddings, _llm, _vector_store_ready
    api_key = get_nvidia_api_key()
    missing = []
    if not api_key:
        missing.append("NVIDIA_API_KEY (set in the environment or api_key.py)")
    missing.extend(k for k in ("QDRANT_URL", "QDRANT_API_KEY") if not os.environ.get(k))
    if missing:
        raise HTTPException(
            503, "RAG is not configured. Set these values: " + ", ".join(missing))
    try:
        if _embeddings is None:
            _embeddings = NVIDIAEmbeddings(
                model=NVIDIA_EMBEDDING_MODEL, nvidia_api_key=api_key,
                truncate="END")
        if _llm is None:
            _llm = ChatNVIDIA(
                model=NVIDIA_CHAT_MODEL, temperature=0, nvidia_api_key=api_key,
                timeout=60)
        if _qdrant is None:
            _qdrant = QdrantClient(
                url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=30)
        if not _vector_store_ready:
            existing = {c.name for c in _qdrant.get_collections().collections}
            if QDRANT_COLLECTION not in existing:
                _qdrant.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=models.VectorParams(
                        size=NVIDIA_EMBEDDING_DIMENSION, distance=models.Distance.COSINE))
            if QDRANT_FEEDBACK_COLLECTION not in existing:
                _qdrant.create_collection(
                    collection_name=QDRANT_FEEDBACK_COLLECTION,
                    vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE))
            for field, schema in (("user_id", models.PayloadSchemaType.INTEGER),
                                  ("doc_name", models.PayloadSchemaType.KEYWORD)):
                _qdrant.create_payload_index(QDRANT_FEEDBACK_COLLECTION, field, schema)
            vector_config = _qdrant.get_collection(QDRANT_COLLECTION).config.params.vectors
            vector_size = getattr(vector_config, "size", None)
            if vector_size != NVIDIA_EMBEDDING_DIMENSION:
                raise RuntimeError(
                    f"Qdrant collection {QDRANT_COLLECTION!r} has vector size "
                    f"{vector_size}; the configured embedding model requires "
                    f"{NVIDIA_EMBEDDING_DIMENSION}. Configure a new collection name.")
            # Indexes for every field used in filters (safe to repeat on each start)
            for field, schema in (("user_id", models.PayloadSchemaType.INTEGER),
                                  ("doc_name", models.PayloadSchemaType.KEYWORD),
                                  ("section_id", models.PayloadSchemaType.INTEGER),
                                  ("batch", models.PayloadSchemaType.KEYWORD)):
                _qdrant.create_payload_index(QDRANT_COLLECTION, field, schema)
            _vector_store_ready = True
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Could not initialize NVIDIA/Qdrant services")
        raise HTTPException(
            503, "Could not initialize NVIDIA or Qdrant. Check service credentials and status. "
                 f"Details: {safe_error_detail(exc)}") from exc
    return _qdrant, _embeddings, _llm


def _match(key: str, value) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def delete_vectors_for_document(client: QdrantClient, user_id: int, name: str):
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[_match("user_id", user_id), _match("doc_name", name)])),
        wait=True)


def delete_feedback_for_document(client: QdrantClient, user_id: int, name: str):
    client.delete(
        collection_name=QDRANT_FEEDBACK_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[_match("user_id", user_id), _match("doc_name", name)])),
        wait=True)


def delete_vectors_for_batch(client: QdrantClient, upload_id: str):
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.FilterSelector(filter=models.Filter(must=[_match("batch", upload_id)])),
        wait=True)


def delete_vectors_except_batch(client: QdrantClient, user_id: int, name: str, upload_id: str):
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.FilterSelector(filter=models.Filter(
            must=[_match("user_id", user_id), _match("doc_name", name)],
            must_not=[_match("batch", upload_id)])),
        wait=True)


def rag_text(blocks: list) -> str:
    """Text used for embeddings. Table rows keep their column headers."""
    out = []
    for b in blocks:
        if b["t"] == "p":
            out.append(b["text"])
            continue
        rows = b["rows"]
        if len(rows) > 1:
            head = rows[0]
            for r in rows[1:]:
                cells = [f"{h}: {v}" if h else v for h, v in zip(head, r) if v]
                if cells:
                    out.append(" | ".join(cells))
        elif rows:
            out.append(" | ".join(v for v in rows[0] if v))
    return "\n".join(out).strip()


def index_document_sections(client, embeddings, user_id: int, name: str, kind: str,
                            sections: list[dict], section_ids: list[int], upload_id: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=140)
    chunks: list[LCDocument] = []
    for section, section_id in zip(sections, section_ids):
        for chunk_index, text_chunk in enumerate(splitter.split_text(rag_text(section["blocks"]))):
            chunks.append(LCDocument(page_content=text_chunk, metadata={
                "user_id": user_id, "doc_name": name, "kind": kind, "section_id": section_id,
                "title": section["title"],
                "location": section.get("location") or section["title"],
                "page_num": section.get("page_num"), "chunk_index": chunk_index,
                "batch": upload_id}))
    if not chunks:
        raise UploadError("no readable text found")
    if len(chunks) > MAX_CHUNKS_PER_FILE:
        raise UploadError(f"too large to index ({len(chunks)} parts; the limit is {MAX_CHUNKS_PER_FILE})")
    for offset in range(0, len(chunks), 64):
        group = chunks[offset:offset + 64]
        embedding_texts = [
            f"title: {item.metadata.get('title') or 'none'} | text: {item.page_content}"
            for item in group
        ]
        vectors = embeddings.embed_documents(embedding_texts)
        points = [models.PointStruct(id=str(uuid.uuid4()), vector=vector,
                                     payload={**item.metadata, "text": item.page_content})
                  for item, vector in zip(group, vectors)]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)


def retrieve_sources(user_id: int, query: str, limit: int = 6) -> list[dict]:
    client, embeddings, _ = rag_components()
    try:
        # NVIDIA's embedding integration sends input_type='query' automatically here.
        vector = embeddings.embed_query(query)
        result = client.query_points(
            collection_name=QDRANT_COLLECTION, query=vector,
            query_filter=models.Filter(must=[_match("user_id", user_id)]),
            limit=limit, with_payload=True)
    except Exception as exc:
        logger.exception("Qdrant retrieval or NVIDIA embedding failed")
        raise HTTPException(
            503, "Document retrieval failed. Check NVIDIA API usage and Qdrant connectivity.") from exc

    points = list(result.points)
    ids = sorted({int((p.payload or {}).get("section_id", 0)) for p in points})
    valid: dict[int, str | None] = {}
    memory: dict[int, tuple[int, int]] = {}
    if ids:
        marks = ",".join("?" * len(ids))
        with conn() as db:
            for r in db.execute(
                    f"select s.id, d.upload_id from sections s join documents d on d.id=s.doc_id "
                    f"where s.user_id=? and s.id in ({marks})", (user_id, *ids)):
                valid[r["id"]] = r["upload_id"]
            for r in db.execute(
                    f"select section_id, good, bad from memory where user_id=? and query=? "
                    f"and section_id in ({marks})", (user_id, norm(query), *ids)):
                memory[r["section_id"]] = (r["good"], r["bad"])

    best: dict[int, dict] = {}
    for point in points:
        payload = point.payload or {}
        similarity = max(0.0, min(1.0, float(point.score or 0.0)))
        section_id = int(payload.get("section_id", 0))
        # Ignore stale vectors from deleted or replaced uploads
        if section_id not in valid or payload.get("batch") != valid[section_id]:
            continue
        if similarity < MIN_VECTOR_SIMILARITY:
            continue
        good, bad = memory.get(section_id, (0, 0))
        source = {
            "section_id": section_id, "title": str(payload.get("title", "")),
            "doc": str(payload.get("doc_name", "")), "kind": str(payload.get("kind", "")),
            "location": str(payload.get("location", "")), "page_num": payload.get("page_num"),
            "excerpt": str(payload.get("text", "")), "score": similarity,
            "good_votes": good, "bad_votes": bad,
        }
        if section_id not in best or similarity > best[section_id]["score"]:
            best[section_id] = source

    def rank(s):  # sources the user confirmed for this exact question come first
        good, bad = memory.get(s["section_id"], (0, 0))
        return (good - bad, s["score"])

    ordered = sorted(best.values(), key=rank, reverse=True)
    for index, source in enumerate(ordered, start=1):
        source["id"] = f"S{index}"
    return ordered


def answer_from_sources(query: str, sources: list[dict]) -> dict:
    if not sources:
        return {"answer": "I couldn't find enough supporting information in your uploaded documents.",
                "citations": [], "match_percent": 0, "not_found": True,
                "match_explanation": "No source met the minimum relevance threshold."}
    _, _, llm = rag_components()
    context = "\n\n".join(
        f"[{s['id']}] {s['doc']} — {s['title']} — {s['location']}\nEvidence: {s['excerpt'][:1800]}"
        for s in sources)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the user's question only from the supplied document evidence. "
         "Do not use outside knowledge, guess, or fill gaps. If the evidence does not "
         "directly support an answer, set not_found=true and explain that the documents "
         "do not say. For every factual answer, list only source labels that directly "
         "support it. Return the requested structured JSON."),
        ("human", "Question:\n{question}\n\nDocument evidence:\n{context}"),
    ])
    try:
        generated = llm.with_structured_output(GeneratedAnswer).invoke(
            prompt.format_messages(question=query, context=context))
    except Exception as exc:
        logger.exception("NVIDIA answer generation failed")
        raise HTTPException(
            502, "The answer service failed. Check the NVIDIA API key, quota and model settings.") from exc
    by_id = {s["id"]: s for s in sources}
    cited = list({sid: by_id[sid] for sid in generated.source_ids if sid in by_id}.values())
    if generated.not_found or not generated.answer.strip() or not cited:
        return {"answer": "I couldn't verify an answer from the retrieved document excerpts.",
                "citations": [], "match_percent": 0, "not_found": True,
                "match_explanation": "No cited source supports a confident answer."}
    citations = [{
        "section_id": s["section_id"], "title": s["title"], "doc": s["doc"], "kind": s["kind"],
        "location": s["location"], "page_num": s["page_num"],
        "excerpt": citation_excerpt(s["excerpt"], query)} for s in cited]
    vector_match = sum(s["score"] for s in cited) / len(cited) * 100
    feedback_adjustment = 5 * sum(
        s.get("good_votes", 0) - s.get("bad_votes", 0) for s in cited
    ) / len(cited)
    confidence = max(0, min(100, round(vector_match + feedback_adjustment)))
    return {
        "answer": generated.answer.strip(), "citations": citations,
        "match_percent": confidence,
        "not_found": False,
        "match_explanation": "Vector similarity adjusted by your Yes/No feedback for this question; not a calibrated probability of correctness.",
    }


def citation_excerpt(text: str, query: str) -> str:
    query_norm = norm(query)
    query_tokens = set(toks(query_norm))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if query_norm:
        exact = [line for line in lines if query_norm in norm(line)]
        if exact:
            return "\n".join(exact[:2])
    matches = [(len(query_tokens & set(toks(line))), line) for line in lines]
    matches = [(score, line) for score, line in matches if score]
    if matches:
        best = max(score for score, _ in matches)
        return "\n".join(line for score, line in matches if score == best)[:700]
    return text[:350].strip()


# ---------------------------------------------------------------- auth

def hash_pw(p: str) -> str:
    salt = os.urandom(16)
    h = hashlib.scrypt(p.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return salt.hex() + ":" + h.hex()


def check_pw(p: str, stored: str) -> bool:
    try:
        s, h = stored.split(":")
        t = hashlib.scrypt(p.encode(), salt=bytes.fromhex(s), n=2**14, r=8, p=1, dklen=32)
        return hmac.compare_digest(t.hex(), h)
    except Exception:
        return False


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


FAILS: dict = {}


def throttle(user: str):
    recent = [t for t in FAILS.get(user, []) if time.time() - t < 300]
    FAILS[user] = recent
    if len(recent) >= 5:
        raise HTTPException(429, "Too many wrong attempts. Wait 5 minutes and try again.")


def auth(request: Request) -> dict:
    tok = request.cookies.get("sid")
    if tok:
        with conn() as c:
            r = c.execute(
                "select s.id, s.user_id, s.started, s.last_seen, u.username, u.role, u.status "
                "from sessions s join users u on u.id = s.user_id "
                "where s.token=? and s.ended is null", (sha(tok),)).fetchone()
            if r:
                now = time.time()
                if now - r["started"] >= SESSION_LIMIT or r["status"] != "approved":
                    c.execute("update sessions set ended=? where id=?",
                              (min(now, r["started"] + SESSION_LIMIT), r["id"]))
                else:
                    c.execute("update sessions set last_seen=? where id=?", (now, r["id"]))
                    return {"id": r["user_id"], "name": r["username"], "sid": r["id"],
                            "role": r["role"], "started": r["started"],
                            "remaining": max(0, int(r["started"] + SESSION_LIMIT - now))}
    raise HTTPException(401, "Please sign in.")


def start_session(uid: int, resp: Response) -> dict:
    tok = secrets.token_urlsafe(32)
    now = time.time()
    with conn() as c:
        c.execute("update sessions set ended=last_seen where user_id=? and ended is null", (uid,))
        c.execute("insert into sessions(user_id, token, started, last_seen) values(?,?,?,?)",
                  (uid, sha(tok), now, now))
    resp.set_cookie("sid", tok, httponly=True, samesite="lax", max_age=SESSION_LIMIT,
                    secure=os.environ.get("COOKIE_SECURE", "0") == "1")
    return {"ok": True}


def day_start() -> float:
    t = time.localtime()
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))


def usage_hours(uid: int) -> float:
    start = day_start()
    total = 0.0
    with conn() as c:
        for r in c.execute("select started, ended, last_seen from sessions "
                           "where user_id=? and coalesce(ended, last_seen) >= ?", (uid, start)):
            end = r["ended"] or min(r["last_seen"], r["started"] + SESSION_LIMIT)
            total += max(0.0, end - max(r["started"], start))
    return round(total / 3600, 1)


def use_search_quota(uid: int):
    with conn() as c:
        used = c.execute("select count(*) n from searches where user_id=? and ts>=?",
                         (uid, day_start())).fetchone()["n"]
        if used >= SEARCH_LIMIT_PER_DAY:
            raise HTTPException(429, f"Daily search limit reached ({SEARCH_LIMIT_PER_DAY}). Try again tomorrow.")
        c.execute("insert into searches(user_id, ts) values(?,?)", (uid, time.time()))


# ---------------------------------------------------------------- parsing

def norm(s: str) -> str:
    return " ".join(s.lower().split())


def toks(s: str) -> list:
    return [w.rstrip("s") if len(w) > 3 else w for w in re.findall(r"[a-z0-9]+", s.lower())]


def parse_docx(stream, fallback: str) -> list:
    d = Document(stream)
    secs, cur = [], None

    def start(title):
        nonlocal cur
        cur = {"title": title, "blocks": [], "page_num": None, "location": title}
        secs.append(cur)

    for el in d.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(el, d)
            txt = p.text.strip()
            if not txt:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            if style.lower().startswith(("heading", "title")):
                start(txt)
            else:
                if cur is None:
                    start(fallback)
                cur["blocks"].append({"t": "p", "text": txt})
        elif tag == "tbl":
            if cur is None:
                start(fallback)
            rows = [[c.text.strip() for c in r.cells] for r in Table(el, d).rows]
            if rows:
                cur["blocks"].append({"t": "table", "rows": rows})
    return [s for s in secs if s["blocks"]]


def parse_xlsx(stream, fallback: str) -> list:
    wb = load_workbook(stream, data_only=True, read_only=True)
    secs = []
    for ws in wb.worksheets:
        rows = []
        for r in ws.iter_rows(values_only=True):
            r = ["" if v is None else str(v).strip() for v in r]
            if any(r):
                rows.append(r)
            if len(rows) >= MAX_SHEET_ROWS:
                break
        if not rows:
            continue
        width = max(max((i for i, v in enumerate(r) if v), default=0) for r in rows) + 1
        rows = [(r + [""] * width)[:width] for r in rows]
        secs.append({"title": ws.title, "blocks": [{"t": "table", "rows": rows}],
                     "page_num": None, "location": f"Sheet {ws.title}"})
    return secs


def parse_pdf(stream, fallback: str) -> list:
    doc = pymupdf.open(stream=stream.read(), filetype="pdf")
    secs = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        secs.append({"title": (lines[0][:160] if lines else fallback) or fallback,
                     "blocks": [{"t": "p", "text": text}],
                     "page_num": page_num, "location": f"Page {page_num}"})
    doc.close()
    return secs


def flatten(blocks: list) -> list:
    out = []
    for b in blocks:
        if b["t"] == "p":
            out.append(b["text"])
        else:
            out += [" ".join(v for v in r if v) for r in b["rows"]]
    return out


def make_codes(title: str, stem: str, blocks: list) -> str:
    cs = {norm(title), norm(stem)}
    cs.update(w.lower() for w in re.findall(r"\b[A-Z][A-Z0-9&/-]{1,11}\b", title))
    for b in blocks:
        if b["t"] == "table":
            for r in b["rows"]:
                for v in r:
                    if 0 < len(v) <= 24:
                        cs.add(norm(v))
            if len(cs) > 4000:
                break
    return "\n".join(c for c in cs if c)


def save_doc(uid: int, name: str, kind: str, secs: list) -> int:
    """
    Store a document: save rows (short DB transaction), index vectors (slow, no DB lock held),
    then retire any older copy of the same file. A failed upload never removes the old copy.
    """
    client, embeddings, _ = rag_components()
    upload_id = uuid.uuid4().hex
    stem = name.rsplit(".", 1)[0]
    with conn() as c:
        old_ids = [r["id"] for r in c.execute(
            "select id from documents where user_id=? and name=?", (uid, name))]
        did = c.execute(
            "insert into documents(user_id, name, kind, uploaded, upload_id) values(?,?,?,?,?)",
            (uid, name, kind, time.time(), upload_id)).lastrowid
        section_ids = []
        for s in secs:
            lines = flatten(s["blocks"])
            section_ids.append(c.execute(
                "insert into sections(doc_id, user_id, title, content, body, codes, preview, "
                "page_num, location) values(?,?,?,?,?,?,?,?,?)",
                (did, uid, s["title"], json.dumps(s["blocks"]), " ".join(lines).lower()[:20000],
                 make_codes(s["title"], stem, s["blocks"]), (lines[0] if lines else "")[:140],
                 s.get("page_num"), s.get("location") or s["title"])).lastrowid)
    try:
        index_document_sections(client, embeddings, uid, name, kind, secs, section_ids, upload_id)
    except Exception:
        with conn() as c:
            c.execute("delete from documents where id=?", (did,))
        try:
            delete_vectors_for_batch(client, upload_id)
        except Exception:
            logger.warning("Could not remove partial vectors for %s", name, exc_info=True)
        raise
    with conn() as c:
        for old in old_ids:
            c.execute("delete from documents where id=?", (old,))
    try:
        delete_vectors_except_batch(client, uid, name, upload_id)
    except Exception:
        logger.warning("Stale vectors for %s could not be removed; they are ignored in search.", name,
                       exc_info=True)
    return len(section_ids)


# ---------------------------------------------------------------- API

class Cred(BaseModel):
    username: str
    password: str


class Fb(BaseModel):
    query: str = Field(min_length=1, max_length=1200)
    section_ids: list[int] = Field(min_length=1, max_length=10)
    helpful: bool


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1200)


class Decision(BaseModel):
    status: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    with conn() as c:
        c.executescript(SCHEMA)
        ensure_column(c, "users", "role", "text not null default 'user'")
        ensure_column(c, "users", "status", "text not null default 'approved'")
        ensure_column(c, "sections", "page_num", "integer")
        ensure_column(c, "sections", "location", "text not null default ''")
        ensure_column(c, "documents", "upload_id", "text")
        c.execute("update sections set location=title where location=''")
        admin_username = os.environ.get("ADMIN_USERNAME", "").strip().lower()
        admin_password = os.environ.get("ADMIN_PASSWORD", "")
        if admin_username and admin_password:
            c.execute(
                "insert into users(username,pw,created,role,status) values(?,?,?,'admin','approved') "
                "on conflict(username) do update set pw=excluded.pw, role='admin', status='approved'",
                (admin_username, hash_pw(admin_password), time.time()))
        else:
            logger.warning("ADMIN_USERNAME / ADMIN_PASSWORD are not set: nobody can approve accounts.")
    yield


app = FastAPI(title="Ask Doc", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)


@app.get("/health")
def health():
    return {"status": "ok", "app": "Ask Doc"}


@app.get("/api/config")
def config():
    return {"register": ALLOW_REGISTER and owner_configured()}


@app.post("/api/register")
def register(b: Cred):
    if not ALLOW_REGISTER:
        raise HTTPException(403, "Sign-up is turned off. Ask your admin for an account.")
    if not owner_configured():
        raise HTTPException(503, "Account approval is not configured. Contact the application owner.")
    u = b.username.strip().lower()
    if not re.fullmatch(r"[a-z0-9._-]{3,32}", u):
        raise HTTPException(400, "Username: 3-32 letters, numbers, dot, dash or underscore.")
    if len(b.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    try:
        with conn() as c:
            c.execute("insert into users(username, pw, created, status) values(?,?,?,'pending')",
                      (u, hash_pw(b.password), time.time()))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "That username is taken.")
    notify_owner("New Ask Doc account request",
                 f"User '{u}' requested an account. Sign in as the owner to approve or reject it.")
    return {"pending": True}


@app.post("/api/login")
def login(b: Cred, resp: Response):
    u = b.username.strip().lower()
    throttle(u)
    with conn() as c:
        r = c.execute("select id, pw, role, status from users where username=?", (u,)).fetchone()
    if not r or not check_pw(b.password, r["pw"]):
        FAILS.setdefault(u, []).append(time.time())
        raise HTTPException(401, "Wrong username or password.")
    if r["status"] == "pending":
        raise HTTPException(403, "Your account is waiting for owner approval.")
    if r["status"] != "approved":
        raise HTTPException(403, "This account was not approved. Contact the application owner.")
    FAILS.pop(u, None)
    return start_session(r["id"], resp)


@app.post("/api/logout")
def logout(resp: Response, user=Depends(auth)):
    with conn() as c:
        c.execute("update sessions set ended=? where id=?", (time.time(), user["sid"]))
    resp.delete_cookie("sid")
    return {"ok": True}


@app.post("/api/heartbeat")
def heartbeat(user=Depends(auth)):
    return {"hours": usage_hours(user["id"]), "remaining": user["remaining"]}


@app.get("/api/me")
def me(user=Depends(auth)):
    with conn() as c:
        files = [dict(r) for r in c.execute(
            "select d.id, d.name, d.kind, d.uploaded, "
            "(select count(*) from sections where doc_id=d.id) sections "
            "from documents d where d.user_id=? order by d.uploaded desc", (user["id"],))]
    return {"username": user["name"], "hours": usage_hours(user["id"]), "files": files,
            "role": user["role"], "remaining": user["remaining"]}


def require_admin(user=Depends(auth)):
    if user["role"] != "admin":
        raise HTTPException(403, "Owner access is required.")
    return user


@app.get("/api/admin/requests")
def admin_requests(_user=Depends(require_admin)):
    with conn() as c:
        rows = [dict(r) for r in c.execute(
            "select id, username, created from users where role='user' and status='pending' "
            "order by created")]
    return {"requests": rows}


@app.post("/api/admin/users/{user_id}/decision")
def admin_decide(user_id: int, decision: Decision, _user=Depends(require_admin)):
    if decision.status not in ("approved", "rejected"):
        raise HTTPException(400, "Decision must be approved or rejected.")
    with conn() as c:
        done = c.execute(
            "update users set status=? where id=? and role='user' and status='pending'",
            (decision.status, user_id)).rowcount
    if not done:
        raise HTTPException(404, "That pending account was not found.")
    return {"ok": True, "status": decision.status}


@app.post("/api/upload")
def upload(files: list[UploadFile] = File(...), user=Depends(auth)):
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(400, f"Upload at most {MAX_FILES_PER_UPLOAD} files at a time.")
    rag_components()  # fail early if NVIDIA/Qdrant are not configured
    done, errors = [], []
    for f in files:
        name = Path(f.filename or "file").name
        ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
        if ext not in ("pdf", "docx", "xlsx"):
            errors.append(f"{name}: only .pdf, .docx and .xlsx files are supported")
            continue
        data = f.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            errors.append(f"{name}: larger than {MAX_UPLOAD_BYTES // 1048576} MB")
            continue
        try:
            parser = {"pdf": parse_pdf, "docx": parse_docx, "xlsx": parse_xlsx}[ext]
            secs = parser(io.BytesIO(data), name.rsplit(".", 1)[0])
        except Exception as e:
            logger.exception("Could not parse %s for user %s", name, user["id"])
            errors.append(f"{name}: could not be read ({type(e).__name__})")
            continue
        if not secs:
            errors.append(f"{name}: no readable text found (scanned PDFs need OCR first)")
            continue
        try:
            count = save_doc(user["id"], name, ext, secs)
        except UploadError as e:
            errors.append(f"{name}: {e}")
            continue
        except HTTPException as e:
            errors.append(f"{name}: {e.detail}")
            continue
        except Exception as e:
            error_type = type(e).__name__.lower()
            error_text = str(e).lower()
            detail = re.sub(r"\s+", " ", str(e)).strip()
            detail = re.sub(r"\bnvapi-[A-Za-z0-9_-]+\b", "[redacted]", detail)
            detail = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [redacted]", detail)
            for secret in (get_nvidia_api_key(), os.environ.get("QDRANT_API_KEY", "")):
                if secret:
                    detail = detail.replace(secret, "[redacted]")
            detail = detail[:240]
            status_code = getattr(e, "status_code", getattr(e, "code", None))
            if ("ratelimit" in error_type or status_code == 429
                    or "resource_exhausted" in error_text or "429" in error_text):
                reason = "NVIDIA rate limit or quota reached. Wait and retry, or review NVIDIA API usage limits."
            elif status_code == 410 or "[410] gone" in error_text or "end of life" in error_text:
                reason = (
                    "The configured NVIDIA embedding model is retired or unavailable. "
                    "Set NVIDIA_EMBEDDING_MODEL to nvidia/nemotron-3-embed-1b and restart the app.")
            elif ("authentication" in error_type or "unauthenticated" in error_text
                  or status_code in (401, 403) or "401" in error_text or "403" in error_text):
                reason = "NVIDIA or Qdrant rejected the request. Check the API keys, account limits, and service status."
            else:
                logger.exception("Could not store/index %s for user %s", name, user["id"])
                reason = f"could not be stored/indexed ({type(e).__name__})"
            if detail:
                reason += f" Details: {detail}"
            errors.append(f"{name}: {reason}")
            continue
        done.append({"name": name, "sections": count})
    return {"done": done, "errors": errors}


@app.delete("/api/files/{doc_id}")
def delete_file(doc_id: int, user=Depends(auth)):
    with conn() as c:
        doc = c.execute("select name from documents where id=? and user_id=?",
                        (doc_id, user["id"])).fetchone()
        if not doc:
            raise HTTPException(404, "That file was not found.")
        c.execute("delete from documents where id=? and user_id=?", (doc_id, user["id"]))
        same_name_left = c.execute("select 1 from documents where user_id=? and name=?",
                                   (user["id"], doc["name"])).fetchone()
    if not same_name_left:
        try:  # best effort: leftovers are ignored by search anyway
            client, _, _ = rag_components()
            delete_vectors_for_document(client, user["id"], doc["name"])
        except Exception:
            logger.warning("Could not remove vectors for %s", doc["name"], exc_info=True)
    return {"ok": True}


@app.post("/api/search")
def search(b: Question, user=Depends(auth)):
    query = b.question.strip()
    if not query:
        raise HTTPException(400, "Type a question.")
    use_search_quota(user["id"])
    sources = retrieve_sources(user["id"], query)
    return answer_from_sources(query, sources)


@app.post("/api/feedback")
def feedback(b: Fb, user=Depends(auth)):
    qn = norm(b.query)
    if not qn:
        raise HTTPException(400, "Missing search text.")
    ids = sorted(set(b.section_ids))
    marks = ",".join("?" * len(ids))
    with conn() as c:
        owned = {r["id"] for r in c.execute(
            f"select id from sections where user_id=? and id in ({marks})", (user["id"], *ids))}
        for sid in ids:
            if sid not in owned:
                continue
            c.execute(
                "insert into memory(user_id, query, section_id, good, bad) values(?,?,?,?,?) "
                "on conflict(user_id, query, section_id) do update set "
                "good = good + excluded.good, bad = bad + excluded.bad",
                (user["id"], qn, sid, int(b.helpful), int(not b.helpful)))
    return {"ok": True, "score_delta": 5 if b.helpful else -5}


# ---------------------------------------------------------------- UI

INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ask Doc</title>
<style>
:root{--ink:#152635;--mut:#61717D;--line:#DCE5EA;--bg:#F4F7F8;--acc:#087F73;--soft:#E3F4F1;--bad:#B42318;--shadow:0 12px 32px rgba(26,48,61,.055)}
*{box-sizing:border-box}
body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--bg)}
.i{width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;flex:none}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;min-height:44px;padding:0 16px;border:1px solid #C5CFD6;border-radius:10px;background:#fff;color:var(--ink);font:500 14px inherit;font-family:inherit;cursor:pointer}
.btn:hover{background:#EEF2F5}.btn.pri{background:var(--acc);border-color:var(--acc);color:#fff}.btn.pri:hover{background:#0B5E57}
.btn:disabled{opacity:.5;cursor:not-allowed}
.ib{width:44px;padding:0;border:0;background:transparent}
.card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:22px 24px;margin-bottom:16px;box-shadow:var(--shadow)}
.muted,.meta{color:var(--mut)}.meta{font-size:13px;margin-top:2px}
h2{margin:0;font-size:18px;font-weight:600}
.top{height:72px;padding:0 max(28px,calc((100vw - 1440px)/2));display:flex;align-items:center;justify-content:space-between;background:#fff;border-bottom:1px solid var(--line)}
.top-actions{display:flex;align-items:center;justify-content:flex-end;gap:12px}
.brand-copy{display:flex;flex-direction:column;gap:1px}
.brand-copy small{font-size:11px;font-weight:400;color:var(--mut)}
.identity{display:flex;align-items:center;gap:8px;font-size:14px}
.role-badge{display:inline-flex;align-items:center;min-height:25px;padding:0 10px;border-radius:999px;background:#EEF2F5;color:#405564;font-size:12px;font-weight:600}
.role-badge.owner{background:#E3F4F1;color:#0B5E57}
.session-pill{font-variant-numeric:tabular-nums;white-space:nowrap}
.logo{display:flex;align-items:center;gap:12px;font-size:18px;font-weight:600}
.logo b{width:36px;height:36px;border-radius:10px;background:var(--acc);color:#fff;display:flex;align-items:center;justify-content:center}
.pill{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border-radius:99px;background:var(--soft);color:#0B5E57;font-size:13px;font-weight:500}
.pill:before{content:"";width:8px;height:8px;border-radius:50%;background:var(--acc)}
.pill.plain:before{display:none}
.grid{display:grid;grid-template-columns:310px minmax(0,1fr);gap:26px;padding:28px max(28px,calc((100vw - 1440px)/2));align-items:start;max-width:1500px;margin:0 auto}
.side{background:#fff;border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:var(--shadow)}
.prof{display:flex;align-items:center;gap:12px;cursor:pointer;padding-bottom:14px;border-bottom:1px solid #E6ECF0;background:none;border-left:0;border-right:0;border-top:0;width:100%;font:inherit;color:inherit;text-align:left}
.av{width:44px;height:44px;border-radius:50%;background:var(--ink);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:600;text-transform:uppercase}
.row{display:flex;align-items:center;gap:10px;min-height:48px;font-size:14px;font-weight:500}
.row .r{margin-left:auto;font-family:'IBM Plex Mono',monospace}
.drop{border:1.5px dashed #9FB0BC;border-radius:10px;padding:14px;text-align:center;font-size:13px;color:var(--mut);cursor:pointer}
.drop.over{border-color:var(--acc);background:var(--soft)}.drop u{color:var(--acc);font-weight:500}
.bar{height:6px;border-radius:3px;background:#E6ECF0;margin-bottom:10px}.bar i{display:block;height:6px;border-radius:3px;background:var(--acc)}
.file{display:flex;align-items:center;gap:10px;min-height:56px;border-top:1px solid #E6ECF0;font-size:13px}
.file .fi{flex:1;min-width:0}.file b{display:block;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.file small{color:var(--mut)}
.tag{font:500 10px 'IBM Plex Mono',monospace;padding:3px 6px;border-radius:5px}.docx{background:#DCE8F8;color:#1B4A8A}.xlsx{background:#DDF0DF;color:#1E6B2E}.pdf{background:#FBE5E2;color:#9A332B}
.sbox{display:flex;align-items:center;gap:12px;background:#fff;border:1px solid var(--line);border-radius:14px;padding:8px 8px 8px 18px;margin-bottom:16px}
.sbox input{flex:1;border:0;outline:0;font:500 20px 'IBM Plex Mono',monospace;color:var(--ink);background:transparent;min-height:44px;min-width:0}
.ch{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:14px}
.card p{margin:0 0 10px;line-height:1.65;font-size:15px;white-space:pre-wrap}
.link{background:none;border:0;color:var(--acc);font:500 14px inherit;font-family:inherit;cursor:pointer;padding:6px 0;margin-bottom:10px;min-height:44px}
.fb{display:flex;align-items:center;gap:14px;flex-wrap:wrap}.fb .t{flex:1;min-width:220px}.fb b{font-size:15px}
.ok{color:#0B5E57;font-weight:500}.err{color:var(--bad);font-size:13px;min-height:18px}
.login{max-width:420px;margin:12vh auto;padding:0 20px}.login label{display:block;font-size:13px;font-weight:500;margin:14px 0 6px}
.login input{width:100%;min-height:44px;border:1px solid #C5CFD6;border-radius:10px;padding:0 12px;font:inherit}
.empty{padding:32px;text-align:center;color:var(--mut);border:1px dashed var(--line);border-radius:12px}
.admin-actions{display:flex;gap:8px}.admin-actions .btn{min-height:38px}
.answer-text{font-size:16px;line-height:1.7;white-space:pre-wrap}
.match-score{display:inline-flex;align-items:center;padding:7px 11px;border-radius:999px;background:var(--soft);color:#0B5E57;font-size:13px;font-weight:600;white-space:nowrap}
.citation{border-top:1px solid var(--line);padding:14px 0}
.citation:first-child{border-top:0}
.citation h3{margin:0 0 4px;font-size:14px}
.citation p{margin:0;color:var(--mut);font-size:13px}
.citation blockquote{margin:8px 0 0;padding:10px 12px;border-left:3px solid var(--acc);background:#F6FBFA;font-size:13px;line-height:1.5;white-space:pre-wrap}
@media(max-width:860px){.grid{grid-template-columns:1fr;padding:16px}.top{padding:0 16px}}
@media(max-width:600px){.top{height:auto;min-height:68px;padding:10px 12px;gap:8px}.top-actions{gap:7px;flex-wrap:wrap}.session-pill{font-size:11px;padding:6px 8px}.identity{font-size:12px;gap:5px}.role-badge{font-size:11px;min-height:22px;padding:0 7px}.top .logo{font-size:15px;gap:8px}.top .logo b{width:32px;height:32px}}
</style></head>
<body><div id="root"></div>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const IC={
search:'<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
power:'<path d="M12 3v9"/><path d="M6.4 6.6a8 8 0 1 0 11.2 0"/>',
up:'<path d="M12 16V4"/><path d="M7 9l5-5 5 5"/><path d="M4 20h16"/>',
clock:'<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
folder:'<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
chev:'<path d="M6 9l6 6 6-6"/>',
trash:'<path d="M4 7h16"/><path d="M10 11v6M14 11v6"/><path d="M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12"/><path d="M9 7V4h6v3"/>',
copy:'<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>',
check:'<path d="M5 12l5 5L20 7"/>',x:'<path d="M6 6l12 12M18 6L6 18"/>'};
const ic=n=>`<svg class="i" viewBox="0 0 24 24">${IC[n]}</svg>`;
const fmt=t=>new Date(t*1000).toLocaleDateString(undefined,{day:'numeric',month:'short'});
let S={q:'',cites:[],copy:'',reg:false,canReg:true,live:false,remaining:0};

async function api(url,opt={}){
  const r=await fetch(url,opt);let d={};try{d=await r.json()}catch(e){}
  if(!r.ok){if(r.status===401&&!opt.quiet)showLogin();const e=new Error(typeof d.detail==='string'?d.detail:'Something went wrong.');e.status=r.status;throw e}
  return d}
const post=(u,b,quiet)=>api(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{}),quiet});

function showLogin(){
  S.live=false;
  $('#root').innerHTML=`<div class="login"><div class="logo"><b>${ic('search')}</b>Ask Doc</div>
  <div class="card" style="margin-top:24px"><h2>${S.reg?'Request an account':'Welcome back'}</h2>
  <p class="muted" style="margin-top:8px">${S.reg?'Your account will be available after the owner approves your request.':'Sign in to search your private document library.'}</p>
  <label for="u">Username</label><input id="u" autocomplete="username">
  <label for="p">Password</label><input id="p" type="password" autocomplete="${S.reg?'new-password':'current-password'}">
  <div class="err" id="lerr" style="margin-top:10px"></div>
  <button class="btn pri" data-act="auth" style="width:100%;margin-top:6px">${S.reg?'Submit approval request':'Sign in'}</button>
  ${S.canReg?`<button class="link" data-act="mode" style="margin:10px 0 0">${S.reg?'I already have an account':'Request an account'}</button>`:''}</div></div>`;
}
async function doAuth(){
  const body={username:$('#u').value,password:$('#p').value};
  try{const d=await post(S.reg?'/api/register':'/api/login',body,true);if(d.pending)showPending();else boot()}
  catch(e){$('#lerr').textContent=e.message}}
function showPending(){
  S.live=false;
  $('#root').innerHTML=`<div class="login"><div class="logo"><b>${ic('search')}</b>Ask Doc</div>
  <div class="card" style="margin-top:24px"><span class="pill">Approval pending</span><h2 style="margin-top:16px">Request received</h2>
  <p class="muted" style="margin-top:10px">Your request is waiting for the application owner. You can sign in after your account is approved.</p>
  <button class="btn" data-act="backlogin" style="margin-top:18px">Back to sign in</button></div></div>`;
}

function shell(){
  $('#root').innerHTML=`<div class="top"><div class="logo"><b>${ic('search')}</b><span class="brand-copy">Ask Doc<small>Ask your knowledge. Find the answer.</small></span></div>
  <div class="top-actions"><span class="pill plain session-pill" id="session-status">Session time remaining: --:--:--</span><button class="btn" id="admin-open" data-act="admin" hidden>Owner approvals</button><span class="identity"><b id="who" style="font-weight:500"></b><span class="role-badge" id="role-badge">User</span></span>
  <button class="btn ib" data-act="logout" aria-label="Sign out" style="border:1px solid #E3B5B0;color:var(--bad)">${ic('power')}</button></div></div>
  <div class="grid"><div class="side">
   <button class="prof" data-act="prof"><span class="av" id="av"></span><span style="flex:1"><b>My profile</b></span>${ic('chev')}</button>
   <div id="pm"><div class="row">${ic('up')}Upload your document</div>
    <label class="drop" id="drop" style="display:block">Drop PDF, Word or Excel files here, or <u>browse</u>
     <input id="file" type="file" multiple accept=".pdf,.docx,.xlsx" hidden></label>
    <div class="err" id="upmsg" style="margin:8px 0"></div>
    <div class="row">${ic('clock')}Daily usage in hours<span class="r" id="hrs">0.0 h</span></div>
    <div class="bar"><i id="hbar" style="width:0"></i></div>
    <div class="row">${ic('folder')}Files uploaded till date<span class="r" id="cnt">0</span></div>
    <div id="files"></div></div></div>
  <div><div class="sbox"><span class="muted">${ic('search')}</span>
   <input id="q" aria-label="Ask a question about your documents" placeholder="Ask a question about your documents…" autocomplete="off" maxlength="1200">
   <button class="btn pri" data-act="search" id="sbtn">Search</button></div><div id="out"></div></div></div>`;
  const f=$('#file'),d=$('#drop');
  f.onchange=()=>upload(f.files);
  d.ondragover=e=>{e.preventDefault();d.classList.add('over')};
  d.ondragleave=()=>d.classList.remove('over');
  d.ondrop=e=>{e.preventDefault();d.classList.remove('over');upload(e.dataTransfer.files)};
}
function paint(m){
  $('#who').textContent=m.username;$('#av').textContent=m.username.slice(0,2);
  const owner=m.role==='admin';
  const rb=$('#role-badge');rb.textContent=owner?'Owner':'User';rb.classList.toggle('owner',owner);
  $('#admin-open').hidden=!owner;
  S.remaining=m.remaining;sessionClock();
  hours(m.hours);$('#cnt').textContent=m.files.length;
  $('#files').innerHTML=m.files.length?m.files.map(f=>`<div class="file"><span class="tag ${esc(f.kind)}">${esc(f.kind.toUpperCase())}</span>
   <div class="fi"><b title="${esc(f.name)}">${esc(f.name)}</b><small>Uploaded ${fmt(f.uploaded)} · ${f.sections} sections</small></div>
   <button class="btn ib" data-act="del" data-id="${f.id}" aria-label="Delete ${esc(f.name)}">${ic('trash')}</button></div>`).join('')
   :'<p class="empty" style="font-size:13px">Your library is empty. Upload a file to start searching.</p>';
}
function sessionClock(){
  const el=$('#session-status');if(!el)return;
  const s=Math.max(0,S.remaining);
  const p=n=>String(n).padStart(2,'0');
  el.textContent=s?`Session time remaining: ${p(Math.floor(s/3600))}:${p(Math.floor(s%3600/60))}:${p(s%60)}`:'Session expired';
}
function hours(h){const a=$('#hrs'),b=$('#hbar');if(!a||!b)return;a.textContent=h.toFixed(1)+' h';b.style.width=Math.min(100,h/8*100)+'%'}
async function refresh(){paint(await api('/api/me'))}
async function upload(files){
  if(!files.length)return;const fd=new FormData();[...files].forEach(f=>fd.append('files',f));
  const m=$('#upmsg');m.className='err muted';m.textContent='Reading and indexing your files… this can take a minute.';
  try{const d=await api('/api/upload',{method:'POST',body:fd});
    const msg=d.done.map(x=>`${x.name}: ${x.sections} sections stored`).concat(d.errors).join('. ');
    m.className=d.errors.length?'err':'err ok';m.textContent=msg;
  }catch(e){m.className='err';m.textContent=e.message}
  $('#file').value='';refresh().catch(()=>{})}

function answerResult(d){
  const cites=d.citations||[];
  S.cites=cites.map(c=>c.section_id);
  const where=c=>c.page_num?`Page ${c.page_num}`:c.location;
  S.copy=d.answer+(cites.length?'\n\nSources:\n'+cites.map(c=>`- ${c.doc} (${where(c)})`).join('\n'):'');
  const cHtml=cites.map(c=>`<div class="citation"><h3>${esc(c.title)}</h3><p>${esc(c.doc)} · ${esc(where(c))}</p><blockquote>${esc(c.excerpt)}</blockquote></div>`).join('');
  $('#out').innerHTML=`<div class="card"><div class="ch"><div><h2>Answer</h2><div class="meta">${d.not_found?'No supporting answer found in your documents.':'Generated from your uploaded documents.'}</div></div>
    <div style="display:flex;gap:8px;align-items:center">${d.not_found?'':`<span id="confidence-score" class="match-score" title="${esc(d.match_explanation)}">Confidence · ${d.match_percent}%</span>`}
    <button class="btn" data-act="copy" id="cp">${ic('copy')}Copy</button></div></div>
    <p class="answer-text">${esc(d.answer)}</p>
    ${cites.length?`<div style="margin-top:20px"><h2 style="font-size:15px;margin-bottom:8px">Sources</h2>${cHtml}</div>`:''}
    ${!d.not_found&&cites.length?`<div class="fb" id="fb" style="padding-top:14px;border-top:1px solid var(--line);margin-top:6px"><div class="t"><b>Was this answer useful?</b><div class="meta">Your answer is remembered for this exact question.</div></div><button class="btn" data-act="yes" style="border-color:var(--acc);color:#0B5E57">${ic('check')}Yes</button><button class="btn" data-act="no">${ic('x')}No</button></div>`:''}
  </div><p class="meta">${esc(d.match_explanation||'Answers are limited to retrieved document evidence.')}</p>`;
}
async function search(){
  const q=$('#q').value.trim();if(!q)return;
  const b=$('#sbtn');b.disabled=true;
  S.q=q;$('#out').innerHTML='<p class="muted">Searching…</p>';
  try{answerResult(await post('/api/search',{question:q}))}
  catch(e){$('#out').innerHTML=`<p class="err">${esc(e.message)}</p>`}
  finally{b.disabled=false}}
async function copyAnswer(){
  try{await navigator.clipboard.writeText(S.copy)}
  catch(e){const t=document.createElement('textarea');t.value=S.copy;document.body.appendChild(t);t.select();try{document.execCommand('copy')}catch(_){}t.remove()}
  const b=$('#cp');if(b){b.innerHTML=ic('check')+'Copied';setTimeout(()=>{const x=$('#cp');if(x)x.innerHTML=ic('copy')+'Copy'},1500)}}
async function vote(helpful){
  const box=$('#fb');
  try{const result=await post('/api/feedback',{query:S.q,section_ids:S.cites,helpful});
    const badge=$('#confidence-score');
    const current=badge&&badge.textContent.match(/(\d+)%/);
    if(current&&Number.isFinite(result.score_delta)){
      const score=Math.max(0,Math.min(100,Number(current[1])+result.score_delta));
      badge.textContent=`Confidence · ${score}%`;
    }
    box.innerHTML=helpful?'<span class="ok">Thanks. Confidence increased, and this feedback will be used for this question next time.</span>'
      :'<span class="muted">Thanks. Confidence decreased, and these sources will rank lower for this question.</span>'}
  catch(e){box.innerHTML=`<span class="err">${esc(e.message)}</span>`}}

const A={
  mode(){S.reg=!S.reg;showLogin()},auth:doAuth,search,copy:copyAnswer,
  backlogin(){S.reg=false;showLogin()},
  admin:loadAdmin,
  backapp(){$('#out').innerHTML='';refresh().catch(()=>{})},
  async decision(t){
    t.disabled=true;
    try{await post('/api/admin/users/'+t.dataset.id+'/decision',{status:t.dataset.status});loadAdmin()}
    catch(e){$('#admin-error').textContent=e.message;t.disabled=false}
  },
  prof(){const p=$('#pm');p.hidden=!p.hidden},
  yes(){vote(true)},no(){vote(false)},
  async del(t){
    if(!confirm('Delete this file and everything learned from it?'))return;
    try{await api('/api/files/'+t.dataset.id,{method:'DELETE'})}catch(e){$('#upmsg').className='err';$('#upmsg').textContent=e.message}
    refresh().catch(()=>{})},
  async logout(){await post('/api/logout',{},true).catch(()=>{});showLogin()}};
async function loadAdmin(){
  try{
    const d=await api('/api/admin/requests');
    $('#out').innerHTML=`<div class="card"><div class="ch"><div><h2>Account requests</h2><div class="meta">Review access requests before users can sign in.</div></div><button class="btn" data-act="backapp">Close</button></div>
    <div class="err" id="admin-error"></div>${d.requests.length?d.requests.map(u=>`<div class="file"><div class="fi"><b>${esc(u.username)}</b><small>Requested ${fmt(u.created)}</small></div><div class="admin-actions"><button class="btn pri" data-act="decision" data-status="approved" data-id="${u.id}">Approve</button><button class="btn" data-act="decision" data-status="rejected" data-id="${u.id}">Reject</button></div></div>`).join(''):'<div class="empty">No account requests need review.</div>'}</div>`;
  }catch(e){$('#out').innerHTML=`<div class="card"><p class="err">${esc(e.message)}</p></div>`}
}
document.addEventListener('click',e=>{const t=e.target.closest('[data-act]');if(t&&A[t.dataset.act])A[t.dataset.act](t)});
document.addEventListener('keydown',e=>{if(e.key!=='Enter')return;
  if(e.target.id==='q')search();else if(e.target.id==='u'||e.target.id==='p')doAuth()});
setInterval(()=>{if(!S.live)return;if(S.remaining<=1){S.live=false;showLogin();return}S.remaining-=1;sessionClock()},1000);
setInterval(()=>{if(S.live)post('/api/heartbeat').then(d=>{hours(d.hours);S.remaining=d.remaining;sessionClock()}).catch(()=>{})},30000);

async function boot(){
  try{S.canReg=(await api('/api/config')).register}catch(e){S.canReg=false}
  try{const m=await api('/api/me',{quiet:true});S.live=true;shell();paint(m)}catch(e){showLogin()}}
boot();
</script></body></html>"""

if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
