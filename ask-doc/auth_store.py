"""Qdrant-backed username/password storage for Ask Doc."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import uuid

from qdrant_client import QdrantClient, models

USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")
PASSWORD_MIN_LENGTH = 12
PASSWORD_MAX_LENGTH = 1024
SESSION_DURATION_SECONDS = 4 * 60 * 60
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
USER_COLLECTION = "ask_doc_users_v1"


class UsernameAlreadyExistsError(Exception):
    """Raised when a username is already registered."""


def session_expired(started_at: float, now: float) -> bool:
    return now - started_at >= SESSION_DURATION_SECONDS


def normalize_username(username: str) -> str:
    normalized = username.strip().lower()
    if not USERNAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Username must be 3-32 characters: letters, numbers, dots, "
            "underscores, or hyphens."
        )
    return normalized


def validate_password(password: str) -> None:
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError("Password must be at least 12 characters long.")
    if len(password) > PASSWORD_MAX_LENGTH:
        raise ValueError("Password must be no more than 1024 characters.")


def user_point_id(username: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ask-doc-user:{username}"))


def user_id_for(username: str) -> int:
    digest = hashlib.sha256(f"ask-doc-user:{username}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _password_digest(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=32,
    )


def initialize_user_store(qdrant_url: str, api_key: str) -> QdrantClient:
    if not qdrant_url.strip() or not api_key.strip():
        raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY in app secrets.")
    client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=30)
    try:
        if not client.collection_exists(USER_COLLECTION):
            client.create_collection(
                collection_name=USER_COLLECTION,
                vectors_config=models.VectorParams(
                    size=1, distance=models.Distance.COSINE
                ),
            )
        client.create_payload_index(
            collection_name=USER_COLLECTION,
            field_name="username",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        client.close()
        raise
    return client


def _get_user(client: QdrantClient, username: str):
    points = client.retrieve(
        collection_name=USER_COLLECTION,
        ids=[user_point_id(username)],
        with_payload=True,
        with_vectors=False,
    )
    return points[0] if points else None


def create_user(client: QdrantClient, username: str, password: str) -> int:
    normalized = normalize_username(username)
    validate_password(password)
    if _get_user(client, normalized) is not None:
        raise UsernameAlreadyExistsError
    salt = os.urandom(16)
    digest = _password_digest(password, salt)
    user_id = user_id_for(normalized)
    point = models.PointStruct(
        id=user_point_id(normalized),
        vector=[1.0],
        payload={
            "user_id": user_id,
            "username": normalized,
            "password_salt": salt.hex(),
            "password_hash": digest.hex(),
        },
    )
    try:
        client.upsert(
            collection_name=USER_COLLECTION,
            points=[point],
            wait=True,
            update_mode=models.UpdateMode.INSERT_ONLY,
        )
    except Exception as exc:
        existing = _get_user(client, normalized)
        if existing is not None:
            raise UsernameAlreadyExistsError from exc
        raise
    return user_id


def authenticate_user(
    client: QdrantClient, username: str, password: str
) -> tuple[int, str] | None:
    normalized = normalize_username(username)
    point = _get_user(client, normalized)
    if point is None or not point.payload:
        candidate = _password_digest(password, b"\0" * 16)
        hmac.compare_digest(candidate, b"\0" * 32)
        return None

    payload = point.payload
    salt = bytes.fromhex(str(payload["password_salt"]))
    expected = str(payload["password_hash"])
    candidate = _password_digest(password, salt)
    if not hmac.compare_digest(candidate.hex(), expected):
        return None
    return int(payload["user_id"]), str(payload["username"])
