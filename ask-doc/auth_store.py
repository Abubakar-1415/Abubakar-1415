"""Persistent username/password storage for Ask Doc."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets

import psycopg
from psycopg.errors import UniqueViolation

USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")
PASSWORD_MIN_LENGTH = 12
PASSWORD_MAX_LENGTH = 1024
SESSION_DURATION_SECONDS = 4 * 60 * 60
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1


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


def _password_digest(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=32,
    )


def initialize_user_store(database_url: str) -> None:
    if not database_url.strip():
        raise RuntimeError("Set DATABASE_URL in Streamlit app secrets.")
    with psycopg.connect(database_url) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ask_doc_users (
                user_id BIGINT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def create_user(database_url: str, username: str, password: str) -> int:
    normalized = normalize_username(username)
    validate_password(password)
    salt = os.urandom(16)
    digest = _password_digest(password, salt)
    user_id = secrets.randbits(63) or 1
    try:
        with psycopg.connect(database_url) as connection:
            connection.execute(
                """
                INSERT INTO ask_doc_users
                    (user_id, username, password_salt, password_hash)
                VALUES (%s, %s, %s, %s)
                """,
                (user_id, normalized, salt.hex(), digest.hex()),
            )
    except UniqueViolation as exc:
        raise UsernameAlreadyExistsError from exc
    return user_id


def authenticate_user(
    database_url: str, username: str, password: str
) -> tuple[int, str] | None:
    normalized = normalize_username(username)
    with psycopg.connect(database_url) as connection:
        row = connection.execute(
            """
            SELECT user_id, username, password_salt, password_hash
            FROM ask_doc_users
            WHERE username = %s
            """,
            (normalized,),
        ).fetchone()

    if row is None:
        candidate = _password_digest(password, b"\0" * 16)
        hmac.compare_digest(candidate, b"\0" * 32)
        return None

    user_id, stored_username, salt_hex, digest_hex = row
    candidate = _password_digest(password, bytes.fromhex(salt_hex))
    if not hmac.compare_digest(candidate.hex(), digest_hex):
        return None
    return int(user_id), str(stored_username)
