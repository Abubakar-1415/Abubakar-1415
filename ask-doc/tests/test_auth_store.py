import unittest
from unittest.mock import MagicMock, patch

import psycopg

import auth_store


class AuthStoreTests(unittest.TestCase):
    def test_username_is_normalized_and_validated(self):
        self.assertEqual(auth_store.normalize_username("  Alice_01 "), "alice_01")
        with self.assertRaises(ValueError):
            auth_store.normalize_username("x")

    def test_password_rules_and_salted_hashes(self):
        with self.assertRaises(ValueError):
            auth_store.validate_password("short")

        auth_store.validate_password("long-password-123")
        first = auth_store._password_digest("long-password-123", b"a" * 16)
        second = auth_store._password_digest("long-password-123", b"b" * 16)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first.hex(), "long-password-123")

    def test_session_expires_at_four_hour_boundary(self):
        started_at = 1000.0
        self.assertFalse(
            auth_store.session_expired(
                started_at,
                started_at + auth_store.SESSION_DURATION_SECONDS - 1,
            )
        )
        self.assertTrue(
            auth_store.session_expired(
                started_at,
                started_at + auth_store.SESSION_DURATION_SECONDS,
            )
        )

    def test_authentication_accepts_only_the_matching_password(self):
        password = "long-password-123"
        salt = b"s" * 16
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.execute.return_value.fetchone.return_value = (
            123,
            "alice",
            salt.hex(),
            auth_store._password_digest(password, salt).hex(),
        )

        with patch.object(auth_store.psycopg, "connect", return_value=connection):
            self.assertEqual(
                auth_store.authenticate_user("postgresql://test", "Alice", password),
                (123, "alice"),
            )
            self.assertIsNone(
                auth_store.authenticate_user(
                    "postgresql://test", "alice", "another-password"
                )
            )

    def test_duplicate_username_is_reported(self):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.execute.side_effect = psycopg.errors.UniqueViolation("duplicate")

        with patch.object(auth_store.psycopg, "connect", return_value=connection):
            with self.assertRaises(auth_store.UsernameAlreadyExistsError):
                auth_store.create_user(
                    "postgresql://test", "alice", "long-password-123"
                )


if __name__ == "__main__":
    unittest.main()
