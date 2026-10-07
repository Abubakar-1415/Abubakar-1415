import unittest
import json
from unittest.mock import MagicMock, patch

import auth_store
import rag_app


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

    def test_usernames_map_to_private_stable_ids(self):
        alice_id = auth_store.user_id_for("alice")
        self.assertEqual(alice_id, auth_store.user_id_for("alice"))
        self.assertNotEqual(alice_id, auth_store.user_id_for("bob"))
        self.assertNotEqual(
            auth_store.user_point_id("alice"),
            auth_store.user_point_id("bob"),
        )

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
        point = MagicMock()
        point.payload = {
            "user_id": 123,
            "username": "alice",
            "password_salt": salt.hex(),
            "password_hash": auth_store._password_digest(password, salt).hex(),
        }
        client = MagicMock()
        client.retrieve.return_value = [point]

        with patch.object(auth_store, "_get_user", return_value=point):
            self.assertEqual(
                auth_store.authenticate_user(client, "Alice", password),
                (123, "alice"),
            )
            self.assertIsNone(
                auth_store.authenticate_user(
                    client, "alice", "another-password"
                )
            )

    def test_registration_uses_atomic_insert_only(self):
        client = MagicMock()

        with patch.object(auth_store, "_get_user", return_value=None):
            user_id = auth_store.create_user(
                client, "Alice", "long-password-123"
            )

        self.assertEqual(user_id, auth_store.user_id_for("alice"))
        call = client.upsert.call_args.kwargs
        self.assertEqual(call["update_mode"], auth_store.models.UpdateMode.INSERT_ONLY)
        saved = call["points"][0]
        self.assertEqual(saved.payload["username"], "alice")
        self.assertNotEqual(saved.payload["password_hash"], "long-password-123")

    def test_existing_username_is_not_overwritten(self):
        client = MagicMock()
        existing = MagicMock()
        with patch.object(auth_store, "_get_user", return_value=existing):
            with self.assertRaises(auth_store.UsernameAlreadyExistsError):
                auth_store.create_user(
                    client, "alice", "long-password-123"
                )
        client.upsert.assert_not_called()

    def test_concurrent_duplicate_registration_is_rejected(self):
        client = MagicMock()
        existing = MagicMock()
        client.upsert.side_effect = RuntimeError("insert-only conflict")
        with patch.object(
            auth_store, "_get_user", side_effect=[None, existing]
        ):
            with self.assertRaises(auth_store.UsernameAlreadyExistsError):
                auth_store.create_user(
                    client, "alice", "long-password-123"
                )

    def test_missing_account_is_rejected(self):
        client = MagicMock()
        with patch.object(auth_store, "_get_user", return_value=None):
            self.assertIsNone(
                auth_store.authenticate_user(
                    client, "missing-user", "long-password-123"
                )
            )

    def test_user_store_requires_qdrant_credentials(self):
        with self.assertRaisesRegex(RuntimeError, "QDRANT_URL and QDRANT_API_KEY"):
            auth_store.initialize_user_store("", "")

    def test_empty_search_answer_is_json_serializable(self):
        answer = rag_app.answer_from_sources("question", [])
        serialized = json.dumps(answer)
        self.assertEqual(json.loads(serialized), answer)
        self.assertTrue(answer["not_found"])

    def test_answer_provider_error_includes_redacted_cause(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError(
            "HTTP 400 invalid temperature"
        )
        sources = [{
            "id": "S1",
            "doc": "sample.docx",
            "title": "Sample",
            "location": "section 1",
            "excerpt": "Relevant evidence.",
        }]
        with patch.object(rag_app, "rag_components", return_value=(None, None, llm)):
            with self.assertRaises(rag_app.HTTPException) as raised:
                rag_app.answer_from_sources("question", sources)

        self.assertEqual(raised.exception.status_code, 502)
        self.assertIn("RuntimeError", raised.exception.detail)
        self.assertIn("invalid temperature", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
