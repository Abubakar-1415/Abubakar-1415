import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import rag_app
import search


class SearchRetrievalTests(unittest.TestCase):
    @staticmethod
    def point(point_id, section_id, doc_name, text, title="Work history"):
        return SimpleNamespace(
            id=point_id,
            score=0.0,
            payload={
                "user_id": 7,
                "section_id": section_id,
                "doc_name": doc_name,
                "kind": "docx",
                "title": title,
                "location": "Experience",
                "page_num": None,
                "text": text,
            },
        )

    def test_query_terms_keep_exact_names_and_keywords(self):
        self.assertEqual(
            search.query_terms("Siddiq Resume work experience"),
            ["siddiq", "resume", "work", "experience"],
        )

    def test_keyword_context_returns_exact_match_with_prefix_and_suffix(self):
        contexts = search.exact_keyword_contexts(
            "Worked at Northwind as a data analyst.", ["analyst"]
        )
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]["keyword"], "analyst")
        self.assertIn("as a data", contexts[0]["prefix"])
        self.assertIn(".", contexts[0]["suffix"])

    def test_answer_sources_are_spread_across_documents(self):
        candidates = [
            {"doc": "resume-a", "lexical_score": 5, "score": 0.8, "section_id": i}
            for i in range(6)
        ] + [
            {"doc": "resume-b", "lexical_score": 4, "score": 0.7, "section_id": i + 10}
            for i in range(6)
        ]

        sources = search.select_answer_sources(candidates)

        self.assertEqual(len(sources), 4)
        self.assertEqual(
            sum(source["doc"] == "resume-a" for source in sources), 2
        )
        self.assertEqual(
            sum(source["doc"] == "resume-b" for source in sources), 2
        )

    def test_full_library_search_returns_matches_from_multiple_documents(self):
        first = self.point(
            "point-a", 101, "Siddiq_Resume_DataReportingAnalyst.docx",
            "Siddiq worked as a reporting analyst for Northwind.",
        )
        second = self.point(
            "point-b", 202, "Siddiq_Resume_GenAI_Developer.docx",
            "Siddiq has experience as a GenAI developer.",
        )
        client = MagicMock()
        client.query_points.return_value.points = []
        client.scroll.side_effect = [([first], "page-2"), ([second], None)]
        embeddings = MagicMock()
        embeddings.embed_query.return_value = [0.1, 0.2]

        with (
            patch.object(rag_app, "rag_components", return_value=(client, embeddings, None)),
            patch.object(search, "feedback_votes", return_value={}),
        ):
            result = search.retrieve(
                7, "Siddiq resume work experience"
            )

        self.assertEqual(result["scanned_chunks"], 2)
        self.assertEqual(result["keyword_match_count"], 2)
        self.assertEqual(result["keyword_match_documents"], 2)
        self.assertEqual(
            {match["doc"] for match in result["keyword_matches"]},
            {
                "Siddiq_Resume_DataReportingAnalyst.docx",
                "Siddiq_Resume_GenAI_Developer.docx",
            },
        )
        self.assertTrue(
            all(match["keyword_contexts"] for match in result["keyword_matches"])
        )
        self.assertEqual(client.scroll.call_count, 2)
        self.assertEqual(
            client.scroll.call_args_list[0].kwargs["scroll_filter"],
            client.scroll.call_args_list[1].kwargs["scroll_filter"],
        )
        self.assertEqual(
            client.scroll.call_args_list[0].kwargs["scroll_filter"].must[0].key,
            "user_id",
        )

    def test_exact_search_still_works_when_embedding_service_fails(self):
        point = self.point(
            "point-a", 101, "resume.docx", "Analyst with experience in reporting."
        )
        client = MagicMock()
        client.scroll.return_value = ([point], None)
        embeddings = MagicMock()
        embeddings.embed_query.side_effect = RuntimeError("embedding unavailable")

        with (
            patch.object(rag_app, "rag_components", return_value=(client, embeddings, None)),
            patch.object(search, "feedback_votes", return_value={}),
        ):
            result = search.retrieve(7, "analyst experience")

        self.assertIn("embedding unavailable", result["semantic_error"])
        self.assertEqual(result["keyword_match_count"], 1)
        self.assertEqual(result["keyword_matches"][0]["doc"], "resume.docx")
        self.assertEqual(result["scanned_chunks"], 1)


if __name__ == "__main__":
    unittest.main()
