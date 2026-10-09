"""Check the pipeline with simple fake models; no network or API key is needed."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment
from core import better_tokenize


class FakeCollection:
    """Return a fixed search result and record the requested candidate counts."""

    def __init__(self, documents, ids, distances):
        self.documents = documents
        self.ids = ids
        self.distances = distances
        self.query_sizes = []

    def query(self, query_texts, n_results):
        self.query_sizes.append(n_results)
        return {
            "documents": [self.documents[:n_results]],
            "ids": [self.ids[:n_results]],
            "distances": [self.distances[:n_results]],
        }


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        return np.array(self.scores[:len(pairs)])


class FakeBM25:
    def get_scores(self, tokens):
        return np.array([4.0, 0.0, 2.0])


class ExperimentTests(unittest.TestCase):
    def test_all_metrics_with_one_round_and_small_corpus(self):
        chunks = ["rare evidence number one", "common second passage", "unrelated third passage"]
        data = [{
            "question": "Where is rare evidence?",
            "answers": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "correct_answer": "A",
            "paper_reference": "rare evidence number one",
        }]
        bm25 = BM25Okapi([better_tokenize(chunk) for chunk in chunks])
        collection = FakeCollection(chunks, ["0", "1", "2"], [0.0, 0.1, 0.2])
        reranker = FakeReranker([3, 2, 1])
        prompts = []

        def generate(prompt):
            prompts.append(prompt)
            return "Answer: A\nSource: rare evidence"

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            retriever = experiment.Retriever(chunks, bm25, collection, reranker)
            results = experiment.run_full_pipeline(data, retriever, generate, output, sleep=0)
            self.assertEqual(len(prompts), 4)
            for size in collection.query_sizes:
                self.assertLessEqual(size, len(chunks))
            self.assertNotIn("CONTEXT:", prompts[0])
            for prompt in prompts[1:]:
                self.assertIn("CONTEXT:", prompt)
            self.assertEqual(results["accuracy"]["accuracy"].tolist(), [100.0] * 4)
            self.assertEqual(results["source"]["source_accuracy"].tolist(), [100.0] * 3)
            for method in ("BM25", "Dense", "Hybrid"):
                self.assertEqual(results["recall"][method].tolist(), ["100.0%"] * 4)
            self.assertEqual(len(list(output.glob("*.png"))), 4)
            self.assertEqual(len(list(output.glob("*.csv"))), 4)
            records = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual([row["method"] for row in records], ["baseline", "bm25", "dense", "hybrid"])
            self.assertEqual([row["prompt"] for row in records], prompts)
            self.assertTrue(all(row["correct"] for row in records))
            self.assertTrue(all(row["response"] == "Answer: A\nSource: rare evidence" for row in records))

    def test_failed_run_keeps_completed_predictions(self):
        data = [{"question": "Question", "answers": dict.fromkeys("ABCD", "Option"),
                 "correct_answer": "A", "paper_reference": "Evidence"}]
        retriever = Mock()
        retriever.retrieve.return_value = ["Evidence"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("experiment.call_gemini", side_effect=["A", RuntimeError("Unavailable")]):
                with self.assertRaises(RuntimeError):
                    experiment.run_full_pipeline(data, retriever, None, output, sleep=0)
            records = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["prediction"], "A")
            self.assertEqual(records[0]["method"], "baseline")
            with self.assertRaises(FileExistsError):
                experiment.run_full_pipeline(data, retriever, None, output, sleep=0)

    def test_completed_tables_survive_a_later_metric_failure(self):
        data = [{"question": "Question", "answers": dict.fromkeys("ABCD", "Option"),
                 "correct_answer": "A", "paper_reference": "Evidence"}]
        retriever = Mock()
        retriever.retrieve.return_value = ["Evidence"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("experiment.evaluate_sources", side_effect=RuntimeError("Search failed")):
                with self.assertRaises(RuntimeError):
                    experiment.run_full_pipeline(data, retriever, lambda _: "A", output, sleep=0)
            self.assertTrue((output / "accuracy.csv").exists())
            self.assertEqual(len((output / "predictions.jsonl").read_text().splitlines()), 4)

    def test_interrupting_the_pause_preserves_the_response(self):
        data = [{"question": "Question", "answers": dict.fromkeys("ABCD", "Option"),
                 "correct_answer": "A", "paper_reference": "Evidence"}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("experiment.time.sleep", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    experiment.run_full_pipeline(data, None, lambda _: "A", output)
            record = json.loads((output / "predictions.jsonl").read_text())
            self.assertEqual(record["response"], "A")

    def test_api_failures_do_not_become_zero_accuracy(self):
        def broken_generate(prompt):
            raise ConnectionError("test failure")

        with self.assertRaises(RuntimeError):
            experiment.call_gemini(broken_generate, "test", max_retries=2, sleep=0)

    def test_empty_response_is_an_abstention(self):
        def empty_generate(prompt):
            return None

        self.assertEqual(experiment.call_gemini(empty_generate, "test", sleep=0), "")

    def test_permanent_api_error_is_not_retried(self):
        attempts = []

        class UnauthorizedError(Exception):
            code = 401

        def unauthorized_generate(prompt):
            attempts.append(prompt)
            raise UnauthorizedError("invalid key")

        with self.assertRaises(RuntimeError):
            experiment.call_gemini(unauthorized_generate, "test", sleep=0)
        self.assertEqual(len(attempts), 1)

    def test_recall_respects_the_requested_threshold(self):
        chunks = ["rare evidence", "other words"]
        data = [{"question": "rare", "paper_reference": "rare evidence absent"}]
        bm25 = BM25Okapi([better_tokenize(chunk) for chunk in chunks])
        collection = FakeCollection(chunks, ["0", "1"], [0.0, 1.0])
        retriever = experiment.Retriever(chunks, bm25, collection)
        result = experiment.evaluate_recall(data, retriever, k_values=[2], threshold=1.0)
        self.assertEqual(result.iloc[0]["Dense"], "0.0%")

    def test_hybrid_weights_and_stable_ties(self):
        chunks = ["lexical", "semantic", "both"]
        collection = FakeCollection(
            ["semantic", "both", "lexical"], ["1", "2", "0"], [0.0, 0.4, 1.0]
        )
        cases = [
            (0, ["lexical", "both", "semantic"]),
            (0.5, ["both", "semantic", "lexical"]),
            (1, ["semantic", "both", "lexical"]),
        ]
        for alpha, expected in cases:
            with self.subTest(alpha=alpha):
                retriever = experiment.Retriever(chunks, FakeBM25(), collection, alpha=alpha)
                actual = retriever.retrieve("query", "hybrid", rerank=False)
                self.assertEqual(actual, expected)

    def test_reranking_reorders_dense_candidates(self):
        chunks = ["first", "second", "third"]
        collection = FakeCollection(chunks, ["0", "1", "2"], [0.0, 0.1, 0.2])
        reranker = FakeReranker([0.1, 0.9, 0.4])
        retriever = experiment.Retriever(chunks, None, collection, reranker)
        self.assertEqual(retriever.retrieve("query", "dense", top_k=2), ["second", "third"])

    def test_hybrid_keeps_bm25_only_candidates_before_reranking(self):
        chunks = ["lexical", "semantic", "both"]
        collection = FakeCollection(["semantic"], ["1"], [0.0])
        retriever = experiment.Retriever(chunks, FakeBM25(), collection, alpha=0.25)
        actual = retriever.retrieve("query", "hybrid", hybrid_pool=1, rerank=False)
        self.assertEqual(actual, ["lexical", "both", "semantic"])


if __name__ == "__main__":
    unittest.main()
