"""Exercise pipeline wiring locally; test doubles are not model-quality evidence."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment
from core import better_tokenize


class ExperimentTests(unittest.TestCase):
    def test_all_metrics_with_one_round_and_small_corpus(self):
        chunks = ["rare evidence number one", "common second passage", "unrelated third passage"]
        data = [{"question": "Where is rare evidence?", "answers": dict(zip("ABCD", "abcd")),
                 "correct_answer": "A", "paper_reference": "rare evidence number one"}]
        bm25 = BM25Okapi([better_tokenize(chunk) for chunk in chunks])
        queries, prompts = [], []

        class Collection:
            def query(self, query_texts, n_results):
                queries.append(n_results)
                self_result = {"ids": [[str(i) for i in range(n_results)]],
                               "documents": [chunks[:n_results]],
                               "distances": [[i / 10 for i in range(n_results)]]}
                return self_result

        class Reranker:
            def predict(self, pairs):
                return np.arange(len(pairs), 0, -1)

        class Model:
            def generate_content(self, prompt):
                prompts.append(prompt)
                return SimpleNamespace(text="Answer: A\nSource: rare evidence")

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(experiment, "OUTPUT_DIR", Path(directory)), patch.object(experiment, "SLEEP_SECONDS", 0):
                results = experiment.run_full_pipeline(data, chunks, Model(), bm25, Collection(), Reranker())
            self.assertEqual(len(prompts), 4)
            self.assertTrue(all(size <= len(chunks) for size in queries))
            self.assertNotIn("CONTEXT:", prompts[0])
            self.assertTrue(all("CONTEXT:" in prompt for prompt in prompts[1:]))
            self.assertEqual(results["accuracy"]["accuracy"].tolist(), [100.0] * 4)
            self.assertEqual(results["source"]["source_accuracy"].tolist(), [100.0] * 3)
            self.assertTrue((results["recall"][["BM25", "Dense", "Hybrid"]] == "100.0%").all().all())
            self.assertEqual(len(list(Path(directory).glob("*.png"))), 4)
            self.assertTrue((Path(directory) / "CONCLUSIONES_PIPELINE.txt").exists())

    def test_api_failures_do_not_become_zero_accuracy(self):
        class BrokenModel:
            def generate_content(self, prompt):
                raise ConnectionError("test failure")
        with self.assertRaises(RuntimeError):
            experiment.call_gemini(BrokenModel(), "test", max_retries=2, sleep=0)

    def test_empty_response_is_an_abstention(self):
        model = SimpleNamespace(generate_content=lambda prompt: SimpleNamespace(text=None))
        self.assertEqual(experiment.call_gemini(model, "test", sleep=0), "")

    def test_recall_respects_the_requested_threshold(self):
        chunks = ["rare evidence", "other words"]
        data = [{"question": "rare", "paper_reference": "rare evidence absent"}]
        bm25 = BM25Okapi([better_tokenize(chunk) for chunk in chunks])
        collection = SimpleNamespace(query=lambda **kwargs: {
            "ids": [["0", "1"]], "documents": [chunks], "distances": [[0.0, 1.0]]})
        result = experiment.calculate_recall_metrics_complete(
            data, chunks, bm25, collection, k_values=[2], threshold=1.0)
        self.assertEqual(result.iloc[0]["Dense"], "0.0%")


if __name__ == "__main__":
    unittest.main()
