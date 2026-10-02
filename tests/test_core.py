import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core import (check_fuzzy_attribution, clean_paper_text, compute_accuracy,
                  load_data, parse_mc_answer, recursive_chunker)


class CoreTests(unittest.TestCase):
    def test_supplied_dataset(self):
        self.assertEqual(len(load_data(ROOT / "data/ModelizaciónEmpresaUCMData.json")), 70)

    def test_invalid_and_duplicate_records(self):
        valid = {"question": "Q", "answers": dict(zip("ABCD", "abcd")),
                 "correct_answer": "A", "paper_reference": "Evidence"}
        invalid = [[], [None], [valid, valid], [{**valid, "correct_answer": "E"}],
                   [{**valid, "answers": {"A": "one"}}], [{**valid, "paper_reference": ""}]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            for value in invalid:
                with self.subTest(value=value):
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_data(path)

    def test_chunk_size_and_overlap_for_a_long_sentence(self):
        words = [f"word{i}" for i in range(811)]
        chunks = [chunk.split() for chunk in recursive_chunker(" ".join(words))]
        self.assertTrue(all(0 < len(chunk) <= 250 for chunk in chunks))
        rebuilt = chunks[0][:]
        for previous, following in zip(chunks, chunks[1:]):
            self.assertEqual(previous[-30:], following[:30])
            rebuilt.extend(following[30:])
        self.assertEqual(rebuilt, words)

    def test_zero_overlap_and_paragraph_overflow(self):
        text = "one two three.\n\nfour five six seven eight nine."
        chunks = recursive_chunker(text, max_size=5, overlap=0)
        self.assertEqual(" ".join(chunks), " ".join(text.split()))
        self.assertTrue(all(len(chunk.split()) <= 5 for chunk in chunks))

    def test_chunk_boundaries_and_invalid_settings(self):
        self.assertEqual(recursive_chunker(""), [])
        self.assertEqual(recursive_chunker("a b c", 3, 0), ["a b c"])
        for size, overlap in [(0, 0), (3, 3), (3, -1)]:
            with self.assertRaises(ValueError):
                recursive_chunker("words", size, overlap)

    def test_cleaning(self):
        text = "A hyphen-\nated line.\n\n3\nNext sentence.\n\nReferences\nRemove this.\nAppendix\nKeep this."
        result = clean_paper_text(text)
        self.assertIn("hyphenated", result)
        self.assertNotIn("Remove this", result)
        self.assertIn("Keep this", result)
        self.assertNotIn("3", result)

    def test_answer_parser_does_not_read_letters_from_evidence(self):
        for text, answer in [("A", "A"), ("Reasoning: data\nAnswer: c\nSource: Appendix D", "C"),
                             ("Answer: N/A\nSource: A and B", "N/A"),
                             ("See Appendix D", "N/A"), ("Answer: Definitely B", "N/A"),
                             ("", "N/A")]:
            with self.subTest(text=text):
                self.assertEqual(parse_mc_answer(text), answer)

    def test_accuracy_rejects_truncation(self):
        self.assertEqual(compute_accuracy(["A", "N/A"], ["A", "B"]), (50.0, 1, 2, [2]))
        with self.assertRaises(ValueError):
            compute_accuracy(["A"], ["A", "B"])

    def test_coverage_counts_reference_occurrences(self):
        self.assertEqual(check_fuzzy_attribution("a b", "a a b c"), (True, 0.75))
        self.assertEqual(check_fuzzy_attribution("a b", "a a b c", 0.8), (False, 0.75))
        self.assertEqual(check_fuzzy_attribution("", "reference"), (False, 0.0))

    def test_cli_runs_outside_repository_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            command = [sys.executable, str(ROOT / "src/main.py"), "--output", str(output)]
            completed = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metadata = json.loads((output / "run.json").read_text())
            self.assertEqual(metadata["dataset_questions"], 70)
            self.assertLessEqual(metadata["max_chunk_words"], 250)
            self.assertEqual(metadata["status"], "prepared")
            repeated = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("not empty", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
