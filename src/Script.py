#!/usr/bin/env python3
"""Prepare the corpus by default; run retrieval or Gemini only when requested."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from core import clean_paper_text, count_words, load_data, recursive_chunker

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "bm25", "full"), default="prepare")
    parser.add_argument("--data", type=Path, default=ROOT / "data/ModelizaciónEmpresaUCMData.json")
    parser.add_argument("--paper", type=Path, default=ROOT / "docs/Paper.txt")
    parser.add_argument("--output", type=Path, help="New output directory; defaults to a timestamped run.")
    parser.add_argument("--limit", type=int, help="Use the first N questions for a smoke check.")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dense weight in hybrid fusion.")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--chunk-size", type=int, default=250, help="Maximum words per chunk.")
    parser.add_argument("--overlap", type=int, default=30, help="Overlap in words.")
    parser.add_argument("--sleep", type=float, default=5, help="Pause after each Gemini response.")
    parser.add_argument("--model", help="Gemini model ID; otherwise read GEMINI_MODEL from .env.")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.rounds < 1 or (args.limit is not None and args.limit < 1):
        parser.error("rounds and limit must be positive")
    if not 0 <= args.alpha <= 1 or not 0 <= args.threshold <= 1:
        parser.error("alpha and threshold must be between 0 and 1")
    if not math.isfinite(args.sleep) or args.sleep < 0:
        parser.error("sleep must be a finite non-negative number")
    if args.chunk_size <= 0 or not 0 <= args.overlap < args.chunk_size:
        parser.error("require chunk-size > 0 and 0 <= overlap < chunk-size")

    data = load_data(args.data)
    dataset_count = len(data)
    if args.limit:
        data = data[:args.limit]
    cleaned = clean_paper_text(args.paper.read_text(encoding="utf-8"))
    chunks = recursive_chunker(cleaned, args.chunk_size, args.overlap)
    if not chunks or not any(character.isalpha() for character in cleaned):
        raise ValueError("The paper contains no usable text after cleaning.")

    api_key = model_name = None
    if args.mode == "full":
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        model_name = args.model or os.getenv("GEMINI_MODEL", "").strip()
        if not api_key or api_key == "YOUR_API_KEY_HERE" or not model_name:
            parser.error("full mode requires GEMINI_API_KEY and GEMINI_MODEL (or --model)")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or ROOT / "outputs" / timestamp).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory is not empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "paper_clean.txt").write_text(cleaned + "\n", encoding="utf-8")
    write_json(output / "chunks.json", chunks)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(output=str(output), model=model_name, dataset_questions=dataset_count,
                    evaluated_questions=len(data), chunk_count=len(chunks),
                    max_chunk_words=max(map(count_words, chunks)), started_at=timestamp,
                    python=platform.python_version(), status="prepared")
    settings["input_sha256"] = {label: hashlib.sha256(path.read_bytes()).hexdigest()
                                for label, path in (("data", args.data), ("paper", args.paper))}
    settings["package_versions"] = {}
    for package in ("numpy", "pandas", "matplotlib", "rank-bm25", "chromadb",
                    "sentence-transformers", "google-genai", "python-dotenv"):
        try:
            settings["package_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(output / "run.json", settings)
    print(f"Validated {dataset_count} questions; selected {len(data)}; created {len(chunks)} chunks.")
    if args.mode == "prepare":
        print(f"Preparation complete (no network or model calls): {output}")
        return output

    try:
        from rank_bm25 import BM25Okapi
        from core import better_tokenize, check_fuzzy_attribution
        bm25 = BM25Okapi([better_tokenize(chunk) for chunk in chunks])
        if args.mode == "bm25":
            rows = []
            for number, item in enumerate(data, 1):
                context = "\n---\n".join(bm25.get_top_n(better_tokenize(item["question"]), chunks, n=3))
                hit, score = check_fuzzy_attribution(context, item["paper_reference"], args.threshold)
                rows.append({"question_number": number, "question": item["question"],
                             "reference_coverage": score, "hit": hit, "context": context})
            write_json(output / "bm25_retrieval.json", rows)
            summary = {"questions": len(rows), "top_k": 3, "threshold": args.threshold,
                       "reference_hit_rate_percent": 100 * sum(row["hit"] for row in rows) / len(rows),
                       "mean_reference_coverage": sum(row["reference_coverage"] for row in rows) / len(rows)}
            write_json(output / "bm25_summary.json", summary)
            print(json.dumps(summary, indent=2))
        else:
            import experiment
            from google import genai
            from sentence_transformers import CrossEncoder

            experiment.OUTPUT_DIR = output
            experiment.SLEEP_SECONDS = args.sleep
            print(f"Full run: {4 * len(data) * args.rounds} Gemini requests before retries; model={model_name}.")
            chroma_client, collection = experiment.build_vector_collection(chunks)
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            with genai.Client(api_key=api_key) as client:
                class GeminiModel:
                    def generate_content(self, prompt):
                        return client.models.generate_content(
                            model=model_name, contents=prompt, config={"temperature": 0})

                results = experiment.run_full_pipeline(
                    data, chunks, GeminiModel(), bm25, collection, reranker,
                    rounds=args.rounds, alpha=args.alpha, threshold=args.threshold)
                for name, table in results.items():
                    table.to_csv(output / f"{name}.csv", index=False)
        settings["status"] = "completed"
    except Exception as error:
        settings.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        write_json(output / "run.json", settings)
    print(f"Results saved: {output}")
    return output


if __name__ == "__main__":
    main()
