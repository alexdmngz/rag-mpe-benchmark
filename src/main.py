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

from core import (
    better_tokenize,
    check_fuzzy_attribution,
    clean_paper_text,
    count_words,
    load_data,
    recursive_chunker,
)

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


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


def validate_arguments(args, parser):
    if args.rounds < 1:
        parser.error("rounds must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if not 0 <= args.alpha <= 1:
        parser.error("alpha must be between 0 and 1")
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    if not math.isfinite(args.sleep) or args.sleep < 0:
        parser.error("sleep must be a finite non-negative number")
    if args.chunk_size <= 0 or not 0 <= args.overlap < args.chunk_size:
        parser.error("require chunk-size > 0 and 0 <= overlap < chunk-size")


def build_run_info(args, output, timestamp, model_name, dataset_count, data, chunks):
    """Record the settings and inputs needed to compare experiment runs."""
    settings = vars(args).copy()
    for name, value in settings.items():
        if isinstance(value, Path):
            settings[name] = str(value)

    settings["output"] = str(output)
    settings["model"] = model_name
    settings["dataset_questions"] = dataset_count
    settings["evaluated_questions"] = len(data)
    settings["chunk_count"] = len(chunks)
    settings["max_chunk_words"] = max(count_words(chunk) for chunk in chunks)
    settings["started_at"] = timestamp
    settings["python"] = platform.python_version()
    settings["status"] = "prepared"

    # Hashes tell us whether two runs used exactly the same input files.
    settings["input_sha256"] = {}
    for name, path in (("data", args.data), ("paper", args.paper)):
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        settings["input_sha256"][name] = file_hash

    settings["package_versions"] = {}
    packages = (
        "numpy", "pandas", "matplotlib", "rank-bm25", "chromadb",
        "sentence-transformers", "google-genai", "python-dotenv",
    )
    for package in packages:
        try:
            settings["package_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass  # Preparation also works without these optional packages.
    return settings


def run_bm25(data, chunks, bm25, output, threshold):
    """Retrieve three chunks per question and measure reference coverage."""
    rows = []
    hit_count = 0
    coverage_scores = []
    for number, item in enumerate(data, 1):
        tokens = better_tokenize(item["question"])
        retrieved_chunks = bm25.get_top_n(tokens, chunks, n=3)
        context = "\n---\n".join(retrieved_chunks)
        hit, score = check_fuzzy_attribution(context, item["paper_reference"], threshold)
        if hit:
            hit_count += 1
        coverage_scores.append(score)
        rows.append({
            "question_number": number,
            "question": item["question"],
            "reference_coverage": score,
            "hit": hit,
            "context": context,
        })

    summary = {
        "questions": len(rows),
        "top_k": 3,
        "threshold": threshold,
        "reference_hit_rate_percent": 100 * hit_count / len(rows),
        "mean_reference_coverage": sum(coverage_scores) / len(rows),
    }
    write_json(output / "bm25_retrieval.json", rows)
    write_json(output / "bm25_summary.json", summary)
    print(json.dumps(summary, indent=2))


def run_gemini(data, chunks, bm25, output, args, api_key, model_name):
    # Import the model libraries only when full mode is requested.
    import experiment
    from google import genai
    from sentence_transformers import CrossEncoder

    request_count = 4 * len(data) * args.rounds
    print(f"Full run: {request_count} Gemini requests before retries; model={model_name}.")
    chroma_client, collection = experiment.build_vector_collection(chunks)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    retriever = experiment.Retriever(chunks, bm25, collection, reranker, args.alpha)

    with genai.Client(api_key=api_key) as client:
        # This small function lets the evaluation loop also run with a test model.
        def generate(prompt):
            response = client.models.generate_content(
                model=model_name, contents=prompt, config={"temperature": 0}
            )
            return response.text

        experiment.run_full_pipeline(
            data, retriever, generate, output,
            rounds=args.rounds, threshold=args.threshold, sleep=args.sleep,
        )


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    validate_arguments(args, parser)

    # 1. Read the questions and split the paper into searchable chunks.
    data = load_data(args.data)
    dataset_count = len(data)
    if args.limit is not None:
        data = data[:args.limit]
    paper_text = args.paper.read_text(encoding="utf-8")
    cleaned = clean_paper_text(paper_text)
    chunks = recursive_chunker(cleaned, args.chunk_size, args.overlap)
    if not chunks or not any(character.isalpha() for character in cleaned):
        raise ValueError("The paper contains no usable text after cleaning.")

    api_key = None
    model_name = None
    if args.mode == "full":
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        model_name = args.model or os.getenv("GEMINI_MODEL", "").strip()
        if not api_key or api_key == "YOUR_API_KEY_HERE" or not model_name:
            parser.error("full mode requires GEMINI_API_KEY and GEMINI_MODEL (or --model)")

    # 2. Save the prepared inputs in a new output directory.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output
    if output is None:
        output = ROOT / "outputs" / timestamp
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory is not empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "paper_clean.txt").write_text(cleaned + "\n", encoding="utf-8")
    write_json(output / "chunks.json", chunks)
    settings = build_run_info(args, output, timestamp, model_name, dataset_count, data, chunks)
    write_json(output / "run.json", settings)
    print(f"Validated {dataset_count} questions; selected {len(data)}; created {len(chunks)} chunks.")
    if args.mode == "prepare":
        print(f"Preparation complete (no network or model calls): {output}")
        return output

    # 3. Run the selected experiment and record whether it completed.
    try:
        from rank_bm25 import BM25Okapi
        tokenized_chunks = [better_tokenize(chunk) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        if args.mode == "bm25":
            run_bm25(data, chunks, bm25, output, args.threshold)
        else:
            run_gemini(data, chunks, bm25, output, args, api_key, model_name)
        settings["status"] = "completed"
    except Exception as error:
        settings["status"] = "failed"
        settings["error_type"] = type(error).__name__
        raise
    finally:
        write_json(output / "run.json", settings)
    print(f"Results saved: {output}")
    return output


if __name__ == "__main__":
    main()
