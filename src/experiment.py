"""Shared retrieval, evaluation and plotting for the four experiment methods."""

import time
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import (
    better_tokenize,
    check_fuzzy_attribution,
    compute_accuracy,
    normalize_scores,
    parse_mc_answer,
)

METHODS = ("bm25", "dense", "hybrid")


class Retriever:
    """Keep the indexes together and use one retrieval path for every metric."""

    def __init__(self, chunks, bm25, collection=None, reranker=None, alpha=0.5):
        self.chunks = chunks
        self.bm25 = bm25
        self.collection = collection
        self.reranker = reranker
        self.alpha = alpha

    def retrieve(self, question, method, *, top_k=3, dense_pool=15,
                 hybrid_dense_pool=50, hybrid_pool=50, rerank=True):
        if method not in METHODS:
            raise ValueError(f"Unknown retrieval method: {method}")
        tokens = better_tokenize(question)
        if method == "bm25":
            return self.bm25.get_top_n(tokens, self.chunks, n=top_k)

        if method == "dense":
            pool = dense_pool
        else:
            pool = hybrid_dense_pool
        result = self.collection.query(
            query_texts=[question], n_results=min(pool, len(self.chunks))
        )
        candidates = result["documents"][0]
        if method == "hybrid":
            candidates = self.combine_scores(tokens, result)
            if rerank:
                candidates = candidates[:hybrid_pool]

        if rerank and candidates:
            candidates = self.rerank_candidates(question, candidates)
        return candidates[:top_k]

    def combine_scores(self, tokens, dense_result):
        """Add the weighted dense and BM25 scores for each chunk."""
        # Chroma returns distances: smaller distances mean better matches.
        similarities = []
        for distance in dense_result["distances"][0]:
            similarities.append(1 - distance)
        dense_scores = normalize_scores(similarities)

        combined_scores = {}
        for chunk_id, score in zip(dense_result["ids"][0], dense_scores):
            combined_scores[int(chunk_id)] = self.alpha * score

        bm25_scores = self.bm25.get_scores(tokens)
        # argsort gives positions from lowest to highest score; reverse them.
        bm25_ids = np.argsort(bm25_scores)[::-1][:50]
        top_scores = []
        for chunk_id in bm25_ids:
            top_scores.append(bm25_scores[chunk_id])
        normalized_bm25 = normalize_scores(top_scores)

        for chunk_id, score in zip(bm25_ids, normalized_bm25):
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = 0
            combined_scores[chunk_id] += (1 - self.alpha) * score

        ordered_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)
        candidates = []
        for chunk_id in ordered_ids:
            candidates.append(self.chunks[chunk_id])
        return candidates

    def rerank_candidates(self, question, candidates):
        pairs = []
        for chunk in candidates:
            pairs.append([question, chunk])
        scores = self.reranker.predict(pairs)
        ordered_ids = np.argsort(scores)[::-1]

        ranked_chunks = []
        for chunk_id in ordered_ids:
            ranked_chunks.append(candidates[chunk_id])
        return ranked_chunks


def build_vector_collection(chunks):
    import chromadb
    from chromadb.utils import embedding_functions

    embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name=f"refrag_{uuid.uuid4().hex[:8]}", embedding_function=embedding,
        metadata={"hnsw:space": "cosine"})
    collection.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])
    return client, collection


def build_prompt(item, method, context=""):
    """Preserve the study's method-specific prompts, including their formatting."""
    question = item["question"]
    answers = item["answers"]
    if method == "baseline":
        options = "\n".join(f"{letter}){answers[letter]}" for letter in "ABCD")
        return f"""
Question:
{question}

Answer:
{options}

Answer only if you know the response, N/A if you do not. Answer with JUST one letter (A,B,C,D)
"""
    options = "\n".join(f"{letter}) {answers[letter]}" for letter in "ABCD")
    if method == "bm25":
        return f'''
Use ONLY the following piece of context to help you answer the question.

CONTEXT:
"""{context}"""


Question:
{question}

Answer:
{options}

Format your answer EXACTLY like this:
   Answer: [Letter A-D]
   Source: [Choose the most important information in the context above]
'''
    if method == "dense":
        reasoning = "One sentence explaining the evidence found"
    else:
        reasoning = "Brief explanation"
    return f'''
You are an expert AI researcher analyzing a technical paper.

TASK:
Answer the multiple-choice question based ONLY on the provided context.

CONTEXT:
"""{context}"""

QUESTION:
{question}

OPTIONS:
{options}

OUTPUT FORMAT:
Reasoning: [{reasoning}]
Answer: [Just the Letter A/B/C/D]
'''


def call_gemini(generate, prompt, max_retries=3, sleep=5):
    """Retry temporary failures; never turn failed API calls into accuracy scores."""
    if max_retries < 1:
        raise ValueError("max_retries must be positive")
    for attempt in range(max_retries):
        try:
            text = (generate(prompt) or "").strip()
        except Exception as error:
            permanent_error = getattr(error, "code", None) in (400, 401, 403, 404)
            last_attempt = attempt + 1 == max_retries
            if permanent_error or last_attempt:
                raise RuntimeError("Gemini request failed; check the model, API key and quota.") from None
            print(f"Gemini request failed ({type(error).__name__}); retrying.")
            time.sleep(sleep * (attempt + 1))
        else:
            time.sleep(sleep)
            return text


def evaluate_accuracy(data, retriever, generate, rounds=1, sleep=5):
    rows = []
    labels = [item["correct_answer"] for item in data]
    for round_number in range(1, rounds + 1):
        for method in ("baseline", "bm25", "dense", "hybrid"):
            predictions = []
            for item in data:
                context = ""
                if method != "baseline":
                    chunks = retriever.retrieve(item["question"], method)
                    context = "\n---\n".join(chunks)
                prompt = build_prompt(item, method, context)
                response = call_gemini(generate, prompt, sleep=sleep)
                predictions.append(parse_mc_answer(response))
            accuracy, correct, total, mistakes = compute_accuracy(predictions, labels)
            rows.append({"round": round_number, "method": method, "accuracy": accuracy})
            print(f"Round {round_number}, {method}: {correct}/{total} ({accuracy:.2f}%); mistakes: {mistakes}")
    return pd.DataFrame(rows)


def evaluate_sources(data, retriever, threshold=0.75):
    rows = []
    for method in METHODS:
        hits = []
        scores = []
        for item in data:
            chunks = retriever.retrieve(item["question"], method, top_k=5, dense_pool=20,
                                        hybrid_dense_pool=20, hybrid_pool=15)
            hit, score = check_fuzzy_attribution(" ".join(chunks), item["paper_reference"], threshold)
            hits.append(hit)
            scores.append(score)
        rows.append({"pipeline": method, "source_accuracy": 100 * np.mean(hits),
                     "avg_overlap_score": 100 * np.mean(scores)})
    return pd.DataFrame(rows)


def evaluate_recall(data, retriever, threshold=0.75, k_values=(1, 3, 5, 10)):
    hits = {}
    for method in METHODS:
        hits[method] = {}
        for k in k_values:
            hits[method][k] = 0
    for item in data:
        for method in METHODS:
            chunks = retriever.retrieve(item["question"], method, top_k=max(k_values),
                                        dense_pool=50, rerank=False)
            for k in k_values:
                hit, _ = check_fuzzy_attribution(" ".join(chunks[:k]), item["paper_reference"], threshold)
                if hit:
                    hits[method][k] += 1
    rows = []
    for k in k_values:
        bm25 = 100 * hits["bm25"][k] / len(data)
        dense = 100 * hits["dense"][k] / len(data)
        hybrid = 100 * hits["hybrid"][k] / len(data)
        # Preserve the original tie rule: Hybrid first, then BM25.
        if hybrid >= max(bm25, dense):
            winner = "Hybrid"
        elif dense > bm25:
            winner = "Dense"
        else:
            winner = "BM25"
        rows.append({"Métrica": f"Recall @ {k}", "BM25": f"{bm25:.1f}%", "Dense": f"{dense:.1f}%",
                     "Hybrid": f"{hybrid:.1f}%",
                     "Ganador": winner})
    return pd.DataFrame(rows)


def evaluate_overhead(data, retriever):
    rows = []
    means = []
    for method in METHODS:
        if method == "bm25":
            top_k = 3
            label = "BM25"
        else:
            top_k = 5
            label = method.title()
        context_lengths = []
        for item in data:
            chunks = retriever.retrieve(
                item["question"], method, top_k=top_k,
                hybrid_dense_pool=15, hybrid_pool=15,
            )
            context = "\n".join(chunks)
            context_lengths.append(len(context))
        mean_chars = np.mean(context_lengths)
        means.append(mean_chars)
        rows.append({"Pipeline": f"{label} (n={top_k})",
                     "Avg Caracteres": int(mean_chars), "Est. Tokens": int(mean_chars / 4)})
    for i, row in enumerate(rows):
        if i == 0:
            row["Coste Relativo"] = "1.0x (Base)"
        else:
            row["Coste Relativo"] = f"{means[i] / means[0]:.1f}x"
    return pd.DataFrame(rows)


def save_plot(output, name, ylabel):
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / f"{name}.png", dpi=150)
    plt.close()


def plot_results(results, output):
    summary = results["accuracy"].groupby("method")["accuracy"].agg(["mean", "std"])
    plt.figure(figsize=(7, 4))
    plt.bar(summary.index, summary["mean"], yerr=summary["std"].fillna(0))
    plt.title("Answer accuracy (mean ± sample standard deviation)")
    save_plot(output, "accuracy", "Accuracy (%)")

    source = results["source"]
    plt.figure(figsize=(7, 4))
    plt.bar(source["pipeline"], source["source_accuracy"])
    plt.title("Reference coverage above threshold")
    save_plot(output, "source_attribution", "Questions (%)")

    recall = results["recall"]
    k_values = recall["Métrica"].str.extract(r"(\d+)")[0].astype(int)
    plt.figure(figsize=(7, 4))
    for method in ("BM25", "Dense", "Hybrid"):
        plt.plot(k_values, recall[method].str.rstrip("%").astype(float), marker="o", label=method)
    plt.title("Reference hit rate at K (lexical proxy)")
    plt.xlabel("K")
    plt.legend()
    save_plot(output, "recall", "Questions (%)")

    overhead = results["overhead"]
    plt.figure(figsize=(7, 4))
    plt.bar(overhead["Pipeline"], overhead["Est. Tokens"])
    plt.title("Context size estimate (characters / 4)")
    save_plot(output, "overhead", "Estimated tokens")


def run_full_pipeline(data, retriever, generate, output, rounds=1, threshold=0.75, sleep=5):
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "accuracy": evaluate_accuracy(data, retriever, generate, rounds, sleep),
        "source": evaluate_sources(data, retriever, threshold),
        "recall": evaluate_recall(data, retriever, threshold),
        "overhead": evaluate_overhead(data, retriever),
    }
    for name, table in results.items():
        table.to_csv(output / f"{name}.csv", index=False)
        print(f"\n{name}\n{table.to_string(index=False)}")
    plot_results(results, output)
    return results
