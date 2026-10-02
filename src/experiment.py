"""Shared retrieval, evaluation and plotting for the four experiment methods."""

import time
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import better_tokenize, check_fuzzy_attribution, compute_accuracy, normalize_scores, parse_mc_answer

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

        pool = dense_pool if method == "dense" else hybrid_dense_pool
        result = self.collection.query(query_texts=[question], n_results=min(pool, len(self.chunks)))
        candidates = result["documents"][0]
        if method == "hybrid":
            dense_scores = normalize_scores([1 - distance for distance in result["distances"][0]])
            scores = {int(idx): self.alpha * score for idx, score in zip(result["ids"][0], dense_scores)}
            bm25_scores = self.bm25.get_scores(tokens)
            bm25_ids = np.argsort(bm25_scores)[::-1][:50]
            for idx, score in zip(bm25_ids, normalize_scores([bm25_scores[i] for i in bm25_ids])):
                scores[idx] = scores.get(idx, 0) + (1 - self.alpha) * score
            ordered = sorted(scores, key=scores.get, reverse=True)
            if rerank:
                ordered = ordered[:hybrid_pool]
            candidates = [self.chunks[idx] for idx in ordered]

        if rerank and candidates:
            scores = self.reranker.predict([[question, chunk] for chunk in candidates])
            candidates = [candidates[idx] for idx in np.argsort(scores)[::-1]]
        return candidates[:top_k]


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
    question, answers = item["question"], item["answers"]
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
    reasoning = "One sentence explaining the evidence found" if method == "dense" else "Brief explanation"
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
    for attempt in range(max_retries):
        try:
            text = (generate(prompt) or "").strip()
        except Exception as error:
            if getattr(error, "code", None) in (400, 401, 403, 404) or attempt + 1 == max_retries:
                raise RuntimeError("Gemini request failed; check the model, API key and quota.") from None
            print(f"Gemini request failed ({type(error).__name__}); retrying.")
            time.sleep(sleep * (attempt + 1))
        else:
            time.sleep(sleep)
            return text
    raise ValueError("max_retries must be positive")


def evaluate_accuracy(data, retriever, generate, rounds=1, sleep=5):
    rows = []
    labels = [item["correct_answer"] for item in data]
    for round_number in range(1, rounds + 1):
        for method in ("baseline", *METHODS):
            predictions = []
            for item in data:
                chunks = [] if method == "baseline" else retriever.retrieve(item["question"], method)
                prompt = build_prompt(item, method, "\n---\n".join(chunks))
                predictions.append(parse_mc_answer(call_gemini(generate, prompt, sleep=sleep)))
            accuracy, correct, total, mistakes = compute_accuracy(predictions, labels)
            rows.append({"round": round_number, "method": method, "accuracy": accuracy})
            print(f"Round {round_number}, {method}: {correct}/{total} ({accuracy:.2f}%); mistakes: {mistakes}")
    return pd.DataFrame(rows)


def evaluate_sources(data, retriever, threshold=0.75):
    rows = []
    for method in METHODS:
        hits, scores = [], []
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
    hits = {method: {k: 0 for k in k_values} for method in METHODS}
    for item in data:
        for method in METHODS:
            chunks = retriever.retrieve(item["question"], method, top_k=max(k_values),
                                        dense_pool=50, rerank=False)
            for k in k_values:
                hit, _ = check_fuzzy_attribution(" ".join(chunks[:k]), item["paper_reference"], threshold)
                hits[method][k] += hit
    rows = []
    for k in k_values:
        bm25, dense, hybrid = [100 * hits[method][k] / len(data) for method in METHODS]
        rows.append({"Métrica": f"Recall @ {k}", "BM25": f"{bm25:.1f}%", "Dense": f"{dense:.1f}%",
                     "Hybrid": f"{hybrid:.1f}%",
                     "Ganador": "Hybrid" if hybrid >= max(bm25, dense) else "Dense" if dense > bm25 else "BM25"})
    return pd.DataFrame(rows)


def evaluate_overhead(data, retriever):
    rows, means = [], []
    for method in METHODS:
        top_k = 3 if method == "bm25" else 5
        contexts = ["\n".join(retriever.retrieve(item["question"], method, top_k=top_k,
                    hybrid_dense_pool=15, hybrid_pool=15)) for item in data]
        mean_chars = np.mean([len(context) for context in contexts])
        means.append(mean_chars)
        rows.append({"Pipeline": f"{method.upper() if method == 'bm25' else method.title()} (n={top_k})",
                     "Avg Caracteres": int(mean_chars), "Est. Tokens": int(mean_chars / 4)})
    for i, row in enumerate(rows):
        row["Coste Relativo"] = "1.0x (Base)" if i == 0 else f"{means[i] / means[0]:.1f}x"
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
