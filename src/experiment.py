"""Original experiment functions, with duplicate definitions and notebook side effects removed.

Retrieval settings intentionally follow the submitted study; see README for
methodological limitations and docs/CHANGES.md for the preparation changes.
"""

import time
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import (better_tokenize, normalize_scores, parse_mc_answer,
                  compute_accuracy, check_fuzzy_attribution, check_fuzzy_match)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
SLEEP_SECONDS = 5


def call_gemini(model, prompt, max_retries=3, sleep=None):
    """Retry failures; stop on permanent errors instead of reporting false accuracy."""
    delay = SLEEP_SECONDS if sleep is None else sleep
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            text = (response.text or "").strip()
        except Exception as error:
            code = getattr(error, "code", None)
            if code in (400, 401, 403, 404) or attempt + 1 == max_retries:
                raise RuntimeError("Gemini request failed; check the model, API key and quota.") from None
            print(f"Gemini request failed ({type(error).__name__}); retrying.")
            time.sleep(delay * (attempt + 1))
        else:
            time.sleep(delay)
            return text
    raise ValueError("max_retries must be positive")


def finish_plot(name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=150)
    plt.close()


def build_prompt(question, answers):
    prompt = f"""
Question:
{question}

Answer:
A){answers['A']}
B){answers['B']}
C){answers['C']}
D){answers['D']}

Answer only if you know the response, N/A if you do not. Answer with JUST one letter (A,B,C,D)
"""
    return prompt


def build_rag_prompt(question, answers, context_text):
    prompt = f"""
Use ONLY the following piece of context to help you answer the question.

CONTEXT:
\"\"\"{context_text}\"\"\"


Question:
{question}

Answer:
A) {answers['A']}
B) {answers['B']}
C) {answers['C']}
D) {answers['D']}

Format your answer EXACTLY like this:
   Answer: [Letter A-D]
   Source: [Choose the most important information in the context above]
"""
    return prompt


def llm_baseline(data, model):
    model_answers = []
    true_answers = []

    for item in data:
        question = item["question"]
        answers = item["answers"]
        correct = item["correct_answer"]

        prompt = build_prompt(question, answers)
        text_response = call_gemini(model, prompt)
        model_letter = parse_mc_answer(text_response)

        model_answers.append(model_letter)
        true_answers.append(correct)

    accuracy, correct_count, total, mistakes = compute_accuracy(model_answers, true_answers)

    print()
    print(f"RESULTADO BASELINE")
    print(f"Aciertos: {correct_count} de {total}")
    print(f"Precisión Final: {accuracy}%")
    print(f"Fallos en preguntas: {mistakes}")

    return accuracy


def build_vector_collection(chunks, name="refrag_collection"):
    import chromadb
    from chromadb.utils import embedding_functions

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name=f"{name}_{uuid.uuid4().hex[:8]}",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])
    return client, collection


def llm_bm25_retrieval(data, chunks, model, bm25, top_n=3):
    model_answers = []
    true_answers = []
    sources_found = []

    for i, item in enumerate(data):
        question = item["question"]
        answers = item["answers"]
        correct = item["correct_answer"]

        tokenized_query = better_tokenize(question)
        top_chunks = bm25.get_top_n(tokenized_query, chunks, n=top_n)
        best_context = "\n---\n".join(top_chunks)
        prompt = build_rag_prompt(question, answers, best_context)
        text_response = call_gemini(model, prompt)

        # Extraemos letra y fuente
        model_letter = parse_mc_answer(text_response)
        match_source = re.search(r'Source:\s*(.*)', text_response, re.DOTALL | re.IGNORECASE)
        source_text = match_source.group(1).strip() if match_source else "No source provided"

        model_answers.append(model_letter)
        true_answers.append(correct)
        sources_found.append(source_text)

    accuracy, correct_count, total, mistakes = compute_accuracy(model_answers, true_answers)

    print()
    print(f"RESULTADO BM25")
    print(f"Aciertos: {correct_count} de {total}")
    print(f"Precisión Final: {accuracy}%")
    print(f"Fallos en preguntas: {mistakes}")

    return accuracy


def llm_dense_retrieval(data, chunks, model, collection, reranker, top_pool=15, top_final=3):
    print("  INICIANDO DENSE RETRIEVAL ")

    model_answers = []
    true_answers = []

    for i, item in enumerate(data):
        question = item["question"]
        answers = item["answers"]
        correct = item["correct_answer"]

        # 1) Dense retrieval (recall grande)
        results = collection.query(
            query_texts=[question],
            n_results=min(top_pool, len(chunks))
        )
        candidates = results['documents'][0]

        # 2) Re-ranking
        pairs = [[question, doc] for doc in candidates]
        scores = reranker.predict(pairs)
        sorted_indices = np.argsort(scores)[::-1]
        top_indices = sorted_indices[:top_final]
        top_chunks = [candidates[idx] for idx in top_indices]

        context_text = "\n---\n".join(top_chunks)

        # 3) Prompt al LLM
        prompt = f"""
You are an expert AI researcher analyzing a technical paper.

TASK:
Answer the multiple-choice question based ONLY on the provided context.

CONTEXT:
\"\"\"{context_text}\"\"\"

QUESTION:
{question}

OPTIONS:
A) {answers['A']}
B) {answers['B']}
C) {answers['C']}
D) {answers['D']}

OUTPUT FORMAT:
Reasoning: [One sentence explaining the evidence found]
Answer: [Just the Letter A/B/C/D]
"""
        text_response = call_gemini(model, prompt)
        prediction = parse_mc_answer(text_response)

        model_answers.append(prediction)
        true_answers.append(correct)

    accuracy, correct_count, total, mistakes = compute_accuracy(model_answers, true_answers)

    print("\n")
    print(f"RESULTADO DENSE RETRIEVAL")
    print(f"Aciertos: {correct_count} de {total}")
    print(f"Precisión Final: {accuracy}%")
    print(f"Preguntas falladas (índices): {mistakes}")

    return accuracy


def llm_hybrid_retrieval(data, chunks, model, bm25, collection, reranker, alpha=0.5, pool_size=50, top_final=3):
    print(f" INICIANDO HYBRID RETRIEVAL (Alpha={alpha}) ")

    model_answers = []
    true_answers = []

    for i, item in enumerate(data):
        question = item["question"]
        answers = item["answers"]
        correct = item["correct_answer"]

        # Resultados del BM25
        tokenized_query = better_tokenize(question)
        bm25_scores_all = bm25.get_scores(tokenized_query)
        top_bm25_idx = np.argsort(bm25_scores_all)[::-1][:pool_size]
        bm25_subset_scores = [bm25_scores_all[x] for x in top_bm25_idx]
        bm25_norm = normalize_scores(bm25_subset_scores)

        # Resultados del Dense retrieval
        results = collection.query(query_texts=[question], n_results=min(pool_size, len(chunks)))
        dense_ids = [int(id_str) for id_str in results['ids'][0]]
        dense_distances = results['distances'][0]
        dense_candidates = results['documents'][0]
        dense_scores_raw = [1 - d for d in dense_distances]
        dense_norm = normalize_scores(dense_scores_raw)

        #  Hybrid
        hybrid_map = {}

        # Aporte Dense
        for idx, score in zip(dense_ids, dense_norm):
            hybrid_map[idx] = hybrid_map.get(idx, 0) + score * alpha

        # Aporte BM25
        for idx, score in zip(top_bm25_idx, bm25_norm):
            hybrid_map[idx] = hybrid_map.get(idx, 0) + score * (1 - alpha)

        # Ordenar candidatos Hybrid
        sorted_hybrid = sorted(hybrid_map.items(), key=lambda x: x[1], reverse=True)
        top_h_indices = [idx for idx, _ in sorted_hybrid[:pool_size]]
        hybrid_chunks = [chunks[idx] for idx in top_h_indices]

        # Ranking final
        if hybrid_chunks:
            pairs = [[question, doc] for doc in hybrid_chunks]
            scores = reranker.predict(pairs)
            sorted_indices = np.argsort(scores)[::-1][:top_final]
            best_chunks = [hybrid_chunks[x] for x in sorted_indices]
        else:
            best_chunks = []

        context_text = "\n---\n".join(best_chunks)

        prompt = f"""
You are an expert AI researcher analyzing a technical paper.

TASK:
Answer the multiple-choice question based ONLY on the provided context.

CONTEXT:
\"\"\"{context_text}\"\"\"

QUESTION:
{question}

OPTIONS:
A) {answers['A']}
B) {answers['B']}
C) {answers['C']}
D) {answers['D']}

OUTPUT FORMAT:
Reasoning: [Brief explanation]
Answer: [Just the Letter A/B/C/D]
"""
        text_response = call_gemini(model, prompt)
        prediction = parse_mc_answer(text_response)

        model_answers.append(prediction)
        true_answers.append(correct)

    accuracy, correct_count, total, mistakes = compute_accuracy(model_answers, true_answers)

    print()
    print(f"RESULTADO HYBRID RETRIEVAL (Alpha={alpha})")
    print(f"Aciertos: {correct_count} de {total}")
    print(f"Precisión Final: {accuracy}%")
    print(f"Fallos en preguntas: {mistakes}")

    return accuracy


def evaluate_source_attribution_complete(data, chunks, bm25, collection, reranker, threshold=0.75, alpha=0.5):
    print(f"  INICIANDO COMPARATIVA DE FUENTES (3 PIPELINES)")
    print(f"   Umbral de Coincidencia: {threshold*100}% | Alpha Híbrido: {alpha}")

    stats = {
        "bm25": {"hits": 0, "scores": []},
        "dense": {"hits": 0, "scores": []},
        "hybrid": {"hits": 0, "scores": []}
    }

    total = len(data)

    for i, item in enumerate(data):
        question = item["question"]
        reference = item["paper_reference"]
        tokenized_query = better_tokenize(question)

        # BM25
        top_chunks_bm25 = bm25.get_top_n(tokenized_query, chunks, n=5)
        match, score = check_fuzzy_attribution(" ".join(top_chunks_bm25), reference, threshold)
        if match: stats["bm25"]["hits"] += 1
        stats["bm25"]["scores"].append(score)

        # Dense
        dense_res = collection.query(query_texts=[question], n_results=min(20, len(chunks)))
        dense_cands = dense_res['documents'][0]
        pairs = [[question, doc] for doc in dense_cands]
        rr_scores = reranker.predict(pairs)
        top_d_idx = np.argsort(rr_scores)[::-1][:5]
        top_chunks_dense = [dense_cands[x] for x in top_d_idx]
        match, score = check_fuzzy_attribution(" ".join(top_chunks_dense), reference, threshold)
        if match: stats["dense"]["hits"] += 1
        stats["dense"]["scores"].append(score)

        # Hybrid
        dense_ids = [int(id_str) for id_str in dense_res['ids'][0]]
        dense_raw_scores = [1 - d for d in dense_res['distances'][0]]
        dense_norm = normalize_scores(dense_raw_scores)

        bm25_scores_raw = bm25.get_scores(tokenized_query)
        top_50_bm25_idx = np.argsort(bm25_scores_raw)[::-1][:50]
        bm25_norm_subset = normalize_scores([bm25_scores_raw[x] for x in top_50_bm25_idx])

        hybrid_map = {}
        for idx, s in zip(dense_ids, dense_norm):
            hybrid_map[idx] = hybrid_map.get(idx, 0) + s * alpha
        for idx, s in zip(top_50_bm25_idx, bm25_norm_subset):
            hybrid_map[idx] = hybrid_map.get(idx, 0) + s * (1 - alpha)

        sorted_hybrid = sorted(hybrid_map.items(), key=lambda x: x[1], reverse=True)[:15]
        hybrid_cands = [chunks[idx] for idx, _ in sorted_hybrid]
        if hybrid_cands:
            h_pairs = [[question, doc] for doc in hybrid_cands]
            h_rr_scores = reranker.predict(h_pairs)
            top_h_idx = np.argsort(h_rr_scores)[::-1][:5]
            top_chunks_hybrid = [hybrid_cands[x] for x in top_h_idx]
        else:
            top_chunks_hybrid = []
        match, score = check_fuzzy_attribution(" ".join(top_chunks_hybrid), reference, threshold)
        if match: stats["hybrid"]["hits"] += 1
        stats["hybrid"]["scores"].append(score)

        if (i+1) % 10 == 0:
            print(f"   Procesado {i+1}/{total} preguntas...")

    rows = []
    best_acc = 0
    best_pipe = ""

    print("\n" + "="*60)
    print(" RESULTADOS FINALES: SOURCE ATTRIBUTION ACCURACY")
    print("="*60)
    print(f"{'PIPELINE':<20} | {'PRECISIÓN':<10} | {'COINCIDENCIA PROMEDIO'}")
    print("-" * 60)

    for pipe in ["bm25", "dense", "hybrid"]:
        acc = (stats[pipe]["hits"] / total) * 100
        avg_score = np.mean(stats[pipe]["scores"]) * 100
        print(f"{pipe.upper():<20} | {acc:6.2f}%    | {avg_score:6.1f}%")

        rows.append({
            "pipeline": pipe,
            "source_accuracy": acc,
            "avg_overlap_score": avg_score
        })

        if acc > best_acc:
            best_acc = acc
            best_pipe = pipe.upper()

    print("-" * 60)
    print(f" GANADOR: {best_pipe} con un {best_acc:.2f}% de precisión de fuente.")

    if best_acc < 70:
        print("\n NOTA: Los resultados siguen siendo bajos. Considera:")
        print("   1. Revisar manualmente la evidencia; no ajustar el umbral al test.")
        print("   2. Revisar si tus chunks están cortando las frases clave.")

    df_source = pd.DataFrame(rows)
    return df_source


def plot_source_attribution_dashboard(df_source):
    plt.figure(figsize=(6,4))
    plt.bar(df_source["pipeline"], df_source["source_accuracy"])
    plt.title("Source Attribution Accuracy (%) por pipeline")
    plt.xlabel("Pipeline")
    plt.ylabel("Accuracy de fuente (%)")
    plt.grid(axis="y", alpha=0.3)
    finish_plot("source_attribution")

    print(df_source.to_string(index=False))


def calculate_recall_metrics_complete(data, chunks, bm25, collection, k_values=(1,3,5,10), alpha=0.5, threshold=0.75):
    print(f" INICIANDO AUDITORÍA DE RECALL 3 VÍAS (K={k_values})")

    hits = {
        'bm25': {k: 0 for k in k_values},
        'dense': {k: 0 for k in k_values},
        'hybrid': {k: 0 for k in k_values}
    }

    total_questions = len(data)
    max_pool_size = 50

    for i, item in enumerate(data):
        question = item["question"]
        target_ref = item["paper_reference"]

        tokenized_query = better_tokenize(question)

        # BM25
        bm25_scores_all = bm25.get_scores(tokenized_query)
        max_k = max(k_values)
        top_bm25_idx = np.argsort(bm25_scores_all)[::-1][:max_k]
        top_bm25_chunks = [chunks[x] for x in top_bm25_idx]
        for k in k_values:
            context_at_k = " ".join(top_bm25_chunks[:k])
            if check_fuzzy_match(context_at_k, target_ref, threshold): hits['bm25'][k] += 1

        # Dense
        results = collection.query(query_texts=[question], n_results=min(max_pool_size, len(chunks)))
        dense_candidates = results['documents'][0]
        for k in k_values:
            limit = min(k, len(dense_candidates))
            context_at_k = " ".join(dense_candidates[:limit])
            if check_fuzzy_match(context_at_k, target_ref, threshold): hits['dense'][k] += 1

        # Hybrid
        dense_ids = [int(id_str) for id_str in results['ids'][0]]
        dense_distances = results['distances'][0]
        dense_scores_raw = [1 - d for d in dense_distances]
        dense_scores_norm = normalize_scores(dense_scores_raw)

        hybrid_map = {}
        for idx, val in zip(dense_ids, dense_scores_norm):
            hybrid_map[idx] = hybrid_map.get(idx, 0) + (val * alpha)

        top_50_bm25_idx = np.argsort(bm25_scores_all)[::-1][:max_pool_size]
        bm25_subset_scores = [bm25_scores_all[x] for x in top_50_bm25_idx]
        bm25_scores_norm = normalize_scores(bm25_subset_scores)

        for idx, val in zip(top_50_bm25_idx, bm25_scores_norm):
            hybrid_map[idx] = hybrid_map.get(idx, 0) + (val * (1-alpha))

        sorted_hybrid = sorted(hybrid_map.items(), key=lambda x: x[1], reverse=True)

        for k in k_values:
            top_k_indices = [idx for idx, _ in sorted_hybrid[:k]]
            top_k_chunks = [chunks[idx] for idx in top_k_indices]
            context_at_k = " ".join(top_k_chunks)
            if check_fuzzy_match(context_at_k, target_ref, threshold): hits['hybrid'][k] += 1

        if (i+1) % 10 == 0:
            print(f"Procesado {i+1}/{total_questions} preguntas")

    results_table = []
    for k in k_values:
        rec_b = (hits['bm25'][k] / total_questions) * 100
        rec_d = (hits['dense'][k] / total_questions) * 100
        rec_h = (hits['hybrid'][k] / total_questions) * 100

        results_table.append({
            "Métrica": f"Recall @ {k}",
            "BM25": f"{rec_b:.1f}%",
            "Dense": f"{rec_d:.1f}%",
            "Hybrid": f"{rec_h:.1f}%",
            "Ganador": "Hybrid" if rec_h >= max(rec_b, rec_d) else ("Dense" if rec_d > rec_b else "BM25")
        })

    df = pd.DataFrame(results_table)

    print("\n" + "="*60)
    print(f" RESULTADOS COMPARATIVOS: RECALL (Alpha={alpha})")
    print("="*60)
    print(df.to_string(index=False))

    return df


def plot_recall_dashboard(metrics_recall_df):
    """
    Espera el DF de calculate_recall_metrics_complete.
    Columnas: ['Métrica','BM25','Dense','Hybrid','Ganador']
    """
    df = metrics_recall_df.copy()
    df["K"] = df["Métrica"].str.extract(r'(\d+)').astype(int)

    plt.figure(figsize=(7,4))
    plt.plot(df["K"], df["BM25"].str.rstrip('%').astype(float), marker="o", label="BM25")
    plt.plot(df["K"], df["Dense"].str.rstrip('%').astype(float), marker="o", label="Dense")
    plt.plot(df["K"], df["Hybrid"].str.rstrip('%').astype(float), marker="o", label="Hybrid")
    plt.title("Recall@K por método")
    plt.xlabel("K")
    plt.ylabel("Recall (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    finish_plot("recall")

    print(df.to_string(index=False))


def calculate_context_overhead_complete(data, chunks, bm25, collection, reranker, alpha=0.5):
    print(" MIDIENDO CONTEXT OVERHEAD (3 PIPELINES)")

    stats = {
        'bm25': {'chars': [], 'tokens': []},
        'dense': {'chars': [], 'tokens': []},
        'hybrid': {'chars': [], 'tokens': []}
    }

    for i, item in enumerate(data):
        query = item["question"]
        tokenized_query = better_tokenize(query)

        # BM25 (Top 3)
        top_chunks_bm25 = bm25.get_top_n(tokenized_query, chunks, n=3)
        context_bm25 = "\n".join(top_chunks_bm25)
        stats['bm25']['chars'].append(len(context_bm25))
        stats['bm25']['tokens'].append(len(context_bm25)/4)

        # Dense (Top 5 final)
        d_res = collection.query(query_texts=[query], n_results=min(15, len(chunks)))
        d_cands = d_res['documents'][0]
        d_pairs = [[query, doc] for doc in d_cands]
        d_scores = reranker.predict(d_pairs)
        d_top_idx = np.argsort(d_scores)[::-1][:5]
        context_dense = "\n".join([d_cands[x] for x in d_top_idx])
        stats['dense']['chars'].append(len(context_dense))
        stats['dense']['tokens'].append(len(context_dense)/4)

        # Hybrid (Top 5 final)
        bm25_all_scores = bm25.get_scores(tokenized_query)
        d_ids = [int(x) for x in d_res['ids'][0]]
        d_raw_scores = [1-d for d in d_res['distances'][0]]

        hybrid_map = {}
        d_norm = normalize_scores(d_raw_scores)
        for idx, val in zip(d_ids, d_norm):
            hybrid_map[idx] = hybrid_map.get(idx, 0) + (val * alpha)

        top_50_bm25 = np.argsort(bm25_all_scores)[::-1][:50]
        b_norm = normalize_scores([bm25_all_scores[x] for x in top_50_bm25])
        for idx, val in zip(top_50_bm25, b_norm):
            hybrid_map[idx] = hybrid_map.get(idx, 0) + (val * (1-alpha))

        sorted_h = sorted(hybrid_map.items(), key=lambda x: x[1], reverse=True)[:15]
        h_chunks = [chunks[idx] for idx, _ in sorted_h]

        if h_chunks:
            h_pairs = [[query, doc] for doc in h_chunks]
            h_scores = reranker.predict(h_pairs)
            h_top_idx = np.argsort(h_scores)[::-1][:5]
            context_hybrid = "\n".join([h_chunks[x] for x in h_top_idx])
        else:
            context_hybrid = ""

        stats['hybrid']['chars'].append(len(context_hybrid))
        stats['hybrid']['tokens'].append(len(context_hybrid)/4)

    df = pd.DataFrame({
        'Pipeline': ['BM25 (n=3)', 'Dense (n=5)', 'Hybrid (n=5)'],
        'Avg Caracteres': [
            int(np.mean(stats['bm25']['chars'])),
            int(np.mean(stats['dense']['chars'])),
            int(np.mean(stats['hybrid']['chars']))
        ],
        'Est. Tokens': [
            int(np.mean(stats['bm25']['tokens'])),
            int(np.mean(stats['dense']['tokens'])),
            int(np.mean(stats['hybrid']['tokens']))
        ]
    })

    base_tokens = np.mean(stats['bm25']['tokens'])
    df['Coste Relativo'] = [
        "1.0x (Base)",
        f"{np.mean(stats['dense']['tokens'])/base_tokens:.1f}x",
        f"{np.mean(stats['hybrid']['tokens'])/base_tokens:.1f}x"
    ]

    print("\n" + "="*60)
    print(" RESULTADOS: CONTEXT OVERHEAD (Eficiencia)")
    print("="*60)
    print(df.to_string(index=False))
    print("-" * 60)

    return df


def plot_overhead_dashboard(df_overhead):
    plt.figure(figsize=(7,4))
    plt.bar(df_overhead["Pipeline"], df_overhead["Est. Tokens"])
    plt.title("Context overhead (tokens promedio enviados al LLM)")
    plt.xlabel("Pipeline")
    plt.ylabel("Tokens estimados")
    plt.grid(axis="y", alpha=0.3)
    finish_plot("overhead")

    print(df_overhead.to_string(index=False))


def run_experiments(
    data,
    chunks,
    model,
    bm25,
    chroma_collection,
    reranker,
    rounds=1,
    alpha=0.5
):
    accuracy_results = []

    for r in range(rounds):
        print(f"\n=== ROUND {r+1}/{rounds} ===")

        # Baseline
        acc_base = llm_baseline(data, model)

        # BM25
        acc_bm25 = llm_bm25_retrieval(data, chunks, model, bm25)

        # Dense
        acc_dense = llm_dense_retrieval(data, chunks, model, chroma_collection, reranker)

        # Hybrid
        acc_hybrid = llm_hybrid_retrieval(data, chunks, model, bm25, chroma_collection, reranker, alpha)

        accuracy_results.append({
            "round": r+1,
            "baseline": acc_base,
            "bm25": acc_bm25,
            "dense": acc_dense,
            "hybrid": acc_hybrid
        })

    import pandas as pd
    df = pd.DataFrame(accuracy_results)
    print("\n=== ACCURACY SUMMARY ===")
    print(df)
    return df


def summarize_accuracy(df_accuracy):
    df_accuracy = ensure_long_accuracy(df_accuracy)
    summary = (
        df_accuracy
        .groupby("method")["accuracy"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    return summary


def plot_accuracy_dashboard(df_accuracy):
    df_accuracy = ensure_long_accuracy(df_accuracy)
    summary = summarize_accuracy(df_accuracy)

    plt.figure(figsize=(6,4))
    plt.bar(summary["method"], summary["mean"], yerr=summary["std"].fillna(0))
    plt.title("Accuracy medio por método (± std)")
    plt.xlabel("Método")
    plt.ylabel("Accuracy (%)")
    plt.grid(axis="y", alpha=0.3)
    finish_plot("accuracy")

    print(summary.to_string(index=False))


def ensure_long_accuracy(df_accuracy):
    """
    Si df_accuracy está en formato ancho (columnas = baseline, bm25, ...),
    lo convierte a formato largo con columnas: round, method, accuracy.
    Si ya está en formato largo, lo devuelve tal cual.
    """
    cols = df_accuracy.columns.tolist()

    # Caso 1: ya está en formato largo
    if "method" in cols and "accuracy" in cols:
        return df_accuracy

    # Caso 2: formato ancho tipo: round, baseline, bm25, dense, hybrid
    id_vars = [c for c in cols if c == "round"]
    value_vars = [c for c in cols if c != "round"]

    df_long = df_accuracy.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="method",
        value_name="accuracy"
    )
    return df_long


def run_full_pipeline(
    data,
    chunks,
    model,
    bm25,
    chroma_collection,
    reranker,
    rounds=1,
    alpha=0.5,
    threshold=0.75
):

    print(" INICIANDO PIPELINE COMPLETO")
    # 1) Accuracy Experiments
    print(" Ejecutando experimentos de Precisión\n")
    df_accuracy = run_experiments(
        data,
        chunks,
        model,
        bm25,
        chroma_collection,
        reranker,
        rounds=rounds,
        alpha=alpha
    )
    # Pasar a formato largo (round, method, accuracy)
    df_accuracy_long = ensure_long_accuracy(df_accuracy)
    print("\n Precisión calculada.\n")

    # 2) Source Attribution
    print(" Calculando Source Atribution\n")
    df_source = evaluate_source_attribution_complete(
        data,
        chunks,
        bm25,
        chroma_collection,
        reranker,
        threshold=threshold,
        alpha=alpha
    )
    print("\n Source Attribution calculado.\n")

    # 3) Recall@K
    print("Calculando RECALL @ K\n")
    df_recall = calculate_recall_metrics_complete(
        data,
        chunks,
        bm25,
        chroma_collection,
        k_values=[1,3,5,10],
        alpha=alpha,
        threshold=threshold
    )
    print("\n Recall calculado.\n")

    # 4) Context Overhead
    print(" Calculando Context Overhead\n")
    df_overhead = calculate_context_overhead_complete(
        data,
        chunks,
        bm25,
        chroma_collection,
        reranker,
        alpha=alpha
    )
    print("\n Context Overhead calculado.\n")

    # 5) Dashboards Visuales
    print(" Generando dashboards\n")
    plot_accuracy_dashboard(df_accuracy_long)
    plot_source_attribution_dashboard(df_source)
    plot_recall_dashboard(df_recall)
    plot_overhead_dashboard(df_overhead)

    print("\n Dashboards generados.\n")

    # 6) Documento de conclusiones
    print(" Generando CONCLUSIONES automáticas\n")

    conclusions = f"""
 CONCLUSIONES AUTOMÁTICAS:

> Comparativa de métodos basada en {rounds} rondas:

Accuracy promedio:
{df_accuracy_long.groupby('method')['accuracy'].mean().to_string()}

Source Attribution (% coincidencia):
{df_source.to_string(index=False)}

Recall@K:
{df_recall.to_string(index=False)}

Context Overhead (tokens / coste relativo):
{df_overhead.to_string(index=False)}
"""

    with open(OUTPUT_DIR / "CONCLUSIONES_PIPELINE.txt", "w", encoding="utf-8") as f:
        f.write(conclusions)

    return {
        "accuracy": df_accuracy_long,
        "source": df_source,
        "recall": df_recall,
        "overhead": df_overhead,
    }
