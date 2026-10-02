# MPE 25/26 — RAG Benchmark for Technical Q&A

A small academic experiment comparing **BM25, dense retrieval and hybrid retrieval** for multiple-choice questions about a technical paper. A Gemini baseline answers the same questions without retrieved context.

The project was developed for *Modelización de Problemas de la Empresa* (2025/26) by **Shadi Fedriani Abdallah and Alejandro Domínguez López**. The original [report and results](docs/Report%20%26%20Results.pdf) are in Spanish. The dataset, source paper and this README are in English; many original code comments and plot labels remain in Spanish.

The knowledge base is *REFRAG: Rethinking RAG based Decoding*. This repository evaluates retrieval over that paper; it does **not** implement REFRAG's compressed decoding architecture.

## What is included

| File | Purpose |
| --- | --- |
| `src/Script.py` | Command-line entry point, input validation, configuration and output files |
| `src/core.py` | Cleaning, bounded word chunking, answer parsing and lexical metrics |
| `src/experiment.py` | Original retrieval experiments and dashboards, with duplicate definitions removed |
| `notebooks/Code.ipynb` | Short notebook using the same code as the command line |
| `data/ModelizaciónEmpresaUCMData.json` | 70 questions, each with A–D options, a correct answer and reference text |
| `docs/Paper.txt` | Supplied text extraction of the REFRAG paper |
| `docs/Report & Results.pdf` | Unmodified academic report |
| `tests/` | Local regression checks and an experiment integration test using test doubles |
| `.github/workflows/checks.yml` | Automated checks without Gemini calls or neural model downloads |

The supplied archive did not include `Paper.pdf`. The runnable version uses `docs/Paper.txt` directly, so PDF extraction is not a prerequisite. The report PDF is a separate document, not the paper.

## Quick start: no API key needed

Use Python 3.11. Run commands from the repository root unless an absolute script path is used.

```bash
python src/Script.py --mode prepare
```

This command uses only the Python standard library. It validates all 70 records, cleans the paper, creates chunks and writes a timestamped folder under `outputs/`. It does not contact an API or download a model. Running `python src/Script.py` has the same safe default.

For local tests and a BM25 retrieval run, first create an environment:

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-check.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe src/Script.py --mode bm25
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-check.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python src/Script.py --mode bm25
```

Once dependencies are installed, BM25 runs locally. Its output measures reference coverage in the top three retrieved chunks. It does **not** produce LLM answer accuracy.

## Run the full experiment

1. Install `requirements.txt` using the environment's Python executable.
2. Copy `.env.example` to `.env` at the repository root.
3. Set `GEMINI_API_KEY` and a `GEMINI_MODEL` available to your account.
4. Start with three questions and one round:

```bash
python -m pip install -r requirements.txt
python src/Script.py --mode full --limit 3 --rounds 1
```

Use `.venv/bin/python` or `.\.venv\Scripts\python.exe` instead of `python` if the environment is not activated. On PowerShell, copy the configuration with `Copy-Item .env.example .env`; on macOS/Linux, use `cp .env.example .env`.

The full mode downloads `all-mpnet-base-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2` on first use. It creates a local in-memory Chroma collection and sends question/option text, plus retrieved context for RAG methods, to Google's Gemini API. The key stays in the environment and is not written to run metadata. `.env` is ignored by Git.

The prepared code uses the [Google Gen AI Python SDK](https://github.com/googleapis/python-genai). `gemini-2.5-flash-lite` is the historical model in the supplied project, not a guarantee of present account access. Check [model availability and deprecations](https://ai.google.dev/gemini-api/docs/deprecations) and set `--model MODEL_ID` when necessary. Using another model creates a new experiment. Quotas and charges depend on your account; no daily allowance is assumed here.

A full run uses **4 × questions × rounds** Gemini calls before retries: 280 calls for 70 questions and one round, or 560 for two rounds. The default five-second pause alone adds at least 23 minutes 20 seconds to a 280-call run; generation, downloads, embeddings and reranking take additional time. CI never performs this run.

```bash
# Historical number of rounds, with the prepared implementation:
python src/Script.py --mode full --rounds 2

# Inspect all available options:
python src/Script.py --help
```

## How the pipeline works

1. **Validate the dataset.** Each item must contain a non-empty question, exactly four labelled options, a valid correct-answer letter and reference text. Duplicate questions are rejected.
2. **Clean the paper.** Join extracted lines, remove numeric-only lines and remove text between the case-sensitive `References` and `Appendix` markers. If no closing marker is found, remove the remaining reference section. This is a document-specific heuristic; inspect `paper_clean.txt` when changing the corpus.
3. **Chunk the text.** Pack paragraphs, then sentences, targeting at most 250 **words**, with up to 30 words of overlap. Long spans fall back to word splitting. This is paragraph/sentence-aware chunking, not an embedding-based semantic boundary detector.
4. **Build retrieval indexes.** BM25 uses normalized lexical tokens; dense retrieval uses sentence embeddings and cosine distance in Chroma.
5. **Answer the questions.** Run a context-free baseline, BM25, dense retrieval with a cross-encoder reranker, and weighted hybrid retrieval with the same reranker.
6. **Evaluate and save.** Write tables, dashboard PNGs, run settings and conclusions into a new output directory.

The baseline sees only the question and options. RAG prompts also receive retrieved chunks. Gold answers and `paper_reference` are used for evaluation, not inserted into prompts or used to rank chunks.

### Retrieval settings

The following settings preserve the original experiment functions. They intentionally remain visible because different metrics use different context budgets.

| Stage | BM25 | Dense | Hybrid |
| --- | --- | --- | --- |
| Answer accuracy | Top 3 | Top 15, rerank to 3 | Fuse top 50 from each retriever, retain 50, rerank to 3 |
| Reference coverage | Top 5 | Top 20, rerank to 5 | Fuse dense 20 and BM25 50, retain 15, rerank to 5 |
| Hit rate at K, labelled Recall@K | Top K | First K of dense top 50, no reranking | First K of fused top 50 lists, no reranking |
| Context-size estimate | Top 3 | Top 15, rerank to 5 | Fuse dense 15 and BM25 50, retain 15, rerank to 5 |

Candidate requests are capped at the actual number of chunks.

For hybrid retrieval, each method's candidate scores are min–max normalized separately. Dense similarity is `1 - cosine_distance`. For chunk `i`:

```text
hybrid_score(i) = alpha * normalized_dense(i)
                + (1 - alpha) * normalized_bm25(i)
```

Missing candidates contribute zero. The default `alpha=0.5` gives equal weight to both score streams; the original normalization convention assigns 1 to every score when a candidate list is constant. The cross-encoder then reranks the selected candidates.

## What the metrics mean

| Output | Definition | Interpretation |
| --- | --- | --- |
| Answer accuracy | Correct answer letters / evaluated questions × 100 | Abstentions and unparseable responses count as incorrect |
| `source_accuracy` | Percentage of questions whose retrieved context covers at least 75% of reference-word occurrences | Lexical reference-coverage proxy; not verification of generated citations |
| `avg_overlap_score` | Mean reference-word coverage × 100 | Matching ignores word order and punctuation |
| Recall@K | Percentage of questions passing the same coverage threshold within the concatenated top K chunks | A question-level hit rate proxy, not standard recall over annotated relevant document IDs |
| Context overhead | Mean context characters / 4 | Rough size estimate, not tokenizer counts, billed tokens, irrelevant-token counts or measured latency |

Coverage counts each word occurrence in the reference when that word occurs anywhere in the retrieved context. This makes long contexts and repeated/common words influential. Dense and hybrid context-size measurements use five final chunks versus BM25's three, so the resulting size ratios are **not** a controlled comparison of retrieval efficiency. Original CSV/plot labels are retained to connect the code to the report.

## Outputs

Every invocation creates `outputs/<UTC timestamp>/`, unless `--output PATH` is supplied. A non-empty destination is rejected to avoid overwriting earlier runs.

| Mode | Files |
| --- | --- |
| All | `run.json`, `paper_clean.txt`, `chunks.json` |
| `bm25` | Also `bm25_retrieval.json` and `bm25_summary.json` |
| `full` | Also `accuracy.csv`, `source.csv`, `recall.csv`, `overhead.csv`, four dashboard PNGs and `CONCLUSIONES_PIPELINE.txt` |

`run.json` records arguments, the model ID when applicable, input SHA-256 hashes, installed direct-package versions, Python version and run status. API failures stop the experiment and mark it failed instead of silently turning a quota/authentication failure into answer accuracy. A failed full run does not resume and may lack final tables. Per-question Gemini responses are not persisted in this version.

## Original results and reproducibility

The supplied Spanish report describes approximate answer accuracies of **73% for the baseline, 93% for BM25 and 96% for dense retrieval**. These are historical report claims, not results generated while preparing this repository. The original archive contained no per-question prediction log, dependency lockfile or complete machine-readable result tables to independently reconstruct those numbers.

The code is now easier to run, but the preparation fixes can affect results: strictly bounded chunking, stricter answer parsing, propagation of the coverage threshold and an explicit generation temperature of zero. Dependencies use version ranges, not the original environment. See [changes](docs/CHANGES.md) and [validation notes](docs/VALIDATION.md).

Remaining research limitations include the small single-paper dataset, unequal retrieval budgets across metrics, different prompts between methods, no paired uncertainty analysis, no measured latency/token billing, and no independent evidence annotations. High baseline accuracy alone cannot establish whether an LLM had seen a paper or whether it relied on deduction. Treat `--limit` as a smoke check, not a representative evaluation.

## Notebook

Install Jupyter separately in the same environment:

```bash
python -m pip install jupyterlab
python -m jupyter lab notebooks/Code.ipynb
```

The notebook imports the same code used by the terminal. Its default cells only prepare the dataset; BM25 and full API commands are commented out until you explicitly choose to run them. Saved outputs and execution counts are cleared for publication.

## Authorship and source material

Credit belongs to **both project authors**. The paper belongs to its own authors and the supplied dataset's original provenance/licence is not documented in the ZIP. See [source and rights notes](docs/ATTRIBUTION.md). No new open-source licence has been assigned automatically.

Repository: [alexdmngz/rag-mpe-benchmark](https://github.com/alexdmngz/rag-mpe-benchmark). See [repository maintenance instructions](docs/PUBLISHING.md) for cloning and updating it.
