# Validation of the prepared repository

Checked on 2 October 2026. These checks concern the prepared code, not a rerun of the historical Gemini experiment.

## Passed locally

- **14 automated tests**: input validation, duplicate questions, cleaning, chunk length and overlap, answer parsing, metric behavior, path independence, overwrite protection, API-failure handling and pipeline integration with test doubles.
- **Preparation on all 70 questions** using the supplied paper: **69 non-empty chunks**, with a maximum of **250 words** each.
- **Real BM25 retrieval on all 70 questions**, using `rank-bm25==0.2.2`: a **reference hit rate of 80.0% (56/70)** with the top 3 chunks and a 0.75 coverage threshold. Mean reference-word coverage was **91.7394%**. These are retrieval metrics, not answer accuracy.
- **One-round dashboard generation** and all four experiment metric paths exercised with a small synthetic corpus and test doubles for the dense index, reranker and Gemini. Synthetic test scores are not scientific results.
- Python files and notebook code cells parsed successfully; notebook outputs and execution counts are empty.
- The dataset, paper text and report match their supplied versions byte for byte.
- A targeted scan found no matching Google API key, GitHub token, PEM private-key header or local workspace path in the text/code files. This is a bounded pattern scan, not a guarantee that every possible secret format is absent.

## Local environment used

| Component | Version |
| --- | --- |
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| Matplotlib | 3.10.8 |
| rank-bm25 | 0.2.2 |

CI is configured for Python 3.11. Check the current run status in [GitHub Actions](https://github.com/alexdmngz/rag-mpe-benchmark/actions). The local checks above were completed before publication. Dependency files specify compatibility ranges rather than an exact lock of all transitive dependencies.

## Not verified in this preparation

- Live Gemini authentication, quota, model availability or answer quality: no user API key was provided and no billable generation was launched.
- Real downloads/inference for SentenceTransformers, the cross-encoder or a live Chroma embedding integration.
- Installation and execution of the complete dependency set on Windows or macOS.
- Exact reproduction of the report's historical results.

## Repeat the checks

```bash
python -m pip install -r requirements-check.txt
python -m unittest discover -s tests -v
python src/Script.py --mode prepare
python src/Script.py --mode bm25
```

Use the Python executable of your virtual environment. The standard-library tests alone can be run with `python -m unittest discover -s tests -p test_core.py -v` before installing dependencies. For the final integration check with real models, configure `.env`, install `requirements.txt`, and deliberately run `python src/Script.py --mode full --limit 3 --rounds 1`.
