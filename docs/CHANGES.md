# Changes made for GitHub preparation

The uploaded ZIP remains the original source. The dataset, paper text and academic report in this package are byte-for-byte copies of the supplied files.

## Execution and organization

- Replaced notebook-only `get_ipython()` installation calls and top-level execution with a command-line entry point.
- Resolved default input paths from the script location. Removed the dependency on the missing `Paper.pdf`.
- Moved standard-library helpers to `src/core.py`; retained the experiment functions in `src/experiment.py`, keeping the last definition where the notebook defined a function twice.
- Moved credentials to `.env` or environment variables. Added explicit full-run mode, model selection and small-run limits.
- Adapted the Gemini call to `google-genai`. Set generation temperature to zero; this was not explicitly set in the original code.
- Removed the destructive collection-delete step; each experiment creates its own ephemeral collection with a unique name.
- Replaced Jupyter `display()` and interactive plots with terminal tables and saved PNGs.
- Added CSV output, input hashes, run metadata, a short notebook, dependency files and CI checks.

## Correctness changes that may change results

- Enforced the chunk word limit, including long sentences and overlap. The original implementation could emit empty or oversized chunks and mishandled zero overlap.
- Made answer parsing require an explicit `Answer:` line or a bare answer; letters inside source text can no longer silently become predictions.
- Rejected mismatched prediction/label lengths instead of truncating via `zip`.
- Capped dense candidate counts at the number of chunks.
- Passed the selected coverage threshold into Recall@K as well as source coverage.
- Replaced API-failure-to-empty-answer behavior with an exception after retries. Permanent request/authentication/model errors stop immediately.
- Handled one-round plot error bars without changing the reported standard deviation: unavailable sample deviations stay undefined in tables and render with zero-length bars.
- Removed an automatic suggestion to lower the evaluation threshold when results are poor.

## Deliberately retained research limitations

The original retriever-specific prompts, stage-specific context budgets, approximate token estimate and lexical coverage metrics remain. Changing these would redesign the experiment; their limitations are explained in the README. This prepared version is not claimed to exactly reproduce the original report.
