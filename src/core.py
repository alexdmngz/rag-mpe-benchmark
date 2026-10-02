"""Data validation and text helpers. This module needs only Python itself."""

import json
import re


def load_data(path):
    """Validate the multiple-choice dataset before starting an experiment."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("The dataset must be a non-empty list.")
    questions = set()
    for number, item in enumerate(data, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {number}: expected an object.")
        for key in ("question", "correct_answer", "paper_reference"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Question {number}: missing or empty {key}.")
        answers = item.get("answers")
        if not isinstance(answers, dict) or set(answers) != set("ABCD"):
            raise ValueError(f"Question {number}: answers must have keys A, B, C, D.")
        if any(not isinstance(value, str) or not value.strip() for value in answers.values()):
            raise ValueError(f"Question {number}: an answer is empty.")
        if item["correct_answer"] not in answers:
            raise ValueError(f"Question {number}: invalid correct_answer.")
        question = item["question"].strip()
        if question in questions:
            raise ValueError(f"Question {number}: duplicate question.")
        questions.add(question)
    return data


def remove_block_between(text, start_marker, end_marker):
    start = text.find(start_marker)
    if start == -1:
        return text
    end = text.find(end_marker, start)
    return text[:start] + (text[end:] if end != -1 else "")


def clean_paper_text(text):
    """Preserve the original paragraph cleaning and reference-removal heuristic."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = remove_block_between(text, "References", "Appendix")
    paragraphs, current = [], ""
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if current and current.endswith((".", "?", "!")) and not current.endswith("et al."):
                paragraphs.append(current)
                current = ""
        elif not re.fullmatch(r"\d+(\.\d+)?", line):
            if current.endswith("-"):
                current = current[:-1] + line
            else:
                current = (current + " " + line).strip()
    if current:
        paragraphs.append(current)
    return "\n\n".join(paragraphs)


def count_words(text):
    return len(text.split())


def recursive_chunker(text, max_size=250, overlap=30):
    """Pack paragraphs/sentences, splitting long spans by words when necessary."""
    if max_size <= 0 or not 0 <= overlap < max_size:
        raise ValueError("Require max_size > 0 and 0 <= overlap < max_size.")
    chunks, current = [], []
    for paragraph in text.split("\n\n"):
        units = [paragraph] if count_words(paragraph) <= max_size else re.split(r"(?<=[.?!])\s+", paragraph)
        for unit in units:
            words = unit.split()
            if not words:
                continue
            if current and len(current) + len(words) > max_size:
                chunks.append(" ".join(current))
                current = current[-overlap:] if overlap else []
            while len(current) + len(words) > max_size:
                capacity = max_size - len(current)
                current.extend(words[:capacity])
                words = words[capacity:]
                chunks.append(" ".join(current))
                current = current[-overlap:] if overlap else []
            current.extend(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def normalize_text_for_match(text):
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split())


def better_tokenize(text):
    return normalize_text_for_match(text).split()


def normalize_scores(scores):
    if len(scores) == 0:
        return []
    minimum, maximum = min(scores), max(scores)
    if maximum == minimum:
        return [1.0] * len(scores)
    return [(score - minimum) / (maximum - minimum) for score in scores]


def parse_mc_answer(text_response):
    """Accept an explicit answer or a bare letter, never a letter in a citation."""
    text = (text_response or "").strip()
    match = re.search(r"^\s*Answer:\s*([A-D]|N/A)[.)]?\s*$", text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).upper()
    return text.upper() if text.upper() in (*"ABCD", "N/A") else "N/A"


def compute_accuracy(model_answers, true_answers):
    if len(model_answers) != len(true_answers):
        raise ValueError("Predictions and labels must have equal lengths.")
    mistakes = [i + 1 for i, (pred, label) in enumerate(zip(model_answers, true_answers)) if pred != label]
    total = len(true_answers)
    correct = total - len(mistakes)
    return (100 * correct / total if total else 0.0), correct, total, mistakes


def check_fuzzy_attribution(retrieved_context, reference, threshold=0.75):
    ref_tokens = better_tokenize(reference)
    ctx_tokens = set(better_tokenize(retrieved_context))
    if not ref_tokens or not ctx_tokens:
        return False, 0.0
    score = sum(token in ctx_tokens for token in ref_tokens) / len(ref_tokens)
    return score >= threshold, score


def check_fuzzy_match(retrieved_context, reference, threshold=0.75):
    return check_fuzzy_attribution(retrieved_context, reference, threshold)[0]
