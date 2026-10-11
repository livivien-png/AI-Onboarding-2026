"""Evaluate a frozen sentence-embedding baseline for intent classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42
TEST_SIZE = 0.15


def load_examples(path: Path) -> tuple[list[str], list[str]]:
    """Read either a JSON array or JSONL rows with intent/intents labels."""
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
        rows = payload if isinstance(payload, list) else [payload]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]

    texts, labels = [], []
    for row_number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"row {row_number}: expected an object")
        text = row.get("text")
        label = row.get("intent")
        if label is None:
            old_labels = row.get("intents")
            if isinstance(old_labels, list) and len(old_labels) == 1:
                label = old_labels[0]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"row {row_number}: text must be a non-empty string")
        if not isinstance(label, str) or not label:
            raise ValueError(f"row {row_number}: expected one intent label")
        texts.append(text.strip())
        labels.append(label)

    if not texts:
        raise ValueError(f"{path} contains no examples")
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=Path("trainingData.json"))
    parser.add_argument("--output", type=Path, default=Path("baseline.txt"))
    args = parser.parse_args()

    texts, labels = load_examples(args.data)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    encoder = SentenceTransformer(MODEL_NAME)
    train_embeddings = encoder.encode(
        train_texts, show_progress_bar=True, convert_to_numpy=True
    )
    test_embeddings = encoder.encode(
        test_texts, show_progress_bar=True, convert_to_numpy=True
    )

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
    classifier.fit(train_embeddings, train_labels)
    predictions = classifier.predict(test_embeddings)

    accuracy = accuracy_score(test_labels, predictions)
    macro_f1 = f1_score(test_labels, predictions, average="macro", zero_division=0)
    report = classification_report(test_labels, predictions, zero_division=0)
    result = (
        f"Model: {MODEL_NAME} + sklearn LogisticRegression\n"
        f"Data: {args.data}\n"
        f"Split: {1 - TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test; seed={RANDOM_SEED}\n"
        f"Rows: {len(texts)}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro-F1: {macro_f1:.4f}\n\n"
        f"Per-intent classification report:\n{report}"
    )
    print(result)
    args.output.write_text(result + "\n", encoding="utf-8")
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
