"""Measure the saved ensemble against a labeled JSON test-input file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score

from model import MAX_LENGTH, MODEL_NAME, IntentClassifier, get_device, load_tokenizer, tokenize
from train import BASELINE_MODEL, load_examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_inputs", nargs="?", type=Path, default=Path("test_inputs.json"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--training-data", type=Path, default=Path("trainingData.json"))
    parser.add_argument("--expected-rows", type=int, default=150)
    parser.add_argument(
        "--token-audit", type=Path,
        help="write one JSONL row per input with encoder tokens and top intent scores",
    )
    args = parser.parse_args()

    test_rows = json.loads(args.test_inputs.read_text(encoding="utf-8"))
    if not isinstance(test_rows, list) or not test_rows:
        parser.error("test input file must contain a non-empty JSON array")
    texts, labels = [], []
    for row_number, row in enumerate(test_rows, 1):
        if not isinstance(row, dict):
            parser.error(f"test row {row_number} must be an object")
        text, label = row.get("text"), row.get("intent")
        if not isinstance(text, str) or not text.strip():
            parser.error(f"test row {row_number} must have non-empty text")
        if not isinstance(label, str) or not label:
            parser.error(f"test row {row_number} must have one intent label")
        texts.append(text.strip())
        labels.append(label)

    normalized = [text.casefold() for text in texts]
    if len(normalized) != len(set(normalized)):
        parser.error("test input file contains duplicate text")
    if args.expected_rows < 1:
        parser.error("--expected-rows must be positive")
    if len(test_rows) != args.expected_rows:
        parser.error(
            f"evaluation file must contain exactly {args.expected_rows} rows; "
            f"found {len(test_rows)}"
        )
    training_texts, _, _ = load_examples(args.training_data)
    overlap = set(normalized) & {text.casefold() for text in training_texts}
    if overlap:
        parser.error(f"test input overlaps training data: {next(iter(overlap))!r}")

    label2id = json.loads((args.runs_dir / "label2id.json").read_text(encoding="utf-8"))
    unknown_labels = sorted(set(labels) - set(label2id))
    if unknown_labels:
        parser.error(f"unknown test labels: {unknown_labels}")
    target_ids = np.asarray([label2id[label] for label in labels])
    metrics = json.loads((args.runs_dir / "metrics.json").read_text(encoding="utf-8"))
    baseline_weight = metrics["baseline_weight"]

    device = get_device()
    model = IntentClassifier(num_labels=len(label2id)).to(device)
    model.load_state_dict(
        torch.load(args.runs_dir / "model.pt", map_location=device, weights_only=True)
    )
    model.eval()
    tokenizer = load_tokenizer()
    encoded_rows = tokenizer(
        texts, truncation=True, max_length=MAX_LENGTH,
        padding=False, add_special_tokens=True,
    )
    raw_lengths = [
        len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])
        for text in texts
    ]
    input_lengths = [len(ids) for ids in encoded_rows["input_ids"]]
    bge_probabilities = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            inputs = tokenize(tokenizer, texts[start : start + 16], device)
            bge_probabilities.extend(model(**inputs).softmax(1).cpu().numpy())

    baseline_encoder = SentenceTransformer(BASELINE_MODEL)
    baseline_classifier = joblib.load(args.runs_dir / "baseline_classifier.joblib")
    embeddings = baseline_encoder.encode(
        texts, convert_to_numpy=True, show_progress_bar=False
    )
    baseline_probabilities = baseline_classifier.predict_proba(embeddings)
    bge_probabilities = np.asarray(bge_probabilities)
    probabilities = baseline_weight * baseline_probabilities + (
        1.0 - baseline_weight
    ) * bge_probabilities
    predictions = probabilities.argmax(axis=1)
    baseline_predictions = baseline_probabilities.argmax(axis=1)
    bge_predictions = bge_probabilities.argmax(axis=1)
    id2label = {index: label for label, index in label2id.items()}
    if args.token_audit:
        args.token_audit.parent.mkdir(parents=True, exist_ok=True)
        with args.token_audit.open("w", encoding="utf-8") as audit:
            for index, (text, ids) in enumerate(zip(texts, encoded_rows["input_ids"])):
                top_indices = np.argsort(probabilities[index])[::-1][:3]
                record = {
                    "row": index + 1,
                    "text": text,
                    "input_token_count": len(ids),
                    "raw_token_count": raw_lengths[index],
                    "truncated": raw_lengths[index] > len(ids),
                    "input_token_ids": ids,
                    "input_tokens": tokenizer.convert_ids_to_tokens(ids),
                    "expected_intent": labels[index],
                    "predicted_intent": id2label[int(predictions[index])],
                    "confidence": float(probabilities[index, predictions[index]]),
                    "top_intents": [
                        {"intent": id2label[int(class_id)], "probability": float(probabilities[index, class_id])}
                        for class_id in top_indices
                    ],
                }
                audit.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Token audit: {args.token_audit}")
        print(
            "Input tokens: "
            f"min={min(input_lengths)}, mean={np.mean(input_lengths):.1f}, "
            f"max={max(input_lengths)}, truncated={sum(n > m for n, m in zip(raw_lengths, input_lengths))}"
        )
    accuracy = accuracy_score(target_ids, predictions)
    macro_f1 = f1_score(
        target_ids,
        predictions,
        labels=list(range(len(label2id))),
        average="macro",
        zero_division=0,
    )
    print(f"Inputs: {len(texts)} across {len(set(labels))} intents")
    print(
        "Frozen baseline: "
        f"accuracy={accuracy_score(target_ids, baseline_predictions):.4f}, "
        f"macro-F1={f1_score(target_ids, baseline_predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        f"Fine-tuned {MODEL_NAME}: "
        f"accuracy={accuracy_score(target_ids, bge_predictions):.4f}, "
        f"macro-F1={f1_score(target_ids, bge_predictions, average='macro', zero_division=0):.4f}"
    )
    print(f"Accuracy: {accuracy:.4f} ({int((predictions == target_ids).sum())}/{len(texts)})")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Model output: class probabilities over intents (no generated output tokens).")
    print("Misclassified inputs:")
    misses = 0
    for text, expected, predicted in zip(texts, target_ids, predictions):
        if expected != predicted:
            misses += 1
            print(f"- {text!r}: expected {id2label[expected]}, predicted {id2label[predicted]}")
    if misses == 0:
        print("None")
    print("Per-intent report:")
    print(
        classification_report(
            target_ids,
            predictions,
            labels=list(range(len(label2id))),
            target_names=[id2label[index] for index in range(len(label2id))],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
