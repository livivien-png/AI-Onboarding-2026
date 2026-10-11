"""Predict an intent with the trained BGE and frozen-baseline ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import torch
from sentence_transformers import SentenceTransformer

from model import IntentClassifier, get_device, load_tokenizer, tokenize


RUNS_DIR = Path("runs")
BASELINE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="sentence to classify")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("text must not be empty")

    label2id = json.loads((args.runs_dir / "label2id.json").read_text(encoding="utf-8"))
    id2label = {index: label for label, index in label2id.items()}
    metrics_path = args.runs_dir / "metrics.json"
    if metrics_path.exists():
        baseline_weight = json.loads(metrics_path.read_text(encoding="utf-8")).get(
            "baseline_weight", 0.7
        )
    else:
        baseline_weight = 0.7

    device = get_device()
    model = IntentClassifier(num_labels=len(label2id)).to(device)
    model.load_state_dict(
        torch.load(args.runs_dir / "model.pt", map_location=device, weights_only=True)
    )
    model.eval()
    tokenizer = load_tokenizer()
    baseline_encoder = SentenceTransformer(BASELINE_MODEL)
    baseline_classifier = joblib.load(args.runs_dir / "baseline_classifier.joblib")

    with torch.inference_mode():
        inputs = tokenize(tokenizer, [args.text.strip()], device)
        bge_probabilities = model(**inputs).softmax(dim=1).cpu().numpy()[0]
    baseline_embedding = baseline_encoder.encode(
        [args.text.strip()], convert_to_numpy=True, show_progress_bar=False
    )
    baseline_probabilities = baseline_classifier.predict_proba(baseline_embedding)[0]
    probabilities = (
        baseline_weight * baseline_probabilities
        + (1.0 - baseline_weight) * bge_probabilities
    )
    selected_id = int(probabilities.argmax())
    confidence = float(probabilities[selected_id])
    selected_intent = id2label[selected_id]

    print(
        json.dumps(
            {
                "text": args.text,
                "selected_intent": selected_intent,
                "confidence": confidence,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
