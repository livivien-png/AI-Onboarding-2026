"""Train BGE with validation-based early stopping and an untouched test split."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import MODEL_NAME, IntentClassifier, get_device, load_tokenizer, tokenize


RANDOM_SEED = 42
VALIDATION_SEED = 43
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
MAX_EPOCHS = 12
BATCH_SIZE = 16
LEARNING_RATE = 2e-6
HEAD_LEARNING_RATE = 5e-5
PATIENCE = 2
TRAINING_SEEDS = (42, 43, 44)
TUNING_MODES = ("head_only", "last_layer", "full_low_lr")
BASELINE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BLEND_WEIGHTS = tuple(np.linspace(0.0, 1.0, 21))


def load_examples(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Read the current JSON array format or JSONL training rows."""
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
        rows = payload if isinstance(payload, list) else [payload]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]

    texts, labels, groups = [], [], []
    for row_number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"row {row_number}: expected an object")
        text = row.get("text")
        label = row.get("intent")
        if label is None:
            legacy_labels = row.get("intents")
            if isinstance(legacy_labels, list) and len(legacy_labels) == 1:
                label = legacy_labels[0]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"row {row_number}: text must be a non-empty string")
        if not isinstance(label, str) or not label:
            raise ValueError(f"row {row_number}: expected one intent label")
        texts.append(text.strip())
        labels.append(label)
        group = row.get("family")
        groups.append(group if isinstance(group, str) and group else f"row_{row_number}")
    if not texts:
        raise ValueError(f"{path} contains no examples")
    return texts, labels, groups


def grouped_split(groups: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split whole phrase families so paraphrases cannot cross data splits."""
    # User-reviewed supplemental examples are always training-only. This lets
    # new real-world examples improve the model without leaking into evaluation.
    training_only = {group for group in groups if group.startswith("training_only_")}
    unique_groups = sorted(set(groups) - training_only)
    if len(unique_groups) < 5:
        raise ValueError("grouped evaluation needs at least five phrase families")
    shuffled = list(np.random.default_rng(RANDOM_SEED).permutation(unique_groups))
    test_count = max(1, round(len(shuffled) * TEST_SIZE))
    validation_count = max(1, round(len(shuffled) * VALIDATION_SIZE))
    test_groups = set(shuffled[:test_count])
    validation_groups = set(shuffled[test_count : test_count + validation_count])
    train_groups = set(shuffled[test_count + validation_count :]) | training_only
    if not train_groups:
        raise ValueError("grouped split left no training phrase families")
    train_indices = np.asarray([i for i, group in enumerate(groups) if group in train_groups])
    validation_indices = np.asarray([i for i, group in enumerate(groups) if group in validation_groups])
    test_indices = np.asarray([i for i, group in enumerate(groups) if group in test_groups])
    return train_indices, validation_indices, test_indices


class IntentDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], self.labels[index]


def make_collate(tokenizer, device: torch.device):
    def collate(batch: list[tuple[str, int]]):
        texts, labels = zip(*batch)
        inputs = tokenize(tokenizer, list(texts), device)
        targets = torch.tensor(labels, dtype=torch.long, device=device)
        return inputs, targets

    return collate


def predict_probabilities(model, tokenizer, texts, device, batch_size):
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenize(tokenizer, batch_texts, device)
            probabilities.extend(model(**inputs).softmax(dim=1).cpu().numpy())
    return np.asarray(probabilities)


def macro_f1(targets, probabilities) -> float:
    return f1_score(
        targets,
        probabilities.argmax(axis=1),
        average="macro",
        zero_division=0,
    )


def configure_tuning_mode(
    model: IntentClassifier, mode: str, encoder_lr: float, head_lr: float
) -> list[dict]:
    """Freeze most or all of the encoder and return optimizer parameter groups."""
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    if mode == "head_only":
        return [{"params": list(model.classifier.parameters()), "lr": 1e-3}]
    if mode == "last_layer":
        encoder_stack = getattr(model.encoder, "encoder", None)
        layers = getattr(encoder_stack, "layer", None)
        if layers is None:
            raise RuntimeError("Could not find the encoder's final transformer layer")
        for parameter in layers[-1].parameters():
            parameter.requires_grad = True
        return [
            {
                "params": [p for p in model.encoder.parameters() if p.requires_grad],
                "lr": encoder_lr,
            },
            {"params": list(model.classifier.parameters()), "lr": head_lr},
        ]
    if mode == "full_low_lr":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = True
        return [
            {"params": list(model.encoder.parameters()), "lr": encoder_lr},
            {"params": list(model.classifier.parameters()), "lr": head_lr},
        ]
    raise ValueError(f"unknown tuning mode: {mode}")


def fit_candidate(
    model,
    mode,
    loader,
    tokenizer,
    validation_texts,
    validation_ids,
    device,
    epochs,
    batch_size,
    patience,
    seed,
    encoder_lr,
    head_lr,
):
    optimizer = torch.optim.AdamW(
        configure_tuning_mode(model, mode, encoder_lr, head_lr), weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()
    best_key = (-1.0, -1.0)
    best_epoch = 0
    best_state = None
    best_probabilities = None
    epochs_without_improvement = 0
    for epoch in range(epochs):
        model.train()
        if mode == "head_only":
            model.encoder.eval()
        elif mode == "last_layer":
            model.encoder.eval()
            model.encoder.encoder.layer[-1].train()
        running_loss = 0.0
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(**inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.size(0)

        probabilities = predict_probabilities(
            model, tokenizer, validation_texts, device, batch_size
        )
        validation_accuracy = accuracy_score(validation_ids, probabilities.argmax(axis=1))
        validation_f1 = macro_f1(validation_ids, probabilities)
        score = (float(validation_accuracy), float(validation_f1))
        print(
            f"  seed {seed} {mode} epoch {epoch + 1}/{epochs} "
            f"loss={running_loss / len(loader.dataset):.4f} "
            f"validation accuracy={validation_accuracy:.4f} macro-F1={validation_f1:.4f}"
        )
        if score > best_key:
            best_key = score
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_probabilities = probabilities
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"{mode} seed {seed} did not produce a checkpoint")
    return {
        "mode": mode,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_accuracy": best_key[0],
        "validation_macro_f1": best_key[1],
        "state": best_state,
        "validation_probabilities": best_probabilities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=Path("trainingData.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="maximum epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--head-learning-rate", type=float, default=HEAD_LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.learning_rate <= 0
        or args.head_learning_rate <= 0
        or args.patience < 1
    ):
        parser.error("epochs, batch size, learning rate, and patience must be positive")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    texts, labels, groups = load_examples(args.data)
    label_names = sorted(set(labels))
    label2id = {label: index for index, label in enumerate(label_names)}
    ids = np.asarray([label2id[label] for label in labels])

    train_indices, validation_indices, test_indices = grouped_split(groups)
    train_texts = [texts[i] for i in train_indices]
    validation_texts = [texts[i] for i in validation_indices]
    test_texts = [texts[i] for i in test_indices]
    train_ids = ids[train_indices]
    validation_ids = ids[validation_indices]
    test_ids = ids[test_indices]

    device = get_device()
    print(f"Device: {device}")
    print(
        f"Split: {len(train_texts)} train / {len(validation_texts)} validation / "
        f"{len(test_texts)} test"
    )
    held_out_indices = np.concatenate((validation_indices, test_indices))
    print(f"Held-out phrase families: {sorted(set(groups[i] for i in held_out_indices))}")
    print(f"Loading {MODEL_NAME}; comparing frozen and low-rate tuning")
    tokenizer = load_tokenizer()
    candidate_results = []
    for seed in TRAINING_SEEDS:
        torch.manual_seed(seed)
        initial_model = IntentClassifier(
            num_labels=len(label_names), dropout=0.2
        ).cpu()
        for mode in TUNING_MODES:
            model = copy.deepcopy(initial_model).to(device)
            loader = DataLoader(
                IntentDataset(train_texts, train_ids.tolist()),
                batch_size=args.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
                collate_fn=make_collate(tokenizer, device),
            )
            candidate_results.append(
                fit_candidate(
                    model,
                    mode,
                    loader,
                    tokenizer,
                    validation_texts,
                    validation_ids,
                    device,
                    args.epochs,
                    args.batch_size,
                    args.patience,
                    seed,
                    args.learning_rate,
                    args.head_learning_rate,
                )
            )
            del model
        del initial_model

    selected_candidate = max(
        candidate_results,
        key=lambda item: (
            item["validation_accuracy"],
            item["validation_macro_f1"],
        ),
    )
    print("Candidate validation scores:")
    for item in candidate_results:
        print(
            f"  seed {item['seed']} {item['mode']}: "
            f"accuracy={item['validation_accuracy']:.4f}, "
            f"macro-F1={item['validation_macro_f1']:.4f}, "
            f"best epoch={item['best_epoch']}"
        )
    print(
        "Selected by validation accuracy: "
        f"seed {selected_candidate['seed']} {selected_candidate['mode']} "
        f"(accuracy={selected_candidate['validation_accuracy']:.4f})"
    )

    torch.manual_seed(selected_candidate["seed"])
    model = IntentClassifier(num_labels=len(label_names), dropout=0.2).to(device)
    model.load_state_dict(selected_candidate["state"])
    validation_bge_probabilities = selected_candidate["validation_probabilities"]
    best_epoch = selected_candidate["best_epoch"]
    best_validation_accuracy = selected_candidate["validation_accuracy"]
    best_validation_f1 = selected_candidate["validation_macro_f1"]

    print(f"Loading frozen baseline encoder: {BASELINE_MODEL}")
    baseline_encoder = SentenceTransformer(BASELINE_MODEL)
    train_embeddings = baseline_encoder.encode(
        train_texts, show_progress_bar=True, convert_to_numpy=True
    )
    validation_embeddings = baseline_encoder.encode(
        validation_texts, show_progress_bar=True, convert_to_numpy=True
    )
    test_embeddings = baseline_encoder.encode(
        test_texts, show_progress_bar=True, convert_to_numpy=True
    )
    baseline_classifier = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
    baseline_classifier.fit(train_embeddings, train_ids)
    validation_baseline_probabilities = baseline_classifier.predict_proba(
        validation_embeddings
    )

    # Choose the blend by validation accuracy. The final test split stays untouched.
    blend_results = []
    for weight in BLEND_WEIGHTS:
        blended = (
            weight * validation_baseline_probabilities
            + (1.0 - weight) * validation_bge_probabilities
        )
        blend_results.append(
            (
                accuracy_score(validation_ids, blended.argmax(axis=1)),
                macro_f1(validation_ids, blended),
                float(weight),
            )
        )
    validation_accuracy, validation_macro_f1, baseline_weight = max(
        blend_results, key=lambda result: (result[0], result[1], result[2])
    )
    print(
        f"Selected baseline weight: {baseline_weight:.2f} "
        f"(validation accuracy: {validation_accuracy:.4f}, "
        f"macro-F1: {validation_macro_f1:.4f})"
    )

    test_baseline_probabilities = baseline_classifier.predict_proba(test_embeddings)
    test_bge_probabilities = predict_probabilities(
        model, tokenizer, test_texts, device, args.batch_size
    )
    test_ensemble_probabilities = (
        baseline_weight * test_baseline_probabilities
        + (1.0 - baseline_weight) * test_bge_probabilities
    )
    test_predictions = test_ensemble_probabilities.argmax(axis=1)
    baseline_predictions = test_baseline_probabilities.argmax(axis=1)
    bge_predictions = test_bge_probabilities.argmax(axis=1)
    accuracy = accuracy_score(test_ids, test_predictions)
    macro_f1_score = f1_score(
        test_ids, test_predictions, average="macro", zero_division=0
    )
    report = classification_report(
        test_ids,
        test_predictions,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
    )
    print(
        f"Untouched test baseline: accuracy={accuracy_score(test_ids, baseline_predictions):.4f}, "
        f"macro-F1={f1_score(test_ids, baseline_predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        f"Untouched test fine-tuned {MODEL_NAME}: "
        f"accuracy={accuracy_score(test_ids, bge_predictions):.4f}, "
        f"macro-F1={f1_score(test_ids, bge_predictions, average='macro', zero_division=0):.4f}"
    )
    print(f"Untouched test ensemble: accuracy={accuracy:.4f}, macro-F1={macro_f1_score:.4f}")
    print("Per-intent test report:")
    print(report)
    print("Test mistakes:")
    misses = 0
    for text, true_id, predicted_id in zip(test_texts, test_ids, test_predictions):
        if true_id != predicted_id:
            misses += 1
            print(
                f"- {text!r}: expected {label_names[true_id]}, "
                f"predicted {label_names[predicted_id]}"
            )
    if misses == 0:
        print("None")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    joblib.dump(baseline_classifier, args.output_dir / "baseline_classifier.joblib")
    (args.output_dir / "label2id.json").write_text(
        json.dumps(label2id, indent=2) + "\n", encoding="utf-8"
    )
    metrics = {
        "model": MODEL_NAME,
        "baseline_model": BASELINE_MODEL,
        "baseline_weight": baseline_weight,
        "accuracy": accuracy,
        "macro_f1": macro_f1_score,
        "baseline_accuracy": accuracy_score(test_ids, baseline_predictions),
        "baseline_macro_f1": f1_score(
            test_ids, baseline_predictions, average="macro", zero_division=0
        ),
        "fine_tuned_accuracy": accuracy_score(test_ids, bge_predictions),
        "fine_tuned_macro_f1": f1_score(
            test_ids, bge_predictions, average="macro", zero_division=0
        ),
        "validation_macro_f1": validation_macro_f1,
        "fine_tuned_validation_accuracy": best_validation_accuracy,
        "fine_tuned_validation_macro_f1": best_validation_f1,
        "selected_tuning_mode": selected_candidate["mode"],
        "selected_seed": selected_candidate["seed"],
        "training_candidates": [
            {
                "mode": item["mode"],
                "seed": item["seed"],
                "best_epoch": item["best_epoch"],
                "validation_accuracy": item["validation_accuracy"],
                "validation_macro_f1": item["validation_macro_f1"],
            }
            for item in candidate_results
        ],
        "seed": RANDOM_SEED,
        "validation_seed": VALIDATION_SEED,
        "train_fraction": len(train_texts) / len(texts),
        "validation_fraction": len(validation_texts) / len(texts),
        "test_fraction": len(test_texts) / len(texts),
        "split_strategy": "grouped_by_phrase_family",
        "train_families": sorted(set(groups[i] for i in train_indices)),
        "validation_families": sorted(set(groups[i] for i in validation_indices)),
        "test_families": sorted(set(groups[i] for i in test_indices)),
        "best_epoch": best_epoch,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "weight_decay": 0.01,
        "train_rows": len(train_texts),
        "validation_rows": len(validation_texts),
        "test_rows": len(test_texts),
        "classification_report": report,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved model, baseline classifier, label map, and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
