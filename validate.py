"""Validate JSON or JSONL training data against intents.json."""
import json
import sys
from collections import Counter
from pathlib import Path

MIN_PER_INTENT = 30    # rows per intent; handwritten or synthetic both count

if len(sys.argv) != 2:
    sys.exit("usage: python validate.py trainingData.json")

intents_file = Path(__file__).with_name("intents.json")
try:
    intent_rows = json.loads(intents_file.read_text(encoding="utf-8"))
    if not isinstance(intent_rows, list):
        raise ValueError("expected a JSON array")
    labels = {
        row["intent"]
        for row in intent_rows
        if isinstance(row, dict) and isinstance(row.get("intent"), str)
    }
    if not labels:
        raise ValueError("no intent labels found")
except (OSError, ValueError, TypeError) as exc:
    sys.exit(f"could not read valid labels from {intents_file}: {exc}")

seen, total, problems = set(), Counter(), []

try:
    data_path = Path(sys.argv[1])
    raw_data = data_path.read_text(encoding="utf-8")
except OSError as exc:
    sys.exit(f"could not read training data {sys.argv[1]}: {exc}")

# Accept the project's JSON array format as well as one-JSON-object-per-line.
try:
    parsed_data = json.loads(raw_data)
    rows = parsed_data if isinstance(parsed_data, list) else [parsed_data]
except ValueError:
    rows = []
    for n, line in enumerate(raw_data.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            problems.append(f"line {n}: invalid JSON")

for n, row in enumerate(rows, 1):
    if not isinstance(row, dict):
        problems.append(f"row {n}: expected an object with text and intent")
        continue

    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        problems.append(f"row {n}: text must be a non-empty string")
        continue

    # Current data uses a singular `intent`; also accept the older JSONL format.
    if "intent" in row:
        intent_values = [row["intent"]]
    else:
        intent_values = row.get("intents")
    if (
        not isinstance(intent_values, list)
        or len(intent_values) != 1
        or not isinstance(intent_values[0], str)
        or intent_values[0] not in labels
    ):
        problems.append(f"row {n}: need exactly one known intent label; got {intent_values!r}")
        continue

    text = text.strip()
    normalized_text = text.casefold()
    if normalized_text in seen:
        problems.append(f"row {n}: duplicate text {text!r}")
        continue
    seen.add(normalized_text)
    total[intent_values[0]] += 1

for label in sorted(labels):
    if total[label] < MIN_PER_INTENT:
        problems.append(f"{label}: only {total[label]} rows (need {MIN_PER_INTENT})")

print("\n".join(problems))
print(f"{sum(total.values())} rows, {len(problems)} problems")
sys.exit(1 if problems else 0)
