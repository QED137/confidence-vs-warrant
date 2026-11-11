#!/usr/bin/env bash
# Export ARC-Easy validation to a plain JSON file in the repo, so the compute
# nodes never need the HF cache or the network for it. Run on the LOGIN node.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
source .venv/bin/activate
source .env
mkdir -p data

python - <<'PY'
import json
from datasets import load_dataset

ds = load_dataset("allenai/ai2_arc", "ARC-Easy")["validation"]
rows = [{"id": r.get("id", str(i)), "question": r["question"],
         "choices": list(r["choices"]["text"]),
         "labels": list(r["choices"]["label"]),
         "answerKey": r["answerKey"]}
        for i, r in enumerate(ds)]
with open("data/arc_easy_validation.json", "w") as f:
    json.dump(rows, f)
print(f"wrote data/arc_easy_validation.json with {len(rows)} items")
PY

echo "Commit data/arc_easy_validation.json so the job never needs the Hub for ARC."
