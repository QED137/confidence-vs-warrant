#!/usr/bin/env bash
# Step 2 - download models and datasets. Run on the LOGIN node.
# Files are downloaded, never loaded into memory.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source .env

MODELS_FILE="${1:-configs/models.txt}"

python - "$MODELS_FILE" <<'PY'
import sys
from huggingface_hub import snapshot_download

models_file = sys.argv[1]

print("== datasets", flush=True)
from datasets import load_dataset
load_dataset("truthful_qa", "multiple_choice")
load_dataset("trivia_qa", "rc.nocontext", split="validation[:5000]")

ALLOW = ["*.json", "*.txt", "*.model", "*.safetensors"]

print("== models (download only, nothing loaded into RAM)", flush=True)
for line in open(models_file):
    name = line.strip()
    if not name or name.startswith("#"):
        continue
    print(f"   {name}", flush=True)
    try:
        snapshot_download(repo_id=name, allow_patterns=ALLOW, max_workers=4)
    except Exception as exc:
        print(f"     no safetensors ({exc.__class__.__name__}); retrying with *.bin", flush=True)
        snapshot_download(repo_id=name, allow_patterns=ALLOW + ["*.bin"],
                          ignore_patterns=["*.h5", "*.msgpack", "*.onnx"], max_workers=4)
PY

echo; echo "Cache size:"; du -sh hf_cache
