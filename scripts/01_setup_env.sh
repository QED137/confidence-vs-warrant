#!/usr/bin/env bash
# Step 1 - create the environment. Run once, on the LOGIN node.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

echo "== creating virtualenv in .venv"
"$PYTHON" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

echo "== installing requirements"
pip install -r requirements.txt

echo "== caching HF downloads inside the project (avoids filling \$HOME quota)"
mkdir -p hf_cache logs results figures
cat > .env <<EOF
export HF_HOME="$(pwd)/hf_cache"
export HF_DATASETS_CACHE="$(pwd)/hf_cache/datasets"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}
EOF

echo "== self-test (no GPU, no network)"
source .env
python -m pytest tests -q

echo
echo "OK. Environment ready."
echo "Next: bash scripts/02_fetch_assets.sh"
