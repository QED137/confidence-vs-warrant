#!/usr/bin/env bash
# Step 4 - the main study. One array task per model.
#
#   sbatch scripts/04_run_cpu.sbatch
#
# Array indices map to lines of configs/models.txt (1-based).
# CPU is sufficient: this is ~600 short forward passes per model.

#SBATCH --job-name=ecal
#SBATCH --partition=cpu_il
#SBATCH --array=1-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ecal_%A_%a.out
#SBATCH --error=logs/ecal_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
source .env

export HF_HUB_OFFLINE=1          # assets were prefetched in step 2
export TRANSFORMERS_OFFLINE=1

MODEL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs/models.txt | tr -d '[:space:]')
if [[ -z "$MODEL" ]]; then
  echo "ERROR: configs/models.txt has no model on line ${SLURM_ARRAY_TASK_ID}." >&2
  echo "       The file has $(grep -c . configs/models.txt) non-empty lines;" >&2
  echo "       keep --array within that range." >&2
  exit 2
fi
echo "=== model: $MODEL"
echo "=== cpus: ${SLURM_CPUS_PER_TASK}  mem: ${SLURM_MEM_PER_NODE:-unset} MB"

python -m src.run \
  --model "$MODEL" \
  --n 300 \
  --dtype float32 \
  --device cpu \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --detectors \
  --out results/

echo "=== done: $MODEL"

