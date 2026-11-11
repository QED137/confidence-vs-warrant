#!/usr/bin/env bash
# Step 5 (OPTIONAL) - the semantic-entropy detector condition.
#
#   sbatch scripts/05_run_gpu_semantic_entropy.sbatch
#
# Much heavier than step 4: 10 sampled generations per item plus pairwise NLI
# calls for clustering. Needs a GPU. Results go to results_se/ so they do not
# overwrite the main run.

#SBATCH --job-name=ecal-se
#SBATCH --array=1-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ecal_se_%A_%a.out
#SBATCH --error=logs/ecal_se_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
source .env

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs/models.txt)
echo "=== model: $MODEL (semantic entropy)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

mkdir -p results_se
python -m src.run \
  --model "$MODEL" \
  --n 300 \
  --dtype float32 \
  --device cuda \
  --detectors \
  --semantic-entropy \
  --out results_se/

echo "=== done: $MODEL"
