#!/usr/bin/env bash
# Re-run the study with the ARC control stratum (human-written distractors)
# and a fixed candidate count. Results go to results_arc/, leaving the earlier
# TriviaQA runs in results/ untouched.
#
#   bash scripts/07_rerun_arc_control.sh
#   PARTITION=my_partition bash scripts/07_rerun_arc_control.sh
#
# ARC must be downloaded first, on the LOGIN node:
#   source .venv/bin/activate && source .env
#   python -c "from datasets import load_dataset; load_dataset('allenai/ai2_arc','ARC-Easy')"
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"          # absolute, so the job cannot lose it
cd "$REPO"

for f in .env .venv/bin/activate configs/models.txt; do
  [[ -e "$REPO/$f" ]] || { echo "ERROR: $REPO/$f missing - run scripts/01_setup_env.sh" >&2; exit 1; }
done

MODELS=$(grep -v '^#' "$REPO/configs/models.txt" | grep . | tr '\n' ' ')
echo "repo:   $REPO"
echo "models: $MODELS"

sbatch --partition="${PARTITION:-cpu_il}" \
       --chdir="$REPO" \
       --mem=64G --time=12:00:00 --cpus-per-task=8 \
       --job-name=ecal-arc \
       --output="$REPO/logs/arc_%j.out" --error="$REPO/logs/arc_%j.err" \
       --wrap="set -euo pipefail; \
               source '$REPO/.venv/bin/activate'; \
               source '$REPO/.env'; \
               export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; \
               for m in $MODELS; do \
                 echo \"=== \$m\"; \
                 python -m src.run --model \$m --n 300 --dtype float32 --device cpu \
                   --threads 8 --detectors --control arc \
                   --out '$REPO/results_arc/'; \
               done; \
               echo '=== all models done'"
