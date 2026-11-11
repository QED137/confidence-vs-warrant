#!/usr/bin/env bash
# Step 3 - MANUAL QA. Do not skip.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source .env

python -m src.inspect_distractors --n 50 --out figures/distractor_sample.csv

cat <<'MSG'

ACTION REQUIRED
  1. Open figures/distractor_sample.csv
  2. Read all 50 rows; mark column 'ok' with y or n
  3. If more than ~5 rows are bad, STOP and report back before continuing
  4. Keep the filled CSV - the inspected bad rate is reported in the paper

Next: sbatch scripts/04_run_cpu.sbatch
MSG
