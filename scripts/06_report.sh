#!/usr/bin/env bash
# Step 6 - collect results into a table and a figure. Run on the LOGIN node.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source .env

python -m src.report --results results/ --sizes configs/pythia_sizes.json
python -m src.plots  --results results/ --sizes configs/pythia_sizes.json \
                     --out figures/figure_E_vs_scale.pdf

cat <<'MSG'

Deliverables to send back:
  figures/results.csv
  figures/results_table.tex
  figures/figure_E_vs_scale.pdf
  figures/distractor_sample.csv   (with the 'ok' column filled in)
  logs/                           (all of it)
MSG
