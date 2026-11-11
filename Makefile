.PHONY: setup fetch export-arc inspect run report paired test clean

setup:      ; bash scripts/01_setup_env.sh
fetch:      ; bash scripts/02_fetch_assets.sh
export-arc: ; bash scripts/08_export_arc.sh
inspect:    ; bash scripts/03_inspect_distractors.sh
run:        ; bash scripts/07_rerun_arc_control.sh
report:     ; python -m src.report --results results_arc/ --sizes configs/pythia_sizes.json --outdir figures_arc
paired:     ; python -m src.paired --results results_arc/ --a EleutherAI/pythia-160m --b EleutherAI/pythia-6.9b
test:       ; . .venv/bin/activate && python -m pytest tests -q
clean:      ; rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache

# scripts/04_run_cpu.sbatch uses the superseded TriviaQA control and has no
# target here on purpose. See README, Known limitations.
