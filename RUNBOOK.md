# RUNBOOK — epistemic calibration study

For whoever is running this on the cluster. Follow the steps in order. Each one
prints what to do next. Total hands-on time is about 90 minutes; the compute
runs unattended.

**What this measures.** For each model we ask multiple-choice questions from two
groups and record how much probability the model puts on the true answer. Group
one (`S_low`) contains questions where the popular answer is wrong; group two
(`S_high`) contains ordinary factual questions. The study asks whether the gap
between the two grows, shrinks, or stays flat as models get larger. No knowledge
of the underlying paper is needed to run it.

---

## Step 0 — get the code

```bash
git clone <repo-url> ecal && cd ecal
```

Requires Python 3.10+. A GPU is **not** needed except for the optional step 5.

## Step 1 — environment (login node, ~10 min)

```bash
bash scripts/01_setup_env.sh
```

Creates `.venv`, installs dependencies, writes `.env`, and runs 8 unit tests.
**All 8 tests must pass before continuing.** If they fail, stop and report.

## Step 2 — download models and data (login node, ~30 min, ~30 GB)

```bash
bash scripts/02_fetch_assets.sh
```

Compute nodes usually have no internet, so everything is cached into `./hf_cache`
now. If your `$HOME` has a small quota, place the repo on scratch first.

## Step 3 — inspect distractors (manual, ~20 min) — **do not skip**

```bash
bash scripts/03_inspect_distractors.sh
```

Opens nothing; it writes `figures/distractor_sample.csv`. Open it in a
spreadsheet. Each row is a question, its true answer, and the wrong answers the
code generated. For each row ask: *are the wrong answers of the right kind?*
A question asking for a river should not have a person as a wrong answer.

Mark column `ok` with `y` or `n`. If more than 5 of the 50 are bad, **stop and
report back** — the code needs a fix before the numbers mean anything. Save the
filled file; it is a deliverable.

## Step 4 — main run (SLURM, ~4 h unattended)

```bash
sbatch scripts/04_run_cpu.sbatch
squeue -u $USER          # check progress
```

Five array tasks, one per model, CPU only, 8 cores and 32 GB each. Output lands
in `results/` as one JSON per model, logs in `logs/`.

If a task is killed for memory, that model is too large for the allocation;
raise `--mem` or drop it from `configs/models.txt` and note which.

## Step 5 — semantic entropy (OPTIONAL, GPU, ~12 h)

```bash
sbatch scripts/05_run_gpu_semantic_entropy.sbatch
```

Only if a GPU is available and there is time. Results go to `results_se/` and do
not overwrite step 4. The study is complete without it.

## Step 6 — report (login node, ~1 min)

```bash
bash scripts/06_report.sh
```

Prints a verdict on each prediction and writes the table and figure.

---

## Deliverables

Send back, and commit to the repo:

- `figures/results.csv`
- `figures/results_table.tex`
- `figures/figure_E_vs_scale.pdf`
- `figures/distractor_sample.csv` (with `ok` filled in)
- `results/*.json`
- the whole `logs/` directory

## Troubleshooting

| Symptom | Fix |
|---|---|
| tests fail in step 1 | stop, send the output; do not continue |
| `OSError: ... offline mode` in step 4 | step 2 did not finish; rerun it on the login node |
| task killed, exit 137 | out of memory: raise `--mem` in the sbatch file |
| disk quota exceeded | move the repo to scratch, rerun step 1 |
| `results/` empty at step 6 | check `logs/*.err` for the failed array task |
| run is very slow | check `--threads` matches `--cpus-per-task` |

## Rules that matter

- **Use float32.** Do not switch to bfloat16 to save time. The study measures
  small differences in probability and bfloat16 is too coarse for them.
- **Do not merge the two groups.** Every number is reported per group.
- **Do not change the prompt template** in `src/confidence.py` between models.
- **Do not silently drop a model.** If one fails, say which and why.
- Any `NOT SUPPORTED` verdict in step 6 is a real result, not an error. Report
  the numbers as they come out.
