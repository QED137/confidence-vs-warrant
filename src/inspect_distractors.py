"""Export a sample of constructed S_high items for manual inspection.

This step is mandatory, not optional. Distractors for the control stratum are
sampled from other questions' answers with only a crude string filter, so some
will be of the wrong type ("Which river ...?" with a person as a distractor).
Implausible distractors inflate accuracy on S_high and weaken the matching that
the whole comparison rests on.

    python -m src.inspect_distractors --n 50 --out figures/distractor_sample.csv

Open the CSV, read every row, and mark column `ok` with y or n. If more than
about 10% are bad, tighten the filter in src/data.py before running anything
else, and record the inspected rate in the paper.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter

from src import data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/distractor_sample.csv")
    args = ap.parse_args()

    low = data.load_truthfulqa_mc1(seed=args.seed)
    k_dist = list(Counter(len(it.candidates) for it in low).elements())
    high = data.load_triviaqa_control(k_dist, limit=args.n, seed=args.seed)

    rng = random.Random(args.seed)
    sample = rng.sample(high, min(args.n, len(high)))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "question", "gold", "distractors", "ok"])
        for it in sample:
            gold = it.candidates[it.gold_index]
            distractors = [c for i, c in enumerate(it.candidates) if i != it.gold_index]
            w.writerow([it.qid, it.question, gold, " | ".join(distractors), ""])

    print(f"wrote {args.out} with {len(sample)} rows")
    print("Read every row and fill the 'ok' column with y or n.")
    print("If the bad rate exceeds ~10%, tighten the filter in src/data.py first.")


if __name__ == "__main__":
    main()
