"""Paired across-model comparison.

Every model answers the same items, so comparing two models with independent
standard errors throws away most of the available power: item difficulty varies
far more than the model-to-model difference does.  This module pairs by `qid`
and tests the per-item difference instead.

    python -m src.paired --results results/ --a EleutherAI/pythia-160m \\
                                            --b EleutherAI/pythia-2.8b

Reports, per stratum:
  * mean paired change in C_M(true answer), with a bootstrap CI
  * the same quantity against chance (1/k) for each model
  * AUROC of top-1 confidence for error detection, with a bootstrap CI,
    so that 'below chance' can be assessed rather than eyeballed
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from src import metrics


def _load(results_dir: str) -> dict[str, dict]:
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            blob = json.load(f)
        out[blob["summary"]["model"]] = blob
    return out


def _by_qid(records: list[dict]) -> dict[str, dict]:
    return {r["qid"]: r for r in records if "qid" in r}


def _conf_gold(rec: dict) -> float:
    c_m, c_w = np.asarray(rec["c_m"]), np.asarray(rec["c_w"])
    return float(c_m[int(np.argmax(c_w))])


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 0,
                 statistic=np.mean) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    stats = statistic(values[idx], axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def auroc_ci(scores: np.ndarray, labels: np.ndarray, n_boot: int = 2000,
             seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = metrics.auroc(scores, labels)
    boots = []
    n = len(scores)
    for _ in range(n_boot):
        i = rng.integers(0, n, size=n)
        a = metrics.auroc(scores[i], labels[i])
        if not np.isnan(a):
            boots.append(a)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--a", required=True, help="smaller model")
    ap.add_argument("--b", required=True, help="larger model")
    args = ap.parse_args()

    runs = _load(args.results)
    for name in (args.a, args.b):
        if name not in runs:
            raise SystemExit(f"no results for {name}; have: {list(runs)}")

    for stratum in ("low", "high"):
        ra = _by_qid(runs[args.a]["records"].get(stratum, []))
        rb = _by_qid(runs[args.b]["records"].get(stratum, []))
        shared = sorted(set(ra) & set(rb))
        if not shared:
            print(f"\nS_{stratum}: no shared qids (were the runs built with the same seed?)")
            continue

        ga = np.array([_conf_gold(ra[q]) for q in shared])
        gb = np.array([_conf_gold(rb[q]) for q in shared])
        diff = gb - ga
        lo, hi = bootstrap_ci(diff)
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))

        print(f"\n=== S_{stratum}  ({len(shared)} paired items)")
        print(f"  C_M(true) {args.a.split('/')[-1]}: {ga.mean():.4f}")
        print(f"  C_M(true) {args.b.split('/')[-1]}: {gb.mean():.4f}")
        print(f"  paired change: {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  t={t:+.2f}")
        verdict = "significant" if (lo > 0 or hi < 0) else "not significant"
        print(f"  -> {verdict} at the 5% level")

        for label, run in ((args.a, ra), (args.b, rb)):
            g = np.array([_conf_gold(run[q]) for q in shared])
            chance = np.array([1.0 / len(run[q]["c_m"]) for q in shared])
            d = g - chance
            clo, chi = bootstrap_ci(d)
            mark = "above chance" if clo > 0 else ("below chance" if chi < 0 else "at chance")
            print(f"  {label.split('/')[-1]:<14} vs chance: {d.mean():+.4f} "
                  f"CI [{clo:+.4f}, {chi:+.4f}]  -> {mark}")

        for label, run in ((args.a, ra), (args.b, rb)):
            s = np.array([1.0 - max(run[q]["c_m"]) for q in shared])
            y = np.array([0 if run[q]["correct"] else 1 for q in shared])
            a, alo, ahi = auroc_ci(s, y)
            mark = "below 0.5" if ahi < 0.5 else ("above 0.5" if alo > 0.5 else "spans 0.5")
            print(f"  {label.split('/')[-1]:<14} AUROC(max-prob): {a:.4f} "
                  f"CI [{alo:.4f}, {ahi:.4f}]  -> {mark}")


if __name__ == "__main__":
    main()
