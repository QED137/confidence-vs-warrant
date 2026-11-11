"""Run the epistemic-calibration study for one model.

    python -m src.run --model EleutherAI/pythia-1.4b --n 300 --out results/

Produces results/<model>.json with per-item records and per-stratum aggregates.
Strata are reported separately and never pooled (Section 7.4).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch

from src import data, detectors, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300, help="items per stratum")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads; 0 = leave default")
    ap.add_argument("--smoke", action="store_true", help="20 items per stratum, for a first run")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-options", type=int, default=0,
                    help="force both strata to exactly this many candidates; "
                         "0 (default) keeps native counts. Truncating S_low can "
                         "discard the imitative distractor, so use only as a check")
    ap.add_argument("--match-k", action="store_true",
                    help="also require equal candidate counts when matching strata")
    ap.add_argument("--control", default="arc", choices=["arc", "triviaqa"],
                    help="source of S_high; arc has human-written distractors "
                         "and is the default, triviaqa reproduces the earlier runs")
    ap.add_argument("--detectors", action="store_true", help="run P3 detector conditions")
    ap.add_argument("--no-length-norm", action="store_true",
                    help="score by summed log-prob instead of per-token mean; "
                         "robustness check for answer-length confounds")
    ap.add_argument("--semantic-entropy", action="store_true", help="slow; needs an NLI model")
    args = ap.parse_args()

    if args.smoke:
        args.n = 20
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- strata ---------------------------------------------------------
    low_all = data.load_truthfulqa_mc1(seed=args.seed)
    k_dist = list(Counter(len(it.candidates) for it in low_all).elements())
    if args.control == "arc":
        high_all = data.load_arc_control(limit=args.n, seed=args.seed)
    else:
        high_all = data.load_triviaqa_control(k_dist, limit=args.n, seed=args.seed)
    if args.k_options:
        low_all = data.truncate_candidates(low_all, args.k_options, seed=args.seed)
        high_all = data.truncate_candidates(high_all, args.k_options, seed=args.seed)
    low, high = data.match_strata(low_all, high_all, args.n, seed=args.seed,
                                  match_on_k=args.match_k)
    print(f"strata: S_low={len(low)}  S_high={len(high)}")

    # ---- model ----------------------------------------------------------
    from src.confidence import ConfidenceScorer

    scorer = ConfidenceScorer(
        args.model,
        device=args.device,
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
        length_normalise=not args.no_length_norm,
    )

    self_eval = detectors.SelfEvaluation(scorer) if args.detectors else None
    sem_ent = None
    if args.semantic_entropy:
        sem_ent = detectors.SemanticEntropy(scorer, device=args.device)

    # ---- scoring --------------------------------------------------------
    records: dict[str, list[dict]] = {"low": [], "high": []}
    total = len(low) + len(high)
    done = 0
    for items in (low, high):
        for it in items:
            done += 1
            if done % 25 == 0:
                print(f"  scored {done}/{total}", flush=True)
            scored = scorer.score_item(it.question, it.candidates)
            c_m, c_w = scored.c_m, it.c_w
            top = int(np.argmax(c_m))
            rec = {
                "qid": it.qid,
                "stratum": it.stratum,
                "question": it.question,
                "candidates": it.candidates,
                "c_m": c_m.tolist(),
                "c_w": c_w.tolist(),
                "p_txt": None,          # fill from src.assertion when an index is available
                "correct": bool(top == it.gold_index),
                "tied_at_top": scored.tied_at_top,
                "E": metrics.epistemic_calibration_error(c_m, c_w),
            }
            if self_eval is not None:
                rec["self_eval"] = self_eval(it.question, it.candidates[top])
            if sem_ent is not None:
                rec["semantic_entropy"] = sem_ent(it.question)
            records[it.stratum].append(rec)

    # ---- aggregation ----------------------------------------------------
    summary = {"model": args.model, "dtype": args.dtype, "n_per_stratum": args.n, "strata": {}}
    for name, recs in records.items():
        agg = metrics.aggregate(name, recs).as_row()
        agg["tied_at_top_rate"] = float(np.mean([r["tied_at_top"] for r in recs]))
        if any("self_eval" in r for r in recs):
            s, y = detectors.evaluate_detector(lambda r: r["self_eval"], recs)
            agg["auroc_self_eval"] = metrics.auroc(s, y)
        if any("semantic_entropy" in r for r in recs):
            s, y = detectors.evaluate_detector(lambda r: r["semantic_entropy"], recs)
            agg["auroc_semantic_entropy"] = metrics.auroc(s, y)
        s, y = detectors.evaluate_detector(
            lambda r: detectors.max_prob_margin(r["c_m"]), recs
        )
        agg["auroc_max_prob"] = metrics.auroc(s, y)
        summary["strata"][name] = agg

    summary["length_normalised"] = not args.no_length_norm
    summary["control"] = args.control
    summary["k_options"] = args.k_options
    summary["match_on_k"] = args.match_k
    tag = args.model.replace("/", "__")
    if args.no_length_norm:
        tag += "__nolennorm"
    if args.control != "arc":
        tag += f"__{args.control}"
    with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
