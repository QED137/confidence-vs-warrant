"""Collect run outputs into a CSV, a LaTeX table, and a verdict on P1-P3.

    python -m src.report --results results/ --sizes configs/pythia_sizes.json

Writes figures/results.csv and figures/results_table.tex, and prints whether
each prediction of Section 7.4 is supported by the runs found.  The verdicts are
descriptive, not significance tests; treat them as a reading aid.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os


FIELDS = [
    "model", "n_params", "stratum", "n", "E", "E_se",
    "corpus_bias", "learning_error", "ECE", "accuracy",
    "mean_k", "mean_top1_conf", "E_uniform", "E_rel", "mean_conf_gold", "mean_conf_gold_se", "chance_conf",
    "tied_at_top_rate", "auroc_max_prob", "auroc_self_eval", "auroc_semantic_entropy",
]


def load_summaries(results_dir: str) -> list[dict]:
    """Load each run's summary, recomputing it from the stored per-item records
    when the summary predates a metric (so new diagnostics need no re-run)."""
    from src import metrics

    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            blob = json.load(f)
        summary, records = blob["summary"], blob.get("records", {})
        for stratum, recs in records.items():
            if not recs:
                continue
            if "E_rel" in summary["strata"].get(stratum, {}):
                continue
            fresh = metrics.aggregate(stratum, recs).as_row()
            fresh.update({k: v for k, v in summary["strata"][stratum].items()
                          if k.startswith("auroc") or k == "tied_at_top_rate"})
            summary["strata"][stratum] = fresh
        out.append(summary)
    return out


def to_rows(summaries: list[dict], sizes: dict) -> list[dict]:
    rows = []
    for s in summaries:
        for stratum in ("low", "high"):
            if stratum not in s["strata"]:
                continue
            row = {"model": s["model"], "n_params": sizes.get(s["model"])}
            row.update(s["strata"][stratum])
            rows.append({k: row.get(k) for k in FIELDS})
    rows.sort(key=lambda r: (r["n_params"] or 0, r["stratum"]))
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def write_latex(rows: list[dict], path: str) -> None:
    def fmt(v, nd=3):
        return "--" if v is None else f"{v:.{nd}f}"

    lines = [
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Model & Stratum & $E$ & ECE & Acc. & AUROC (self-eval) \\", r"\midrule",
    ]
    for r in rows:
        name = r["model"].split("/")[-1].replace("_", r"\_")
        stratum = r"$\mathcal{S}_{\mathrm{low}}$" if r["stratum"] == "low" \
            else r"$\mathcal{S}_{\mathrm{high}}$"
        lines.append(
            f"{name} & {stratum} & {fmt(r['E'])} & {fmt(r['ECE'])} & "
            f"{fmt(r['accuracy'])} & {fmt(r.get('auroc_self_eval'))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def check_predictions(rows: list[dict]) -> list[str]:
    msgs = []
    low = [r for r in rows if r["stratum"] == "low"]
    high = [r for r in rows if r["stratum"] == "high"]
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r["stratum"]] = r

    # P1 - per model, not averaged: an average hides a crossover
    # E is only comparable across strata when k matches, since its indifference
    # baseline is 2(k-1)/k^2.  Fall back to E_rel = E / baseline when it does not.
    use_rel = any(abs((d["low"].get("mean_k") or 0) - (d["high"].get("mean_k") or 0)) > 0.1
                  for d in by_model.values() if "low" in d and "high" in d)
    fld = "E_rel" if use_rel else "E"
    pairs = [(m, d["low"][fld], d["high"][fld]) for m, d in by_model.items()
             if "low" in d and "high" in d and d["low"].get(fld) is not None]
    pairs.sort(key=lambda t: by_model[t[0]]["low"]["n_params"] or 0)
    if pairs:
        holds = [m for m, el, eh in pairs if el > eh]
        if use_rel:
            msgs.append("P1 dissociation, per model, using E_rel = E / indifference "
                        "baseline (strata differ in k, so raw E is not comparable):")
        else:
            msgs.append("P1 dissociation, per model (E_low > E_high expected):")
        for m, el, eh in pairs:
            mark = "yes" if el > eh else "NO "
            msgs.append(f"    {mark}  {m.split('/')[-1]:<14} low={el:.3f} high={eh:.3f}")
        if len(holds) == len(pairs):
            msgs.append("    [SUPPORTED at every scale]")
        elif holds:
            msgs.append(f"    [SUPPORTED only above {pairs[len(pairs)-len(holds)-1][0].split('/')[-1]}"
                        f" - report the crossover, do not average it away]")
        else:
            msgs.append("    [NOT SUPPORTED]")

    # P2 - the central prediction
    # P2 on the margin above each stratum's own chance level.  This is the only
    # cross-stratum statistic that is unaffected by k, and unlike E or E_rel it
    # records direction rather than dispersion.
    def margins(grp):
        s = sorted([x for x in grp if x.get("mean_conf_gold") is not None and x["n_params"]],
                   key=lambda x: x["n_params"])
        return s, [x["mean_conf_gold"] - (x.get("chance_conf") or float("nan")) for x in s]

    s_lo, m_lo = margins(low)
    s_hi, m_hi = margins(high)
    if len(m_lo) >= 3 and len(m_hi) >= 3:
        d_lo, d_hi = m_lo[-1] - m_lo[0], m_hi[-1] - m_hi[0]
        verdict = "SUPPORTED" if (d_hi > 0.02 and d_lo <= 0.005) else "NOT SUPPORTED"
        msgs.append(
            f"P2 scale dissociation [margin above chance]: "
            f"d(low)={d_lo:+.3f}  d(high)={d_hi:+.3f}"
            f"  over {s_lo[0]['n_params']/1e6:.0f}M-{s_lo[-1]['n_params']/1e9:.1f}B  [{verdict}]"
        )
        if d_lo < 0 < d_hi:
            msgs.append("    the two strata move in OPPOSITE directions with scale")
        sized = s_lo
        if all(r.get("learning_error") is not None for r in sized):
            cb = [r["corpus_bias"] for r in sized]
            le = [r["learning_error"] for r in sized]
            msgs.append(f"    decomposition: corpus bias {cb[0]:.3f}->{cb[-1]:.3f}, "
                        f"learning error {le[0]:.3f}->{le[-1]:.3f}")
        else:
            msgs.append("    decomposition: no P_text estimates (see src/assertion.py)")
        msgs.append(f"    [E_rel for reference: d(low)={sized[-1]['E_rel']-sized[0]['E_rel']:+.3f} "
                    f"d(high)={s_hi[-1]['E_rel']-s_hi[0]['E_rel']:+.3f}; E_rel measures "
                    f"dispersion, not direction, so it can fall while the margin worsens]")

    # margin above each stratum's own chance level: comparable even when k differs
    for name, grp in (("low", low), ("high", high)):
        s = sorted([x for x in grp if x.get("mean_conf_gold") is not None and x["n_params"]],
                   key=lambda x: x["n_params"])
        if len(s) >= 2:
            marg = [x["mean_conf_gold"] - (x.get("chance_conf") or float("nan")) for x in s]
            msgs.append(f"    margin above chance on S_{name}: "
                        + " -> ".join(f"{m:+.3f}" for m in marg))

    # P3 - a detector that cannot discriminate anywhere supports nothing
    for det in ("auroc_max_prob", "auroc_self_eval", "auroc_semantic_entropy"):
        l = [r[det] for r in low if r.get(det) is not None]
        h = [r[det] for r in high if r.get(det) is not None]
        if not (l and h):
            continue
        a_l, a_h = sum(l) / len(l), sum(h) / len(h)
        if a_h < 0.60:
            verdict = "INCONCLUSIVE - detector is at chance on BOTH strata, so it " \
                      "tests nothing at this scale"
        elif a_h - a_l > 0.10:
            verdict = "SUPPORTED"
        else:
            verdict = "NOT SUPPORTED"
        msgs.append(f"P3 [{det}]: AUROC low={a_l:.3f} high={a_h:.3f}  [{verdict}]")
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--sizes", default="configs/pythia_sizes.json")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.sizes) as f:
        sizes = json.load(f)
    summaries = load_summaries(args.results)
    if not summaries:
        raise SystemExit(f"no result files in {args.results}/ - did step 4 finish?")

    rows = to_rows(summaries, sizes)
    write_csv(rows, os.path.join(args.outdir, "results.csv"))
    write_latex(rows, os.path.join(args.outdir, "results_table.tex"))

    print(f"\n{len(summaries)} model(s) found\n")
    for line in check_predictions(rows):
        print("  " + line)
    print(f"\nwrote {args.outdir}/results.csv and {args.outdir}/results_table.tex")


if __name__ == "__main__":
    main()
