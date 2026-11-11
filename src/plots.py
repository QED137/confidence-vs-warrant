"""The figure for Section 7: E by stratum against model scale, with the
Corollary 2.2 decomposition as stacked components.

    python -m src.plots --results results/ --sizes pythia_sizes.json --out figure.pdf
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load(results_dir: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            out.append(json.load(f)["summary"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--sizes", required=True, help="JSON: {model_name: n_params}")
    ap.add_argument("--out", default="figure_E_vs_scale.pdf")
    args = ap.parse_args()

    with open(args.sizes) as f:
        sizes = json.load(f)
    summaries = load(args.results)
    summaries = [s for s in summaries if s["model"] in sizes]
    summaries.sort(key=lambda s: sizes[s["model"]])
    x = np.array([sizes[s["model"]] for s in summaries], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), constrained_layout=True)

    # left: E by stratum
    ax = axes[0]
    for name, marker in (("low", "o"), ("high", "s")):
        e = np.array([s["strata"][name]["E"] for s in summaries])
        se = np.array([s["strata"][name]["E_se"] for s in summaries])
        label = r"$\mathcal{S}_{\mathrm{low}}$" if name == "low" else r"$\mathcal{S}_{\mathrm{high}}$"
        ax.errorbar(x, e, yerr=se, marker=marker, capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel(r"$E$")
    ax.set_title("Epistemic calibration error by stratum")
    ax.legend(frameon=False)

    # right: decomposition on S_low (needs p_txt estimates)
    ax = axes[1]
    cb = np.array([s["strata"]["low"].get("corpus_bias") or np.nan for s in summaries])
    le = np.array([s["strata"]["low"].get("learning_error") or np.nan for s in summaries])
    if not np.all(np.isnan(cb)):
        ax.stackplot(x, np.nan_to_num(cb), np.nan_to_num(le),
                     labels=["corpus bias", "learning error"], alpha=0.75)
        ax.set_xscale("log")
        ax.legend(frameon=False, loc="upper right")
    else:
        ax.text(0.5, 0.5, "no $\\hat{P}_{\\mathrm{text}}$ estimates\n(see src/assertion.py)",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("parameters")
    ax.set_ylabel("contribution to the bound")
    ax.set_title(r"Decomposition on $\mathcal{S}_{\mathrm{low}}$ (eq. 8)")

    # third panel: probability mass on the true answer (sharpness-free)
    ax = axes[2]
    for name, marker in (("low", "o"), ("high", "s")):
        g = np.array([s["strata"][name].get("mean_conf_gold", np.nan) for s in summaries])
        ge = np.array([s["strata"][name].get("mean_conf_gold_se", 0.0) for s in summaries])
        label = r"$\mathcal{S}_{\mathrm{low}}$" if name == "low" else r"$\mathcal{S}_{\mathrm{high}}$"
        ax.errorbar(x, g, yerr=ge, marker=marker, capsize=3, label=label)
    chance = summaries[0]["strata"]["low"].get("chance_conf")
    if chance:
        ax.axhline(chance, ls="--", lw=1, color="0.5")
        ax.annotate("chance", (x[0], chance), textcoords="offset points",
                    xytext=(2, 4), fontsize=8, color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel(r"$C_M$(true answer)")
    ax.set_title("Mass on the true answer")
    ax.legend(frameon=False)

    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()