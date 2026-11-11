# Epistemic calibration: experiment scaffold

> **Running this on a cluster?** Follow [RUNBOOK.md](RUNBOOK.md) step by step.
> It needs no knowledge of the paper. This README is the reference for what
> the code computes and why.

Code for the study proposed in Section 7 of *What Does a Language Model Know?*
It measures **E**, the gap between model confidence and world validity, separately
on two strata, and decomposes it into corpus bias and learning error (eq. 8).

```
pip install -r requirements.txt
python -m pytest tests -q                       # 8 tests, no GPU or network needed
python -m src.run --model EleutherAI/pythia-160m --smoke        # 1-2 min, checks the pipeline
python -m src.run --model EleutherAI/pythia-1.4b --n 300 --detectors --threads 8
python -m src.plots --results results/ --sizes pythia_sizes.json
```

## What is measured

| quantity | meaning | where |
|---|---|---|
| `E` | `E_{y~Q}|C_M - C_W|` (Definition 2) | `metrics.epistemic_calibration_error` |
| corpus bias / learning error | the two terms of eq. (8) | `metrics.gap_decomposition` |
| `ECE` | standard calibration, for contrast | `metrics.expected_calibration_error` |
| detector AUROC | prediction P3 | `detectors`, `metrics.auroc` |

## Strata

- **S_low** — TruthfulQA MC1, whose questions are built so that the commonly
  asserted answer is false.
- **S_high** — TriviaQA with distractors, matched to S_low on question-length
  quartile and candidate-set size.

Answer order is shuffled per item to remove position bias. Matching is only
approximate on topic; `match_strata` prints the residual imbalance, and that
number belongs in the paper.

## The four predictions

1. **P1 dissociation** — `E(S_low) >> E(S_high)` while agreement between `C_M`
   and `P̂_text` is comparable across strata.
2. **P2 scale dissociation** — across a model family, learning error falls with
   scale in both strata; corpus bias is flat; so `E(S_low)` does not fall.
3. **P3 detector dissociation** — detector AUROC is high on S_high and near 0.5
   on S_low. **This is the falsifiable one.**
4. **P4 grounding** — retrieval reduces `E(S_low)` in proportion to source
   faithfulness. Not implemented here; add a retrieval condition.

## Estimating P̂_text

`src/assertion.py` takes an injected `count_fn`, so any n-gram index works.
This only estimates the right quantity when the index is over the model's own
training corpus — which is why the Pythia family (trained on the public Pile)
is the natural choice. Report the index version in run metadata, and report the
`agreement` diagnostic between the count-based and human-baseline estimators
rather than picking one.

Until an index is wired in, `p_txt` stays `None`, `E` is still computed, and the
decomposition panel of the figure is left empty.

## Notes and caveats

- Run in **float32**, not bfloat16. At bf16 the relative resolution is ~1e-3 and
  near-ties become kernel-dependent (Remark 2). `tied_at_top_rate` reports how
  often the top two candidates fall within `RESOLUTION_FLOOR`; if it is
  non-trivial on either stratum, say so in the paper.
- Never pool strata. Pooling reintroduces exactly the aggregation the paper
  objects to in Section 7.1.
- Check TruthfulQA contamination for any instruction-tuned model before
  including it; a model trained on the answer key answers correctly for the
  wrong reason (Section 7.5).
- `SemanticEntropy` assumes the MNLI label order `(contradiction, neutral,
  entailment)`. Verify against your checkpoint's `config.id2label`.
- Distractor quality in `load_triviaqa_control` is the weakest link — the type
  filter is crude string matching. Inspect 50 items by hand before trusting the
  S_high numbers.

## Running on a 16 GB laptop (CPU only)

Candidates are scored in one padded forward pass per item, so 600 items is a few
thousand token positions in total. Feasible sizes, float32, `--detectors` on:

| model | weights | wall clock, 600 items |
|---|---|---|
| pythia-160m | 0.6 GB | ~5 min |
| pythia-410m | 1.6 GB | ~12 min |
| pythia-1.4b | 5.6 GB | ~35 min |
| pythia-2.8b | 11 GB | will swap on 16 GB — run on a cluster |

Three sizes still span an order of magnitude, which is enough to see the P2
trend; add 2.8b and 6.9b later for the published figure.

Set `--threads` to your physical core count (not the hyperthread count).
Keep float32: bf16 on CPU is both slower and below the resolution floor.

`--semantic-entropy` is much heavier — 10 generations plus pairwise NLI calls per
item — and is a GPU job. The self-evaluation and max-probability detectors
already test P3 without it.
