"""Construction of the two strata (Section 7.4).

S_low   questions whose commonly asserted answer is false.  TruthfulQA is built
        around exactly this property, so its MC1 split is used directly.
S_high  questions matched in format, topic-breadth and length whose commonly
        asserted answer is true.  Built from TriviaQA by attaching distractors
        drawn from the answer pool, so that both strata are k-way multiple
        choice with the same k distribution.

Matching is on question length and candidate-set size.  Topic matching is only
approximate; report the residual imbalance rather than claiming it away.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Item:
    qid: str
    question: str
    candidates: list[str]
    gold_index: int
    stratum: str
    meta: dict = field(default_factory=dict)

    @property
    def c_w(self) -> np.ndarray:
        v = np.zeros(len(self.candidates))
        v[self.gold_index] = 1.0
        return v


def load_truthfulqa_mc1(limit: Optional[int] = None, seed: int = 0) -> list[Item]:
    """S_low. Requires `datasets`; TruthfulQA MC1 has exactly one correct choice."""
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice")["validation"]
    rng = random.Random(seed)
    items: list[Item] = []
    for i, row in enumerate(ds):
        choices = list(row["mc1_targets"]["choices"])
        labels = list(row["mc1_targets"]["labels"])
        gold = int(np.argmax(labels))
        order = list(range(len(choices)))
        rng.shuffle(order)                       # remove position bias
        items.append(
            Item(
                qid=f"tqa-{i}",
                question=row["question"],
                candidates=[choices[j] for j in order],
                gold_index=order.index(gold),
                stratum="low",
            )
        )
    return items[:limit] if limit else items


def load_triviaqa_control(
    n_options_dist: list[int], limit: Optional[int] = None, seed: int = 0
) -> list[Item]:
    """S_high. Distractors are sampled from other questions' answers.

    Distractors are filtered for string overlap with the gold answer; this is a
    crude type check and should be inspected manually on a sample.
    """
    from datasets import load_dataset

    # a slice is plenty: only a few thousand candidates are needed for matching
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation[:5000]")
    rng = random.Random(seed)
    pool = [row["answer"]["value"] for row in ds]
    items: list[Item] = []
    for i, row in enumerate(ds):
        gold = row["answer"]["value"]
        k = rng.choice(n_options_dist)
        distractors: list[str] = []
        while len(distractors) < k - 1:
            cand = rng.choice(pool)
            if cand.lower() == gold.lower() or cand.lower() in gold.lower():
                continue
            if cand in distractors:
                continue
            distractors.append(cand)
        candidates = distractors + [gold]
        rng.shuffle(candidates)
        items.append(
            Item(
                qid=f"tvqa-{i}",
                question=row["question"],
                candidates=candidates,
                gold_index=candidates.index(gold),
                stratum="high",
            )
        )
        if limit and len(items) >= limit * 4:      # oversample, then match
            break
    return items


def load_arc_control(
    limit: Optional[int] = None, seed: int = 0, config: str = "ARC-Easy"
) -> list[Item]:
    """S_high with human-written distractors.

    Sampling distractors from an answer pool (``load_triviaqa_control``) produces
    options of the wrong type - a question about a king offered "Transistor" as an
    alternative - so the control stratum becomes a type-matching task rather than a
    factual one, and its accuracy is inflated. ARC supplies distractors written by
    people to be plausible, which is what the matched comparison requires.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local = os.path.join(repo, "data", "arc_easy_validation.json")
    if os.path.exists(local):                     # preferred: no Hub, no cache
        print(f"[load_arc_control] reading {local}")
        with open(local) as f:
            ds = json.load(f)
    else:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", config)["validation"]

    rng = random.Random(seed)
    items: list[Item] = []
    for i, row in enumerate(ds):
        if isinstance(row.get("choices"), dict):   # HF schema
            choices = list(row["choices"]["text"])
            labels = list(row["choices"]["label"])
        else:                                      # exported JSON schema
            choices = list(row["choices"])
            labels = list(row["labels"])
        key = row["answerKey"]
        if key not in labels or len(choices) < 3:
            continue                       # a few malformed rows
        gold = labels.index(key)
        order = list(range(len(choices)))
        rng.shuffle(order)
        items.append(
            Item(
                qid=f"arc-{i}",
                question=row["question"],
                candidates=[choices[j] for j in order],
                gold_index=order.index(gold),
                stratum="high",
                meta={"source": config},
            )
        )
        if limit and len(items) >= limit * 4:
            break
    return items


def truncate_candidates(items: list[Item], k: int = 4, seed: int = 0) -> list[Item]:
    """Reduce every item to exactly k candidates, always keeping the gold answer.

    TruthfulQA items carry 4 to 13 choices while ARC items carry 4, so matching on
    candidate count discards most of the sample.  Fixing k for both strata equalises
    the chance level exactly (1/k), removes k as a confound in E, and lets the strata
    be matched on question length alone.  Items with fewer than k candidates are
    dropped; the count of dropped items is printed.
    """
    rng = random.Random(seed)
    out, dropped = [], 0
    for it in items:
        if len(it.candidates) < k:
            dropped += 1
            continue
        gold = it.candidates[it.gold_index]
        distractors = [c for i, c in enumerate(it.candidates) if i != it.gold_index]
        keep = rng.sample(distractors, k - 1) + [gold]
        rng.shuffle(keep)
        out.append(
            Item(qid=it.qid, question=it.question, candidates=keep,
                 gold_index=keep.index(gold), stratum=it.stratum, meta=dict(it.meta))
        )
    if dropped:
        print(f"[truncate_candidates] dropped {dropped} item(s) with fewer than {k} choices")
    return out


def match_strata(
    low: list[Item], high: list[Item], n_per_stratum: int, seed: int = 0,
    match_on_k: bool = False,
) -> tuple[list[Item], list[Item]]:
    """Match S_high to S_low on question-length quartile, and optionally on k.

    Matching on candidate count as well is tempting but costly: TruthfulQA items
    carry 4 to 13 choices and ARC items carry 4, so an exact match discards most of
    the sample.  Equalising k by subsampling TruthfulQA's distractors is worse still,
    because the imitative false answer is one of the distractors and dropping it
    removes the property that defines the stratum.  The default therefore matches on
    length alone and reports the residual difference in k, which is accounted for by
    using the per-item chance level 1/k rather than a single pooled baseline.
    """
    rng = random.Random(seed)

    def key(it: Item):
        bucket = len(it.question) // 25
        return (bucket, len(it.candidates)) if match_on_k else (bucket,)

    low_sample = rng.sample(low, min(n_per_stratum, len(low)))
    buckets: dict[tuple[int, int], list[Item]] = {}
    for it in high:
        buckets.setdefault(key(it), []).append(it)

    matched, unmatched = [], 0
    for it in low_sample:
        pool = buckets.get(key(it))
        if pool:
            matched.append(pool.pop())
        else:
            unmatched += 1
    if unmatched:
        print(f"[match_strata] {unmatched} low items had no match; "
              f"S_high is smaller by that many. Report this imbalance.")
    if matched:
        k_lo = sum(len(i.candidates) for i in low_sample) / len(low_sample)
        k_hi = sum(len(i.candidates) for i in matched) / len(matched)
        print(f"[match_strata] mean candidates: S_low={k_lo:.2f}  S_high={k_hi:.2f} "
              f"(chance levels 1/k differ accordingly; report both)")
    return low_sample, matched
