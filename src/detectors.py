"""Type-(B) error detectors, for prediction P3 (Section 7.4).

Each detector returns a score per item, higher meaning 'more likely wrong'.
All are functions of the corpus alone (Section 6.1), so the prediction is high
AUROC on S_high, near-chance on S_low.

Detectors
---------
max_prob_margin     1 - top-1 confidence; the trivial baseline.
predictive_entropy  entropy of C_M over the candidate set.
self_evaluation     the model's own P(True) for its chosen answer.
semantic_entropy    entropy over meaning-clusters of sampled generations
                    (Farquhar et al., 2024), clustered by bidirectional entailment.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

SELF_EVAL_TEMPLATE = (
    "Question: {question}\n"
    "Proposed answer: {answer}\n"
    "Is the proposed answer true? Answer True or False.\n"
    "Answer:"
)


def max_prob_margin(c_m: Sequence[float]) -> float:
    return float(1.0 - np.max(c_m))


def predictive_entropy(c_m: Sequence[float]) -> float:
    p = np.asarray(c_m, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


class SelfEvaluation:
    """P(False) under the model for its own top answer."""

    def __init__(self, scorer):          # a ConfidenceScorer
        self.scorer = scorer

    def __call__(self, question: str, answer: str) -> float:
        prompt = SELF_EVAL_TEMPLATE.format(question=question.strip(), answer=answer.strip())
        lp_true, _ = self.scorer._candidate_logprob(prompt, "True")
        lp_false, _ = self.scorer._candidate_logprob(prompt, "False")
        m = max(lp_true, lp_false)
        p_true = np.exp(lp_true - m) / (np.exp(lp_true - m) + np.exp(lp_false - m))
        return float(1.0 - p_true)


class SemanticEntropy:
    """Discrete semantic entropy over sampled generations.

    Samples M continuations, clusters them by bidirectional entailment under an
    NLI model, and returns the entropy of the cluster distribution.  This is the
    discrete approximation; see the original paper for the length-normalised
    variant.
    """

    def __init__(self, scorer, nli_model_name: str = "microsoft/deberta-large-mnli",
                 n_samples: int = 10, temperature: float = 1.0, device: str = "cuda"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.scorer = scorer
        self.n_samples = n_samples
        self.temperature = temperature
        self.device = device
        self.nli_tok = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.nli.to(device).eval()
        # label order for MNLI checkpoints is (contradiction, neutral, entailment)
        self.entail_idx = 2

    @torch.no_grad()
    def _entails(self, premise: str, hypothesis: str) -> bool:
        enc = self.nli_tok(premise, hypothesis, return_tensors="pt", truncation=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        pred = int(self.nli(**enc).logits.argmax(dim=-1).item())
        return pred == self.entail_idx

    def _cluster(self, generations: list[str]) -> list[int]:
        clusters: list[list[str]] = []
        assignment: list[int] = []
        for g in generations:
            placed = False
            for ci, members in enumerate(clusters):
                rep = members[0]
                if self._entails(rep, g) and self._entails(g, rep):
                    members.append(g)
                    assignment.append(ci)
                    placed = True
                    break
            if not placed:
                clusters.append([g])
                assignment.append(len(clusters) - 1)
        return assignment

    @torch.no_grad()
    def _sample(self, question: str, max_new_tokens: int = 24) -> list[str]:
        from src.confidence import PROMPT_TEMPLATE

        prompt = PROMPT_TEMPLATE.format(question=question.strip())
        ids = self.scorer.tokenizer(prompt, return_tensors="pt").input_ids.to(self.scorer.device)
        out = self.scorer.model.generate(
            ids,
            do_sample=True,
            temperature=self.temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=self.n_samples,
            pad_token_id=self.scorer.tokenizer.pad_token_id,
        )
        texts = self.scorer.tokenizer.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
        return [t.strip().split("\n")[0] for t in texts]

    def __call__(self, question: str) -> float:
        gens = self._sample(question)
        assignment = self._cluster(gens)
        _, counts = np.unique(assignment, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log(p)).sum())


def evaluate_detector(
    score_fn: Callable[[dict], float], records: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, is_error) for AUROC; call once per stratum, never pooled."""
    scores = np.array([score_fn(r) for r in records], dtype=float)
    errors = np.array([0 if r["correct"] else 1 for r in records], dtype=int)
    return scores, errors
