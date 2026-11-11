"""Extraction of C_M: model confidence over a candidate answer set.

Scores each candidate by its length-normalised conditional log-probability given
a fixed prompt, then renormalises across the candidate set (Section 7.4).  The
same prompt template is used for every stratum; template sensitivity is a
reported robustness check, not a free parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = "Q: {question}\nA:"

# Below this separation two candidates are treated as tied rather than ranked
# (Remark 2: the resolution floor).  Loosen it if running in bfloat16.
RESOLUTION_FLOOR = 1e-3


@dataclass
class ScoredItem:
    question: str
    candidates: list[str]
    c_m: np.ndarray          # normalised over candidates
    logprobs: np.ndarray     # length-normalised, unnormalised across candidates
    tied_at_top: bool        # top two within RESOLUTION_FLOOR


class ConfidenceScorer:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        length_normalise: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        self.model.to(device).eval()
        self.device = device
        self.length_normalise = length_normalise

    @torch.no_grad()
    def _candidate_logprob(self, prompt: str, candidate: str) -> tuple[float, int]:
        """log p(candidate | prompt), summed over candidate tokens (single pair)."""
        lps, ns = self._batch_logprobs(prompt, [candidate])
        return lps[0], ns[0]

    @torch.no_grad()
    def _batch_logprobs(self, prompt: str, candidates: Sequence[str]) -> tuple[list[float], list[int]]:
        """Score every candidate for one prompt in a single padded forward pass.

        Right padding is used so that prompt tokens always start at index 0 and
        the candidate span begins at n_prompt for every row.
        """
        n_prompt = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        texts = [prompt + " " + c.strip() for c in candidates]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = enc.input_ids.to(self.device)
        attn = enc.attention_mask.to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attn).logits[:, :-1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        targets = input_ids[:, 1:]
        token_lp = logprobs.gather(2, targets.unsqueeze(2)).squeeze(2)   # (B, L-1)

        positions = torch.arange(targets.shape[1], device=self.device).unsqueeze(0)
        real = attn[:, 1:].bool()                       # drop padding
        mask = real & (positions >= n_prompt - 1)       # candidate tokens only

        sums = (token_lp * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return sums.tolist(), counts.tolist()

    def score_item(self, question: str, candidates: Sequence[str]) -> ScoredItem:
        prompt = PROMPT_TEMPLATE.format(question=question.strip())
        sums, counts = self._batch_logprobs(prompt, candidates)
        if self.length_normalise:
            raw = np.array([s / max(n, 1) for s, n in zip(sums, counts)], dtype=float)
        else:
            raw = np.asarray(sums, dtype=float)
        c_m = _softmax(raw)
        top2 = np.sort(c_m)[-2:] if len(c_m) > 1 else np.array([0.0, 1.0])
        return ScoredItem(
            question=question,
            candidates=list(candidates),
            c_m=c_m,
            logprobs=raw,
            tied_at_top=bool(abs(top2[-1] - top2[-2]) < RESOLUTION_FLOOR),
        )


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
