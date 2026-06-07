"""
active_sampler.py — Uncertainty-based active learning sampler using MC Dropout.

Strategy:
  1. Enable dropout at inference time (model.train() but no grad)
  2. Run T forward passes over the unlabeled pool
  3. Compute token-level entropy across passes
  4. Select top-K most uncertain samples for annotation

References:
  - Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Tuple
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)

# Number of MC Dropout forward passes
MC_T = 10


def _enable_dropout(model):
    """Switch dropout layers to train mode while keeping BN in eval."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


@torch.no_grad()
def compute_uncertainty_scores(
    model,
    dataset,
    candidate_indices: List[int],
    device,
    batch_size: int = 4,
    mc_passes: int = MC_T,
) -> np.ndarray:
    """
    Compute per-sample uncertainty scores via MC Dropout.

    Returns:
        scores: np.ndarray of shape (len(candidate_indices),)
                Higher = more uncertain = higher priority for labeling.
    """
    model.eval()
    _enable_dropout(model)  # re-enable dropout for MC sampling

    subset  = Subset(dataset, candidate_indices)
    loader  = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
    scores  = []

    for batch in loader:
        pixel_values   = batch["pixel_values"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Collect logits from T passes
        # Shape per pass: (B, seq_len, vocab)
        all_probs = []
        for _ in range(mc_passes):
            out = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            # out.logits: (B, seq_len, vocab)
            probs = F.softmax(out.logits.float(), dim=-1)  # cast to float32
            all_probs.append(probs.cpu())

        # all_probs: list of T tensors, each (B, seq_len, vocab)
        stacked = torch.stack(all_probs, dim=0)  # (T, B, seq_len, vocab)

        # Mean probability across passes
        mean_probs = stacked.mean(dim=0)  # (B, seq_len, vocab)

        # Predictive entropy per token: H[E[p]] = -sum(p_bar * log(p_bar))
        eps     = 1e-10
        entropy = -(mean_probs * (mean_probs + eps).log()).sum(dim=-1)  # (B, seq_len)

        # Only consider answer token positions (labels != -100)
        label_mask = (labels.cpu() != -100).float()   # (B, seq_len)
        answer_entropy = (entropy * label_mask).sum(dim=-1) / (label_mask.sum(dim=-1) + eps)

        scores.extend(answer_entropy.tolist())

    model.eval()  # restore full eval mode
    return np.array(scores)


def select_query_indices(
    scores: np.ndarray,
    candidate_indices: List[int],
    query_size: int,
    strategy: str = "uncertainty",
    rng: np.random.Generator = None,
) -> List[int]:
    """
    Given uncertainty scores for candidate indices, return the top-K
    indices (in terms of original dataset indices) to query for labels.

    strategy: 'uncertainty' | 'random'
    """
    assert len(scores) == len(candidate_indices)

    if strategy == "random":
        if rng is None:
            rng = np.random.default_rng()
        chosen = rng.choice(len(candidate_indices), size=min(query_size, len(candidate_indices)), replace=False)
    elif strategy == "uncertainty":
        chosen = np.argsort(scores)[::-1][:query_size]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return [candidate_indices[i] for i in chosen]