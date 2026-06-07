"""
metrics.py — Evaluation metrics for VQA-RAD.

Decode logic is CONFIRMED correct for Salesforce/blip2-opt-2.7b:
  - model.generate() returns NEW tokens only (prompt is NOT echoed)
  - First token is always EOS (id=2) — skip it
  - Answer ends at first \\n (token id=50118) or EOS
  - Truncate prediction to 8 words max (answers are short)
"""

import re
from typing import List

import torch
from rouge_score import rouge_scorer

# OPT tokenizer constants (confirmed)
_NEWLINE_ID = 50118
_EOS_ID     = 2


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def _decode_clean(tokenizer, output_ids: torch.Tensor) -> str:
    """
    Decode a single output tensor from model.generate().

    output_ids shape: (seq_len,)
    """
    ids   = output_ids.tolist()
    start = 1 if ids and ids[0] == _EOS_ID else 0
    end   = len(ids)
    for i in range(start, len(ids)):
        if ids[i] in (_NEWLINE_ID, _EOS_ID):
            end = i
            break
    pred = tokenizer.decode(ids[start:end], skip_special_tokens=True).strip()
    pred = pred.replace("<pad>", "").replace("\n", " ").strip()
    words = pred.split()
    return " ".join(words[:8]) if len(words) > 8 else pred


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_rougeL(predictions: List[str], references: List[str]) -> float:
    """Return mean ROUGE-L F1 over a list of (pred, ref) pairs."""
    assert len(predictions) == len(references), "Length mismatch"
    scores = [
        _scorer.score(ref.lower().strip(), pred.lower().strip())["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Exact match (binary yes/no)
# ---------------------------------------------------------------------------

def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    correct = sum(
        p.strip().lower() == r.strip().lower()
        for p, r in zip(predictions, references)
    )
    return correct / len(predictions) if predictions else 0.0


# ---------------------------------------------------------------------------
# Batch evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, processor, dataloader, device, max_new_tokens: int = 20) -> dict:
    """
    Run generation over the entire dataloader and return metrics.

    Returns:
        {
          "rougeL": float,
          "exact_match": float,
          "predictions": List[str],
          "references": List[str],
        }
    """
    model.eval()
    tokenizer   = processor.tokenizer
    predictions = []
    references  = []

    for batch in dataloader:
        pixel_values   = batch["pixel_values"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Store references (decode non-masked labels)
        for label_row, input_row in zip(batch["labels"], batch["input_ids"]):
            valid_ids = input_row[label_row != -100]
            ref_text  = tokenizer.decode(valid_ids.tolist(), skip_special_tokens=True).strip()
            references.append(ref_text.lower())

        outputs = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
        )

        for out_ids in outputs:
            predictions.append(_decode_clean(tokenizer, out_ids))

    rouge  = compute_rougeL(predictions, references)
    exact  = compute_exact_match(predictions, references)
    return {
        "rougeL":      rouge,
        "exact_match": exact,
        "predictions": predictions,
        "references":  references,
    }