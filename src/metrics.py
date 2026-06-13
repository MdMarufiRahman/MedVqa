"""
metrics.py — Evaluation metrics for VQA-RAD.

Root cause of Method A vs B gap (resolved):
  baseline_eval.py calls processor(image, prompt) — image+text encoded jointly.
  The old evaluate() encoded them separately, breaking BLIP-2 cross-attention
  and causing prompt leakage into predictions.

Fix: evaluate() now accepts a HuggingFace dataset split directly and calls
processor(image, prompt) per sample, identical to baseline_eval.py.

Decode logic (confirmed for Salesforce/blip2-opt-2.7b):
  - generate() returns NEW tokens only (prompt not echoed)
  - First token is always EOS (id=2) — skip it
  - Answer ends at first newline (token 50118) or EOS
  - Truncate to 8 words max
"""

import torch
from typing import List
from rouge_score import rouge_scorer

_NEWLINE_ID = 50118
_EOS_ID     = 2


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def _decode_clean(tokenizer, output_ids: torch.Tensor) -> str:
    ids = output_ids.tolist()
    start = 0
    while start < len(ids) and ids[start] in (_NEWLINE_ID, _EOS_ID):
        start += 1
    end = len(ids)
    for i in range(start, len(ids)):
        if ids[i] in (_NEWLINE_ID, _EOS_ID):
            end = i
            break
    pred = tokenizer.decode(ids[start:end], skip_special_tokens=True).strip()
    pred = pred.replace("<pad>", "").replace("\n", " ").strip()
    if pred.startswith(":"):
        pred = pred[1:].strip()
    words = pred.split()
    return " ".join(words[:8]) if len(words) > 8 else pred


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_rougeL(predictions: List[str], references: List[str]) -> float:
    assert len(predictions) == len(references)
    scores = [
        _scorer.score(ref.lower().strip(), pred.lower().strip())["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    correct = sum(
        p.strip().lower() == r.strip().lower()
        for p, r in zip(predictions, references)
    )
    return correct / len(predictions) if predictions else 0.0


# ---------------------------------------------------------------------------
# Evaluation — mirrors baseline_eval.py exactly
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, processor, hf_dataset, device, max_new_tokens: int = 20) -> dict:
    """
    Evaluate model on a HuggingFace dataset split.

    Args:
        model:       loaded BLIP-2 model
        processor:   Blip2Processor
        hf_dataset:  HF dataset split with 'image', 'question', 'answer' columns
                     (e.g. ds["test"] from load_dataset("flaviagiammarino/vqa-rad"))
        device:      torch device string or object
        max_new_tokens: generation length cap

    Returns dict with rougeL, exact_match, predictions, references.

    NOTE: accepts a HF dataset, NOT a DataLoader. This matches baseline_eval.py
    which calls processor(image, prompt) per sample. Encoding image+text jointly
    is required for correct BLIP-2 cross-attention — encoding separately causes
    prompt leakage into predictions.
    """
    from PIL import Image

    model.eval()
    # Set repetition controls on generation_config — kwargs are ignored by
    # quantized bitsandbytes models
    model.generation_config.repetition_penalty   = 2.0
    model.generation_config.no_repeat_ngram_size = 3

    predictions = []
    references  = []

    for sample in hf_dataset:
        image    = sample["image"]
        question = sample["question"]
        answer   = str(sample["answer"]).lower().strip()

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        prompt = f"Question: {question} Short answer:"
        inputs = processor(image, prompt, return_tensors="pt").to(device, torch.float16)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
        )

        pred = _decode_clean(processor.tokenizer, out[0])
        predictions.append(pred)
        references.append(answer)

    return {
        "rougeL":      compute_rougeL(predictions, references),
        "exact_match": compute_exact_match(predictions, references),
        "predictions": predictions,
        "references":  references,
    }