"""
verify_eval_parity.py

Checks whether metrics.py evaluate() and baseline_eval.py produce consistent
results when run on the same zero-shot model. Run this BEFORE any training.

Usage:
    python src/verify_eval_parity.py

Expected result: both ROUGE-L scores within ±0.01 of each other, and
both within ±0.02 of the saved baseline of 0.3071.

If they diverge, the section labelled DIAGNOSIS tells you exactly what's wrong.
"""

import os, sys, json, torch
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
sys.path.insert(0, "src")

from torch.utils.data import DataLoader
from transformers import (
    Blip2ForConditionalGeneration,
    BlipImageProcessor,
    AutoTokenizer,
    Blip2Processor,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from rouge_score import rouge_scorer as rouge_scorer_lib

# ---------------------------------------------------------------------------
# Config — match your project paths
# ---------------------------------------------------------------------------
MODEL_ID     = "Salesforce/blip2-opt-2.7b"
CACHE_DIR    = "data/vqa_rad/hf_cache"
SAVED_BASELINE = 0.3071
# How many test samples to use — 50 is enough for a stable check, fast to run
N_SAMPLES    = 50
# ---------------------------------------------------------------------------

def load_model_and_processor():
    image_processor = BlipImageProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    return model, processor, tokenizer


def rouge_l(predictions, references):
    scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(r.lower().strip(), p.lower().strip())["rougeL"].fmeasure
        for p, r in zip(predictions, references)
    ]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Method A: baseline_eval.py style
#   - prompt: "Question: X Answer:"
#   - decode: processor.decode(out[0], skip_special_tokens=True).strip()
#   - no truncation, no custom token logic
# ---------------------------------------------------------------------------
@torch.no_grad()
def method_a_baseline_style(model, processor, samples):
    """Replicates baseline_eval.py as closely as possible."""
    preds, refs = [], []
    for sample in samples:
        image = sample["image"].convert("RGB")
        question = sample["question"]
        prompt = f"Question: {question} Answer:"
        inputs = processor(image, prompt, return_tensors="pt").to("cuda", torch.float16)
        out = model.generate(**inputs, max_new_tokens=30)
        pred = processor.decode(out[0], skip_special_tokens=True).strip()
        preds.append(pred)
        refs.append(str(sample["answer"]).lower().strip())
    return preds, refs


# ---------------------------------------------------------------------------
# Method B: metrics.py evaluate() style
#   - uses dataset.py prompt: "Question: X Short answer:"
#   - uses _decode_clean() with EOS/newline stopping
#   - re-encodes prompt separately for generation
# ---------------------------------------------------------------------------
from metrics import evaluate as metrics_evaluate
from dataset import VQARadDataset

@torch.no_grad()
def method_b_metrics_style(model, processor, n_samples):
    """Runs the actual metrics.py evaluate() on the test split."""
    val_ds = VQARadDataset(
        "test", processor=processor, max_length=128, cache_dir=CACHE_DIR
    )
    # Subset to same N_SAMPLES
    from torch.utils.data import Subset
    # val_ds = Subset(val_ds, list(range(min(n_samples, len(val_ds)))))
    # val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    # NEW — replace with:
    from datasets import load_dataset
    hf_test_b = load_dataset("flaviagiammarino/vqa-rad", 
                          cache_dir=CACHE_DIR)["test"].select(range(N_SAMPLES))
    device = next(model.language_model.parameters()).device
    # results = metrics_evaluate(model, processor, val_loader, device, max_new_tokens=20)
    results_b = metrics_evaluate(model, processor, hf_test_b, "cuda")
    # return results["predictions"], results["references"], results["rougeL"]
    return results_b["predictions"], results_b["references"], results_b["rougeL"]


# ---------------------------------------------------------------------------
# Diagnosis helper
# ---------------------------------------------------------------------------
def diagnose(preds_a, refs_a, preds_b, refs_b, score_a, score_b):
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    # Check if references match
    matching_refs = sum(r_a == r_b for r_a, r_b in zip(refs_a, refs_b))
    print(f"\nReference strings matching: {matching_refs}/{len(refs_a)}")
    if matching_refs < len(refs_a) * 0.95:
        print("  ⚠ REFERENCES DIFFER — the two methods are decoding ground-truth")
        print("    answers differently. This alone will cause score divergence.")
        for i, (ra, rb) in enumerate(zip(refs_a[:5], refs_b[:5])):
            if ra != rb:
                print(f"    Sample {i}: baseline_ref='{ra}'  metrics_ref='{rb}'")

    # Check prediction patterns
    empty_a = sum(1 for p in preds_a if p.strip() == "")
    empty_b = sum(1 for p in preds_b if p.strip() == "")
    print(f"\nEmpty predictions — method A: {empty_a}, method B: {empty_b}")
    if empty_b > 0:
        print("  ⚠ metrics.py is producing empty predictions → decode bug")

    # Show side-by-side for first 5 samples
    print("\nFirst 5 sample comparison:")
    print(f"{'Ref':<20} {'Method A (baseline)':<30} {'Method B (metrics)':<30}")
    print("-" * 80)
    for i in range(min(5, len(preds_a))):
        print(f"{refs_a[i]:<20} {preds_a[i]:<30} {preds_b[i]:<30}")

    # Score gap
    gap = abs(score_a - score_b)
    print(f"\nScore gap: {gap:.4f}")
    if gap < 0.01:
        print("  ✓ Scores are consistent — eval pipeline is aligned")
    elif gap < 0.03:
        print("  ~ Small gap — likely prompt wording difference ('Answer:' vs 'Short answer:')")
        print("    Fix: change dataset.py _build_prompt to use 'Answer:' to match baseline")
    else:
        print("  ✗ Large gap — structural mismatch in decode or reference extraction")
        print("    Most likely causes:")
        print("    1. Empty predictions from metrics.py (check empty count above)")
        print("    2. Reference decoded from labels instead of raw answer string")
        print("    3. Prompt mismatch causing wrong prompt_len → wrong label mask")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading model ({MODEL_ID})...")
    model, processor, tokenizer = load_model_and_processor()

    print(f"Loading {N_SAMPLES} test samples from {CACHE_DIR}...")
    ds = load_dataset("flaviagiammarino/vqa-rad", cache_dir=CACHE_DIR)
    samples_a = [ds["test"][i] for i in range(N_SAMPLES)]

    print("\n--- Method A: baseline_eval.py style ---")
    preds_a, refs_a = method_a_baseline_style(model, processor, samples_a)
    score_a = rouge_l(preds_a, refs_a)
    print(f"ROUGE-L (method A): {score_a:.4f}  [saved baseline: {SAVED_BASELINE}]")

    print("\n--- Method B: metrics.py evaluate() style ---")
    # preds_b, refs_b, score_b = method_b_metrics_style(model, processor, N_SAMPLES)
    preds_b, refs_b, score_b = results_b["predictions"], results_b["references"], results_b["rougeL"]
    print(f"ROUGE-L (method B): {score_b:.4f}")

    print(f"\n{'='*60}")
    print(f"Method A (baseline style):  {score_a:.4f}")
    print(f"Method B (metrics.py):      {score_b:.4f}")
    print(f"Saved baseline:             {SAVED_BASELINE}")
    print(f"Gap A↔B:                    {abs(score_a - score_b):.4f}")
    print(f"Gap A↔saved:                {abs(score_a - SAVED_BASELINE):.4f}")

    # Pass/fail
    if abs(score_a - score_b) < 0.01:
        print("\n✓ PASS — eval methods are consistent. Safe to compare training results.")
    else:
        print("\n✗ FAIL — eval methods are inconsistent. Fix before reporting any numbers.")
        diagnose(preds_a, refs_a, preds_b, refs_b, score_a, score_b)

    # Save results for reference
    os.makedirs("results", exist_ok=True)
    with open("results/eval_parity_check.json", "w") as f:
        json.dump({
            "n_samples": N_SAMPLES,
            "method_a_rougeL": score_a,
            "method_b_rougeL": score_b,
            "saved_baseline": SAVED_BASELINE,
            "gap": abs(score_a - score_b),
            "method_a_sample_preds": list(zip(refs_a[:10], preds_a[:10])),
            "method_b_sample_preds": list(zip(refs_b[:10], preds_b[:10])),
        }, f, indent=2)
    print("\nFull results saved to results/eval_parity_check.json")


if __name__ == "__main__":
    main()