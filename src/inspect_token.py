"""
Run this on your machine to see exactly what the tokenizer produces.
Copy the output back here.
"""
import sys
sys.path.insert(0, "src")
import torch
from transformers import BlipImageProcessor, AutoTokenizer, Blip2Processor

MODEL_ID = "Salesforce/blip2-opt-2.7b"
image_processor = BlipImageProcessor.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)

# Simulate exactly what dataset.py does for one sample
question = "Is there pleural effusion?"
answer   = "yes"

prompt    = f"Question: {question} Short answer:"
full_text = prompt + " " + answer

# Step 1: encode full_text (as dataset.py does)
enc = tokenizer(full_text, max_length=32, padding="max_length",
                truncation=True, return_tensors="pt")
input_ids = enc["input_ids"].squeeze(0)
attn      = enc["attention_mask"].squeeze(0)

# Step 2: encode prompt alone (as dataset.py does)
prompt_enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
prompt_len = prompt_enc["input_ids"].shape[1]

print("=== TOKEN INSPECTION ===")
print(f"prompt_len (no BOS): {prompt_len}")
print(f"input_ids length:    {input_ids.shape[0]}")
print()
print("Full input_ids with decoded tokens:")
for i, tid in enumerate(input_ids.tolist()):
    tok = tokenizer.decode([tid])
    marker = ""
    if i < prompt_len:     marker = "  ← prompt (masked)"
    elif tid == 1:         marker = "  ← PAD"
    elif tok.strip():      marker = "  ← ANSWER"
    print(f"  [{i:3d}] id={tid:6d}  tok='{tok}'  {marker}")

print()
print(f"answer_start candidates:")
print(f"  prompt_len       = {prompt_len}  → labels[:{prompt_len}] masked")
print(f"  prompt_len + 1   = {prompt_len+1} → labels[:{prompt_len+1}] masked")
print()

for start in [prompt_len, prompt_len+1]:
    labels = input_ids.clone()
    labels[:start] = -100
    labels[attn == 0] = -100
    valid = (labels != -100).sum().item()
    valid_ids = input_ids[labels != -100]
    ref = tokenizer.decode(valid_ids.tolist(), skip_special_tokens=True).strip()
    print(f"  answer_start={start}: valid_positions={valid}, decoded_ref='{ref}'")

print()
print("=== WHAT METRICS.PY RECONSTRUCTS AS PROMPT ===")
for start in [prompt_len, prompt_len+1]:
    labels = input_ids.clone()
    labels[:start] = -100
    labels[attn == 0] = -100
    prompt_ids = input_ids[labels == -100]
    prompt_ids = prompt_ids[prompt_ids != 1]
    reconstructed = tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=True).strip()
    print(f"  answer_start={start}: reconstructed_prompt='{reconstructed}'")

print()
print("=== WHAT BASELINE_EVAL.PY USES ===")
print(f"  prompt='{prompt}'  →  feeds directly to processor(image, prompt)")