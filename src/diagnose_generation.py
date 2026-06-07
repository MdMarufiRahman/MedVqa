# src/diagnose_generation.py
"""
Sanity check for generation decode. No training. ~2 min.
Usage: python src/diagnose_generation.py
"""
import os, torch
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

from datasets import load_from_disk
from transformers import (
    Blip2ForConditionalGeneration, BlipImageProcessor,
    AutoTokenizer, Blip2Processor, BitsAndBytesConfig,
)

MODEL_ID     = "Salesforce/blip2-opt-2.7b"
DATASET_PATH = "data/vqa_rad/hf_cache"
_NEWLINE_ID  = 50118   # \n in OPT tokenizer

image_processor = BlipImageProcessor.from_pretrained(MODEL_ID)
tokenizer       = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=bnb_config,
    device_map="auto", max_memory={0: "7500MiB", "cpu": "20GiB"},
)
model.eval()

def decode_clean(output_ids):
    ids   = output_ids.tolist()
    start = 1 if ids and ids[0] == tokenizer.eos_token_id else 0
    if _NEWLINE_ID in ids[start:]:
        end = ids.index(_NEWLINE_ID, start)
        ids = ids[start:end]
    else:
        ids = ids[start:]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()

ds      = load_from_disk(DATASET_PATH)
samples = ds["test"].select(range(5))

print("\n" + "="*60)
print("GENERATION SANITY CHECK")
print("Expected: short clean answers — 'no', 'yes', 'right', etc.")
print("="*60)

all_clean = True
for i in range(len(samples)):
    image    = samples[i]["image"].convert("RGB")
    question = f"Question: {samples[i]['question']} Answer:"
    ref      = samples[i]["answer"]

    inputs = processor(
        [image], [question],
        return_tensors="pt", padding=True, truncation=True, max_length=128
    ).to("cuda", torch.float16)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=20, num_beams=4,
            repetition_penalty=1.5, early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    pred   = decode_clean(out[0])
    words  = len(pred.split())
    status = "✓" if words <= 6 else "✗ TOO LONG"
    if words > 6:
        all_clean = False

    print(f"  [{i}] {status}")
    print(f"       Q   : {samples[i]['question'][:60]}")
    print(f"       Pred: {repr(pred)}")
    print(f"       Ref : {repr(ref)}")
    print(f"       Raw ids: {out[0].tolist()}")
    print()

print("="*60)
if all_clean:
    print("✓ All clean! Copy metrics.py to src/ and run train.py")
else:
    print("✗ Still issues — share output for further diagnosis")
print("="*60)