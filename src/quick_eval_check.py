"""
Runs evaluate_model on just 20 test samples.
No training. Takes ~1 minute.
Usage: python src/quick_eval_check.py
"""
import os, sys, torch
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
sys.path.insert(0, "src")

from datasets import load_from_disk
from transformers import (
    Blip2ForConditionalGeneration, BlipImageProcessor,
    AutoTokenizer, Blip2Processor, BitsAndBytesConfig,
)
from metrics import evaluate_model, _decode_clean

MODEL_ID     = "Salesforce/blip2-opt-2.7b"
DATASET_PATH = "data/vqa_rad/hf_cache"

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

ds       = load_from_disk(DATASET_PATH)
mini_test = ds["test"].select(range(20))

print("\nRunning evaluate_model on 20 samples...")
scores = evaluate_model(model, processor, tokenizer, mini_test, batch_size=4)
print(f"ROUGE-L on 20 samples: {scores['rougeL']:.4f}")
print("\nIf predictions above are short and clean (no yesyesyes / lobe lobe)")
print("→ you are good to run: python src/train.py")