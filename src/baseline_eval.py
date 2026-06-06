# src/baseline_eval.py
import os
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["BNB_CUDA_VERSION"] = "118"
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from datasets import load_from_disk
from evaluate import load as load_metric
from tqdm import tqdm

# Load model in 8-bit to fit in 8GB VRAM
from transformers import Blip2Processor, AutoTokenizer, BlipImageProcessor

image_processor = BlipImageProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)

processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "7500MiB", "cpu": "20GiB"},
)
model.eval()

ds = load_from_disk("data/vqa_rad/hf_cache")
test_set = ds["test"]
rouge = load_metric("rouge")

predictions, references = [], []

with torch.no_grad():
    for sample in tqdm(test_set, desc="Zero-shot eval"):
        image = sample["image"].convert("RGB")
        question = sample["question"]
        question_prompt = f"Question: {question} Answer:"
        inputs = processor(image, question_prompt, return_tensors="pt").to("cuda", torch.float16)
        #inputs = processor(image, question, return_tensors="pt").to("cuda", torch.float16)
        out = model.generate(**inputs, max_new_tokens=30)
        pred = processor.decode(out[0], skip_special_tokens=True).strip()
        predictions.append(pred)
        references.append(sample["answer"])

scores = rouge.compute(predictions=predictions, references=references)
print(f"\n=== ZERO-SHOT BASELINE ===")
print(f"ROUGE-L: {scores['rougeL']:.4f}")   # LOG THIS NUMBER
print(f"ROUGE-1: {scores['rouge1']:.4f}")
print(f"ROUGE-2: {scores['rouge2']:.4f}")

#Fact check:
# Run this on just 3 samples to see what's actually being generated
for i in range(3):
    sample = test_set[i]
    image = sample["image"].convert("RGB")
    question = sample["question"]
    question_prompt = f"Question: {question} Answer:"
    inputs = processor(image, question_prompt, return_tensors="pt").to("cuda", torch.float16)
    #inputs = processor(image, question, return_tensors="pt").to("cuda", torch.float16)
    out = model.generate(**inputs, max_new_tokens=30)
    pred = processor.decode(out[0], skip_special_tokens=True).strip()
    print(f"Q: {question}")
    print(f"Pred: repr={repr(pred)}")
    print(f"GT:   {sample['answer']}\n")
out = model.generate(**inputs, max_new_tokens=30)
print("Raw tokens:", out[0].tolist())
pred = processor.tokenizer.decode(out[0], skip_special_tokens=False).strip()
print("Raw decoded:", repr(pred))
# Make sure baseline_scores.json reflects the corrected numbers
import json, os
os.makedirs("results", exist_ok=True)
with open("results/baseline_scores.json", "w") as f:
    json.dump({
        "zero_shot": {"rougeL": 0.3071, "rouge1": 0.3072, "rouge2": 0.0056},
        "model": "blip2-opt-2.7b",
        "dataset": "vqa-rad",
        "decode_fix": "slice input_ids length before decoding"
    }, f, indent=2)