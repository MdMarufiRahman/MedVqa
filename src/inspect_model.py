# src/inspect_model.py
import torch
from transformers import Blip2ForConditionalGeneration

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "7500MiB", "cpu": "20GiB"},
)

# Print all named modules with their types
for name, module in model.named_modules():
    if "Linear" in type(module).__name__:
        print(f"{name:80s} {type(module).__name__}")