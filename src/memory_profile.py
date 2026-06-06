# src/memory_profile.py
import torch
from transformers import Blip2Processor, BlipImageProcessor, AutoTokenizer
from datasets import load_from_disk
from transformers import Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# paste your bnb_config + lora_config from 2.4 above
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,    # nested quantization, saves ~0.4GB
    bnb_4bit_quant_type="nf4",         # NormalFloat4, best for LLM weights
    bnb_4bit_compute_dtype=torch.float16,
)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "7500MiB", "cpu": "20GiB"},
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
lora_config = LoraConfig(
    r=16,                          # rank — start here, ablate later
    lora_alpha=32,                 # scaling = alpha/r = 2.0
    target_modules=[               # Q-Former + LLM attention projections
        "q_proj", "v_proj",        # OPT decoder attention
        "query", "value",          # Q-Former cross-attention
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
# Load processor (your working fix from Phase 1)
image_processor = BlipImageProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)
processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)

# Load dataset
ds = load_from_disk("data/vqa_rad/hf_cache")
sample = ds["train"][0]

# Simulate one forward + backward pass
image = sample["image"].convert("RGB")
question = f"Question: {sample['question']} Answer:"
answer = sample["answer"]

inputs = processor(image, question, return_tensors="pt").to("cuda", torch.float16)
labels = tokenizer(answer, return_tensors="pt").input_ids.to("cuda")

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

outputs = model(**inputs, labels=labels)
loss = outputs.loss
print(f"Loss: {loss.item():.4f}")

loss.backward()

peak = torch.cuda.max_memory_allocated() / 1024**3
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"Peak VRAM: {peak:.2f} GB / {total:.2f} GB")
print(f"Headroom:  {total - peak:.2f} GB")