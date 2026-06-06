# src/qlora_config.py
import torch
from transformers import Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ── 4-bit quantization config ──────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,    # nested quantization, saves ~0.4GB
    bnb_4bit_quant_type="nf4",         # NormalFloat4, best for LLM weights
    bnb_4bit_compute_dtype=torch.float16,
)

# ── Load model in 4-bit ────────────────────────────────────────────────
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "7500MiB", "cpu": "20GiB"},
)

# ── Prepare for k-bit training (adds gradient hooks) ──────────────────
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# ── LoRA config ────────────────────────────────────────────────────────
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

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expect: trainable params ~3-8M out of ~2.7B  (~0.1-0.3%)
# At the end of qlora_config.py — save configs for reproducibility
import json

model.print_trainable_parameters()

config_log = {
    "base_model": "Salesforce/blip2-opt-2.7b",
    "quantization": "4-bit NF4 double quant",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "query", "value"],
    "task_type": "CAUSAL_LM",
}

with open("results/qlora_config.json", "w") as f:
    json.dump(config_log, f, indent=2)

print("QLoRA config saved.")