import os, json
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["BITSANDBYTES_CUDA_VERSION"] = "118"

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from transformers import (
    Blip2ForConditionalGeneration,
    BlipImageProcessor,
    AutoTokenizer,
    Blip2Processor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from datasets import load_from_disk
from dataset import VQARadDataset
from metrics import evaluate_model

# ── Config ─────────────────────────────────────────────────────────────
CFG = {
    "model_id": "Salesforce/blip2-opt-2.7b",
    "dataset_path": "data/vqa_rad/hf_cache",
    "output_dir": "checkpoints/blip2-medvqa",
    "epochs": 1,
    "batch_size": 1,
    "grad_accum_steps": 8,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "max_memory": {0: "7500MiB", "cpu": "20GiB"},
    "eval_every_n_epochs": 1,
}

wandb.init(project="medvqa-blip2", config=CFG)
os.makedirs(CFG["output_dir"], exist_ok=True)

# ── Manual LoRA layer ───────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.original = original_linear
        self.r = r
        self.scale = alpha / r
        d_in  = original_linear.in_features
        d_out = original_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(r, d_in)  * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        self.dropout = nn.Dropout(dropout)
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.original(x)
        A = self.lora_A.to(x.device)
        B = self.lora_B.to(x.device)
        lora = self.dropout(x.to(A.dtype)) @ A.T @ B.T
        return base + self.scale * lora.to(base.dtype)
def inject_lora(model, target_names, r, alpha, dropout):
    injected = 0
    for name, module in list(model.named_modules()):
        for target in target_names:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                original = getattr(parent, parts[-1])
                setattr(parent, parts[-1], LoRALinear(original, r, alpha, dropout))
                injected += 1
    print(f"Injected LoRA into {injected} layers")
    return model

def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.3f}%)")

# ── Processor ──────────────────────────────────────────────────────────
image_processor = BlipImageProcessor.from_pretrained(CFG["model_id"])
tokenizer = AutoTokenizer.from_pretrained(CFG["model_id"], use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)

# ── Model ──────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = Blip2ForConditionalGeneration.from_pretrained(
    CFG["model_id"],
    quantization_config=bnb_config,
    device_map="auto",
    max_memory=CFG["max_memory"],
)

# Freeze everything
for p in model.parameters():
    p.requires_grad = False

# Inject LoRA manually — zero PEFT, zero hooks
model = inject_lora(
    model,
    target_names=["q_proj", "v_proj"],
    r=CFG["lora_r"],
    alpha=CFG["lora_alpha"],
    dropout=CFG["lora_dropout"],
)
count_params(model)

# ── Data ───────────────────────────────────────────────────────────────
ds = load_from_disk(CFG["dataset_path"])
train_dataset = VQARadDataset(ds["train"], processor, tokenizer)
test_dataset  = VQARadDataset(ds["test"],  processor, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"],
                          shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=CFG["batch_size"],
                          shuffle=False, num_workers=0)

# ── Optimizer & scheduler ──────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=CFG["learning_rate"], weight_decay=0.01,
)
total_steps  = (len(train_loader) // CFG["grad_accum_steps"]) * CFG["epochs"]
warmup_steps = int(total_steps * CFG["warmup_ratio"])
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# ── Training loop ──────────────────────────────────────────────────────
best_rouge = 0.0
results_log = []

for epoch in range(1, CFG["epochs"] + 1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        pixel_values   = batch["pixel_values"].to("cuda", torch.float16)
        input_ids      = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels         = batch["labels"].to("cuda")

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss / CFG["grad_accum_steps"]
        loss.backward()
        total_loss += loss.item()

        if (step + 1) % CFG["grad_accum_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % 50 == 0:
            actual_loss = loss.item() * CFG["grad_accum_steps"]
            print(f"Epoch {epoch} | Step {step}/{len(train_loader)} | Loss {actual_loss:.4f}")
            wandb.log({"train/loss": actual_loss, "epoch": epoch})

    avg_loss = total_loss / len(train_loader) * CFG["grad_accum_steps"]
    print(f"\nEpoch {epoch} complete | Avg loss: {avg_loss:.4f}")

    scores  = evaluate_model(model, processor, tokenizer, test_loader)
    rouge_l = scores["rougeL"]
    delta   = (rouge_l - 0.3071) / 0.3071 * 100
    print(f"Epoch {epoch} | ROUGE-L: {rouge_l:.4f} | Target: 0.3440 | Δ {delta:+.1f}%")
    wandb.log({"eval/rougeL": rouge_l, "eval/rouge1": scores["rouge1"],
               "eval/rouge2": scores["rouge2"], "epoch": epoch})

    results_log.append({"epoch": epoch, "loss": avg_loss, "rougeL": rouge_l})
    with open("results/training_log.json", "w") as f:
        json.dump(results_log, f, indent=2)

    if rouge_l > best_rouge:
        best_rouge = rouge_l
        os.makedirs(CFG["output_dir"], exist_ok=True)
        lora_state = {k: v for k, v in model.state_dict().items()
                      if "lora_A" in k or "lora_B" in k}
        torch.save(lora_state, f"{CFG['output_dir']}/lora_weights.pt")
        tokenizer.save_pretrained(CFG["output_dir"])
        print(f"  ✓ New best saved: ROUGE-L {rouge_l:.4f}")

print(f"\nTraining complete.")
print(f"Best ROUGE-L: {best_rouge:.4f}")
print(f"Baseline:     0.3071")
print(f"Improvement:  {(best_rouge - 0.3071) / 0.3071 * 100:+.1f}%")
wandb.finish()