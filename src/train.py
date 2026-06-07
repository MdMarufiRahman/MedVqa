import os, json
os.environ["BITSANDBYTES_NOWELCOME"]    = "1"
os.environ["BITSANDBYTES_CUDA_VERSION"] = "118"

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from collections import Counter
from evaluate import load as load_metric
from tqdm import tqdm

rouge_metric = load_metric("rouge")

CFG = {
    "model_id":         "Salesforce/blip2-opt-2.7b",
    "dataset_path":     "data/vqa_rad/hf_cache",
    "output_dir":       "checkpoints/blip2-medvqa",
    "epochs":           5,
    "batch_size":       1,
    "grad_accum_steps": 8,
    "learning_rate":    3e-5,
    "warmup_ratio":     0.1,
    "lora_r":           8,
    "lora_alpha":       16,
    "lora_dropout":     0.05,
    "max_memory":       {0: "7500MiB", "cpu": "20GiB"},
}

wandb.init(project="medvqa-blip2", config=CFG)
os.makedirs(CFG["output_dir"], exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── LoRA ───────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.original = original_linear
        self.scale    = alpha / r
        d_in, d_out   = original_linear.in_features, original_linear.out_features
        self.lora_A   = nn.Parameter(torch.randn(r, d_in)  * 0.01)
        self.lora_B   = nn.Parameter(torch.zeros(d_out, r))
        self.dropout  = nn.Dropout(dropout)
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.original(x)
        A = self.lora_A.to(x.device)
        B = self.lora_B.to(x.device)
        lora = self.dropout(x.to(A.dtype)) @ A.T @ B.T
        return base + self.scale * lora.to(base.dtype)

def inject_lora(model, targets, r, alpha, dropout):
    count = 0
    for name, _ in list(model.named_modules()):
        for t in targets:
            if name.endswith(t):
                parts  = name.split(".")
                parent = model
                for p in parts[:-1]: parent = getattr(parent, p)
                orig   = getattr(parent, parts[-1])
                if isinstance(orig, nn.Linear):
                    setattr(parent, parts[-1], LoRALinear(orig, r, alpha, dropout))
                    count += 1
    print(f"LoRA injected into {count} layers")
    return model

# ── Processor / Tokenizer ──────────────────────────────────────────────
image_processor = BlipImageProcessor.from_pretrained(CFG["model_id"])
tokenizer       = AutoTokenizer.from_pretrained(CFG["model_id"], use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)

# Known OPT token ids (verified by diagnose_generation.py)
EOS_ID     = tokenizer.eos_token_id   # 2
NEWLINE_ID = 50118                     # \n

def decode_answer(output_ids):
    """
    BLIP-2 generate() output format (confirmed):
      [EOS, answer_tokens..., \n, Question: ...]
    We skip the leading EOS and stop at the first \n or second EOS.
    Hard cap at 8 words — VQA-RAD answers are never longer.
    """
    ids   = output_ids.tolist()
    start = 1 if ids and ids[0] == EOS_ID else 0
    end   = len(ids)
    for i in range(start, len(ids)):
        if ids[i] in (NEWLINE_ID, EOS_ID):
            end = i
            break
    pred  = tokenizer.decode(ids[start:end], skip_special_tokens=True).strip()
    words = pred.split()
    return " ".join(words[:8]) if len(words) > 8 else pred

# ── Model ──────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
)
model = Blip2ForConditionalGeneration.from_pretrained(
    CFG["model_id"], quantization_config=bnb_config,
    device_map="auto", max_memory=CFG["max_memory"],
)
for p in model.parameters():
    p.requires_grad = False

model = inject_lora(model, ["q_proj","v_proj"],
                    CFG["lora_r"], CFG["lora_alpha"], CFG["lora_dropout"])

# Set repetition penalty on the model's generation config directly
# so it cannot be overridden by generate() kwargs being ignored
model.generation_config.repetition_penalty = 2.0
model.generation_config.no_repeat_ngram_size = 3

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

# ── Data ───────────────────────────────────────────────────────────────
ds       = load_from_disk(CFG["dataset_path"])
train_hf = ds["train"]
test_hf  = ds["test"]

# Dataset class inline (no external import needed)
class VQARadDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, processor, tokenizer, max_length=128):
        self.data      = hf_dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item     = self.data[idx]
        image    = item["image"].convert("RGB")
        question = f"Question: {item['question']} Answer: {item['answer']}"
        inputs   = self.processor(
            image, question,
            return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_len,
        )
        input_ids      = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Labels: mask the question part, only supervise the answer tokens
        answer_enc = self.tokenizer(
            item["answer"], add_special_tokens=False
        ).input_ids
        labels = input_ids.clone()
        labels[:-len(answer_enc)] = -100
        labels[attention_mask == 0] = -100

        return {
            "pixel_values":   inputs["pixel_values"].squeeze(0),
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

train_dataset = VQARadDataset(train_hf, processor, tokenizer)
train_loader  = DataLoader(train_dataset, batch_size=CFG["batch_size"],
                           shuffle=True, num_workers=0)

# ── Class-weighted loss ────────────────────────────────────────────────
answer_counts = Counter(a.lower().strip().rstrip('.') for a in train_hf["answer"])
total_ans     = sum(answer_counts.values())
yes_freq = answer_counts.get("yes", 1) / total_ans
no_freq  = answer_counts.get("no",  1) / total_ans
yes_tok  = tokenizer("yes", add_special_tokens=False).input_ids[0]
no_tok   = tokenizer("no",  add_special_tokens=False).input_ids[0]
vocab_sz  = model.language_model.lm_head.out_features
cw        = torch.ones(vocab_sz, dtype=torch.float32)
cw[yes_tok] = (1.0 / yes_freq) / 2
cw[no_tok]  = (1.0 / no_freq)  / 2
cw          = cw.to("cuda")
print(f"yes token {yes_tok} weight={cw[yes_tok]:.2f} | no token {no_tok} weight={cw[no_tok]:.2f}")

# ── Optimizer ──────────────────────────────────────────────────────────
optimizer    = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=CFG["learning_rate"], weight_decay=0.01,
)
total_steps  = (len(train_loader) // CFG["grad_accum_steps"]) * CFG["epochs"]
warmup_steps = int(total_steps * CFG["warmup_ratio"])
scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# ── Eval function (self-contained, no metrics.py import) ───────────────
def evaluate(model):
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_hf), 4), desc="Evaluating"):
            batch   = test_hf[i:i+4]
            images  = [img.convert("RGB") for img in batch["image"]]
            prompts = [f"Question: {q} Answer:" for q in batch["question"]]
            inputs  = processor(
                images, prompts, return_tensors="pt",
                padding=True, truncation=True, max_length=128,
            ).to("cuda", torch.float16)

            out = model.generate(
                **inputs,
                max_new_tokens=20,
                num_beams=4,
                repetition_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
                eos_token_id=EOS_ID,
            )
            for j, oids in enumerate(out):
                pred = decode_answer(oids)
                preds.append(pred)
                refs.append(batch["answer"][j])

    print("\n── Eval sample check ────────────────────────────────")
    for k in range(min(5, len(preds))):
        print(f"  Pred [{k}]: {repr(preds[k])}")
        print(f"  Ref  [{k}]: {repr(refs[k])}")
    print("─────────────────────────────────────────────────────\n")

    scores = rouge_metric.compute(predictions=preds, references=refs)
    model.train()
    return scores

# ── Training loop ──────────────────────────────────────────────────────
best_rouge, results_log = 0.0, []

for epoch in range(1, CFG["epochs"] + 1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        pv  = batch["pixel_values"].to("cuda", torch.float16)
        ids = batch["input_ids"].to("cuda")
        am  = batch["attention_mask"].to("cuda")
        lbl = batch["labels"].to("cuda")

        out     = model(pixel_values=pv, input_ids=ids, attention_mask=am, labels=lbl)
        logits  = out.logits
        B, T, V = logits.shape
        loss    = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, V).float(),
            lbl[:, 1:].contiguous().view(-1),
            weight=cw, ignore_index=-100, reduction="mean",
        ) / CFG["grad_accum_steps"]

        loss.backward()
        total_loss += loss.item()

        if (step + 1) % CFG["grad_accum_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 0.3)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        if step % 50 == 0:
            print(f"Epoch {epoch} | Step {step}/{len(train_loader)} | Loss {loss.item()*CFG['grad_accum_steps']:.4f}")
            wandb.log({"train/loss": loss.item()*CFG["grad_accum_steps"], "epoch": epoch})

    avg_loss = total_loss / len(train_loader) * CFG["grad_accum_steps"]
    print(f"\nEpoch {epoch} complete | Avg loss: {avg_loss:.4f}")

    scores  = evaluate(model)
    rouge_l = scores["rougeL"]
    delta   = (rouge_l - 0.3071) / 0.3071 * 100
    print(f"Epoch {epoch} | ROUGE-L: {rouge_l:.4f} | Target: 0.3440 | Δ {delta:+.1f}%")
    wandb.log({"eval/rougeL": rouge_l, "eval/rouge1": scores["rouge1"], "epoch": epoch})

    results_log.append({"epoch": epoch, "loss": avg_loss, "rougeL": rouge_l})
    with open("results/training_log.json", "w") as f:
        json.dump(results_log, f, indent=2)

    if rouge_l > best_rouge:
        best_rouge = rouge_l
        torch.save(
            {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k},
            f"{CFG['output_dir']}/lora_weights.pt"
        )
        print(f"  ✓ New best: ROUGE-L {rouge_l:.4f}")

print(f"\nBest ROUGE-L: {best_rouge:.4f} | Baseline: 0.3071 | Δ {(best_rouge-0.3071)/0.3071*100:+.1f}%")
wandb.finish()