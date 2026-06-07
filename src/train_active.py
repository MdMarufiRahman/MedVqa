# src/train_active.py
"""
Main entry point for the active learning pipeline.

Usage:
    cd ~/medvqa
    python src/train_active.py

    # Ablation (random baseline):
    python src/train_active.py --strategy random
"""

import os, sys, argparse
os.environ["BITSANDBYTES_NOWELCOME"]   = "1"
os.environ["BITSANDBYTES_CUDA_VERSION"] = "118"

import torch
import torch.nn as nn
import wandb
from datasets import load_from_disk
from transformers import (
    Blip2ForConditionalGeneration,
    BlipImageProcessor,
    AutoTokenizer,
    Blip2Processor,
    BitsAndBytesConfig,
)

# Make sure src/ is on the path when running from repo root
sys.path.insert(0, os.path.dirname(__file__))
from dataset import VQARadDataset
from active_sampler import active_learning_loop

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy",   default="uncertainty", choices=["uncertainty", "random"])
    p.add_argument("--rounds",     type=int, default=5)
    p.add_argument("--seed_size",  type=int, default=150,
                   help="Labelled samples to start with (round 0)")
    p.add_argument("--query_size", type=int, default=150,
                   help="New samples queried each round")
    p.add_argument("--mc_passes",  type=int, default=10)
    p.add_argument("--epochs",     type=int, default=1,
                   help="Fine-tuning epochs per round")
    return p.parse_args()


def build_cfg(args):
    return {
        # ── active learning ──────────────────────────────────
        "al_strategy":           args.strategy,
        "al_rounds":             args.rounds,
        "al_seed_size":          args.seed_size,
        "al_query_size":         args.query_size,
        "al_mc_passes":          args.mc_passes,
        "al_epochs_per_round":   args.epochs,
        # ── model / training ─────────────────────────────────
        "model_id":             "Salesforce/blip2-opt-2.7b",
        "dataset_path":         "data/vqa_rad/hf_cache",
        "output_dir":           "checkpoints/blip2-active",
        "batch_size":           1,
        "grad_accum_steps":     8,
        "learning_rate":        2e-4,
        "warmup_ratio":         0.05,
        "lora_r":               8,
        "lora_alpha":           16,
        "lora_dropout":         0.05,
        "max_memory":           {0: "7500MiB", "cpu": "20GiB"},
        "baseline_rougeL":      0.3071,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Manual LoRA (same as train.py — no PEFT dependency)
# ─────────────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.original = original_linear
        self.r        = r
        self.scale    = alpha / r
        d_in  = original_linear.in_features
        d_out = original_linear.out_features
        self.lora_A  = nn.Parameter(torch.randn(r, d_in)  * 0.01)
        self.lora_B  = nn.Parameter(torch.zeros(d_out, r))
        # ← Dropout here is what MC Dropout exploits at inference time
        self.dropout = nn.Dropout(dropout)
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        base  = self.original(x)
        A     = self.lora_A.to(x.device)
        B     = self.lora_B.to(x.device)
        lora  = self.dropout(x.to(A.dtype)) @ A.T @ B.T
        return base + self.scale * lora.to(base.dtype)


def inject_lora(model, target_names, r, alpha, dropout):
    injected = 0
    for name, module in list(model.named_modules()):
        for target in target_names:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parts  = name.split(".")
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg  = build_cfg(args)

    run = wandb.init(
        project="medvqa-blip2",
        name=f"active-{args.strategy}-r{args.rounds}-q{args.query_size}",
        config=cfg,
    )

    # ── Processor ────────────────────────────────────────────────────────────
    image_processor = BlipImageProcessor.from_pretrained(cfg["model_id"])
    tokenizer       = AutoTokenizer.from_pretrained(cfg["model_id"], use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)

    # ── Model (4-bit QLoRA) ───────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        cfg["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=cfg["max_memory"],
    )

    # Freeze base weights
    for p in model.parameters():
        p.requires_grad = False

    # Inject LoRA (dropout layers are the key for MC Dropout uncertainty)
    model = inject_lora(
        model,
        target_names=["q_proj", "v_proj"],
        r=cfg["lora_r"],
        alpha=cfg["lora_alpha"],
        dropout=cfg["lora_dropout"],
    )
    count_params(model)

    # ── Data ──────────────────────────────────────────────────────────────────
    ds = load_from_disk(cfg["dataset_path"])
    train_hf = ds["train"]
    test_hf  = ds["test"]

    print(f"\nDataset sizes — train: {len(train_hf)} | test: {len(test_hf)}")
    print(f"Active learning plan:")
    print(f"  Strategy   : {cfg['al_strategy']}")
    print(f"  Seed size  : {cfg['al_seed_size']}")
    print(f"  Query size : {cfg['al_query_size']} per round")
    print(f"  Rounds     : {cfg['al_rounds']}")
    total_queried = cfg['al_seed_size'] + cfg['al_query_size'] * cfg['al_rounds']
    print(f"  Total used : {min(total_queried, len(train_hf))} / {len(train_hf)} "
          f"({min(total_queried, len(train_hf))/len(train_hf)*100:.1f}%)")

    # ── Run active learning ───────────────────────────────────────────────────
    al_log = active_learning_loop(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        train_dataset_hf=train_hf,
        test_dataset_hf=test_hf,
        VQARadDataset=VQARadDataset,
        cfg=cfg,
        results_dir="results",
        checkpoint_dir=cfg["output_dir"],
        wandb_run=run,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
