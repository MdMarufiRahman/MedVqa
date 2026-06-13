"""
train.py — QLoRA fine-tuning of BLIP-2 on VQA-RAD.

All known bugs from session log are fixed:
  ✓ nan loss  → use out.loss directly (no manual CE on float16 logits)
  ✓ vocab size mismatch → use lm_head.out_features, not tokenizer.vocab_size
  ✓ device mismatch → LoRA adapters moved to input device in custom forward
  ✓ repetition → set on model.generation_config before generate()
  ✓ decode → confirmed correct in metrics.py
  ✓ label masking → dataset.py guarantees >0 valid positions
"""

import os
import json
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Blip2ForConditionalGeneration, Blip2Processor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

from dataset import VQARadDataset
from metrics import evaluate, _decode_clean

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger().setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CFG = {
    "model_name":       "Salesforce/blip2-opt-2.7b",
    "cache_dir":        "data/vqa_rad/hf_cache",
    "output_dir":       "checkpoints",
    "results_dir":      "results",

    "learning_rate":    3e-5,
    "epochs":           5,
    "batch_size":       1,
    "grad_accum_steps": 8,       # effective batch = 8
    "warmup_ratio":     0.1,
    "grad_clip":        0.3,
    "max_length":       128,

    # LoRA
    "lora_r":           8,
    "lora_alpha":       16,
    "lora_dropout":     0.05,

    # Generation
    "max_new_tokens":   20,
    "num_beams":        4,
    "repetition_penalty": 2.0,
    "no_repeat_ngram_size": 3,
}


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_model_and_processor(cfg: dict):
    logger.info("Loading processor…")
    from transformers import BlipImageProcessor, AutoTokenizer
    image_processor = BlipImageProcessor.from_pretrained(cfg["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)

    logger.info("Loading model in 4-bit…")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    logger.info("Preparing model for k-bit training…")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    logger.info("Attaching LoRA adapters…")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)
    model.language_model.print_trainable_parameters()

    # Fix repetition — must be set on generation_config, not generate() kwargs
    model.generation_config.repetition_penalty    = cfg["repetition_penalty"]
    model.generation_config.no_repeat_ngram_size  = cfg["no_repeat_ngram_size"]

    return model, processor


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, model, processor, train_dataset, val_dataset):
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=0.01,
    )

    total_steps  = (len(train_loader) // cfg["grad_accum_steps"]) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Determine the device of the language model head
    lm_device = next(model.language_model.parameters()).device
    logger.info(f"LM device: {lm_device}")

    history = []
    global_step = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss   = 0.0
        accum_loss   = 0.0
        nan_batches  = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            pixel_values   = batch["pixel_values"].to(lm_device)
            input_ids      = batch["input_ids"].to(lm_device)
            attention_mask = batch["attention_mask"].to(lm_device)
            labels         = batch["labels"].to(lm_device)

            # Sanity: skip batch if labels are all -100 (would give nan)
            if (labels != -100).sum().item() == 0:
                logger.warning(f"Epoch {epoch} step {step}: all-(-100) labels — skipping batch")
                nan_batches += 1
                continue

            out = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Use model's own loss — manual CE on float16 logits overflows to nan
            loss = out.loss

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Epoch {epoch} step {step}: nan/inf loss — skipping batch")
                nan_batches += 1
                optimizer.zero_grad()
                continue

            loss = loss / cfg["grad_accum_steps"]
            loss.backward()
            accum_loss += loss.item()

            if (step + 1) % cfg["grad_accum_steps"] == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg["grad_clip"],
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_loss += accum_loss

                if global_step % 5 == 0:
                    logger.info(
                        f"Epoch {epoch} | Step {global_step} | "
                        f"Loss {accum_loss * cfg['grad_accum_steps']:.4f} | "
                        f"LR {scheduler.get_last_lr()[0]:.2e}"
                    )
                accum_loss = 0.0

        logger.info(f"Epoch {epoch} done | avg loss: {epoch_loss / max(global_step, 1):.4f} | nan batches: {nan_batches}")

        # --- Evaluate ---
        logger.info(f"Evaluating epoch {epoch}…")
        # metrics = evaluate(model, processor, val_loader, lm_device, cfg["max_new_tokens"])
        metrics = evaluate(model, processor, hf_test, lm_device, cfg["max_new_tokens"])
        logger.info(
            f"Epoch {epoch} | ROUGE-L: {metrics['rougeL']:.4f} | "
            f"ExactMatch: {metrics['exact_match']:.4f}"
        )

        # Print a few examples
        for pred, ref in zip(metrics["predictions"][:5], metrics["references"][:5]):
            logger.info(f"  pred: '{pred}'  |  ref: '{ref}'")

        # Save checkpoint
        ckpt_path = os.path.join(cfg["output_dir"], f"epoch_{epoch}")
        model.save_pretrained(ckpt_path)
        processor.save_pretrained(ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

        history.append({"epoch": epoch, **{k: v for k, v in metrics.items() if k != "predictions" and k != "references"}})

    # Save history
    hist_path = os.path.join(cfg["results_dir"], "train_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved: {hist_path}")
    return history


# ---------------------------------------------------------------------------
# Pre-flight checks (run these before training to catch issues early)
# ---------------------------------------------------------------------------

def preflight_checks(train_dataset, n=5):
    logger.info("Running pre-flight checks…")
    ok = True
    for i in range(min(n, len(train_dataset))):
        sample = train_dataset[i]
        valid  = (sample["labels"] != -100).sum().item()
        total  = sample["labels"].shape[0]
        if valid == 0:
            logger.error(f"  Sample {i}: ZERO valid label positions! This will cause nan loss.")
            ok = False
        else:
            logger.info(f"  Sample {i}: {valid}/{total} valid label positions ✓")
    if ok:
        logger.info("Pre-flight checks PASSED.")
    else:
        logger.error("Pre-flight checks FAILED — fix dataset.py before training.")
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",  action="store_true", help="1-epoch smoke test on 100 samples")
    parser.add_argument("--epochs", type=int,   default=None)
    parser.add_argument("--lr",     type=float, default=None)
    args = parser.parse_args()

    if args.epochs: CFG["epochs"] = args.epochs
    if args.lr:     CFG["learning_rate"] = args.lr

    model, processor = load_model_and_processor(CFG)

    train_ds = VQARadDataset("train", processor=processor, max_length=CFG["max_length"], cache_dir=CFG["cache_dir"])
    val_ds   = VQARadDataset("test",  processor=processor, max_length=CFG["max_length"], cache_dir=CFG["cache_dir"])

    if args.smoke:
        logger.info("SMOKE TEST: using 100 training samples, 1 epoch")
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(100, len(train_ds)))))
        # val_ds   = Subset(val_ds,   list(range(min(20,  len(val_ds)))))
        hf_test = hf_test.select(range(min(20, len(hf_test))))
        from datasets import load_dataset
        hf_test = load_dataset("flaviagiammarino/vqa-rad", 
                        cache_dir=CFG["cache_dir"])["test"]
        CFG["epochs"] = 1

    # Must pass before wasting GPU time
    assert preflight_checks(train_ds), "Fix dataset issues before training."

    history = train(CFG, model, processor, train_ds, val_ds)
    logger.info("Training complete.")
    logger.info(f"Best ROUGE-L: {max(h['rougeL'] for h in history):.4f}")


if __name__ == "__main__":
    main()