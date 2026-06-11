"""
dataset.py — VQA-RAD dataset for BLIP-2 fine-tuning.

Label masking strategy:
  - We want the model to learn to predict the ANSWER tokens only.
  - Everything before the answer (the prompt/question part) is masked with -100.
  - Padding tokens are also masked with -100.
  - CRITICAL: if the answer cannot be located in input_ids, we fall back to
    masking only padding — this prevents all-(-100) labels → nan loss.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


def _build_prompt(question: str) -> str:
    return f"Question: {question} Short answer:"


class VQARadDataset(Dataset):
    """
    Wraps the VQA-RAD HuggingFace dataset for BLIP-2 (processor + causal-LM head).

    Each item returns:
        pixel_values   : (3, H, W) float tensor
        input_ids      : (seq_len,) long tensor  — prompt + answer
        attention_mask : (seq_len,) long tensor
        labels         : (seq_len,) long tensor  — -100 everywhere except answer tokens
    """

    def __init__(
        self,
        split: str = "train",
        processor=None,
        max_length: int = 128,
        cache_dir: str = "data/vqa_rad/hf_cache",
    ):
        assert processor is not None, "Pass a BLIP-2 processor."
        self.processor = processor
        self.processor.tokenizer = self.processor.tokenizer.__class__.from_pretrained( "Salesforce/blip2-opt-2.7b", use_fast=False)
        self.max_length = max_length

        ds = load_dataset("flaviagiammarino/vqa-rad", cache_dir=cache_dir)
        # VQA-RAD has train/test splits only
        if split == "val":
            split = "test"
        self.data = ds[split]
        logger.info(f"Loaded VQA-RAD {split}: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        question = sample["question"]
        answer   = str(sample["answer"]).strip().lower()
        image    = sample["image"]  # PIL.Image

        # --- image ---
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        # --- text: encode prompt + answer together ---
        prompt     = _build_prompt(question)
        full_text  = prompt + " " + answer

        enc = self.processor.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)       # (seq_len,)
        attention_mask = enc["attention_mask"].squeeze(0)  # (seq_len,)

        # --- labels: mask everything except answer tokens ---
        labels = input_ids.clone()

        # Encode the prompt alone to find where the answer starts
        prompt_enc = self.processor.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        # Find the first non-padding token position from the right that belongs
        # to the answer.  We mask [0 : prompt_len] and all padding.
        # +1 because processor may prepend a BOS token that shifts everything.
        # We try prompt_len first; if that leaves zero valid positions, fall back.
        answer_start = min(prompt_len, input_ids.shape[0] - 1)

        # Mask prompt tokens
        labels[:answer_start] = -100
        # Mask padding tokens
        labels[attention_mask == 0] = -100

        # Sanity guard — if masking left nothing to learn from, fall back to
        # masking only padding (we'll still learn *something* rather than nan).
        valid = (labels != -100).sum().item()
        if valid == 0:
            logger.warning(
                f"Sample {idx} (q='{question[:40]}') produced zero valid label "
                f"positions after answer-only masking. Falling back to full-sequence labels."
            )
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

        # --- pixel values ---
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.squeeze(0)  # (3, H, W)

        return {
            "pixel_values":   pixel_values,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import Blip2Processor
    logging.basicConfig(level=logging.INFO)

    from transformers import BlipImageProcessor, AutoTokenizer
    image_processor = BlipImageProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)
    processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)
    ds = VQARadDataset(split="train", processor=processor)

    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        print(f"  {k}: {v.shape} {v.dtype}")

    valid = (sample["labels"] != -100).sum().item()
    total = sample["labels"].shape[0]
    print(f"  label valid positions: {valid}/{total}")
    assert valid > 0, "BUG: zero valid label positions on sample 0!"
    print("dataset.py smoke test PASSED.")