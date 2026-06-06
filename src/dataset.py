# src/dataset.py
import torch
from torch.utils.data import Dataset

class VQARadDataset(Dataset):
    def __init__(self, hf_dataset, processor, tokenizer, max_length=32):
        self.data = hf_dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample["image"].convert("RGB")
        question = f"Question: {sample['question']} Answer:"
        answer = sample["answer"]

        encoding = self.processor(
            image, question,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        )

        labels = self.tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        # Mask padding tokens from loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "pixel_values": encoding["pixel_values"].squeeze(),
            "labels": labels.squeeze(),
        }