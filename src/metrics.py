# src/evaluate.py
import torch
from evaluate import load as load_metric
from tqdm import tqdm

rouge = load_metric("rouge")

def evaluate_model(model, processor, tokenizer, dataloader, device="cuda"):
    model.eval()
    predictions, references = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device, torch.float16)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=30,
            )

            for i, output in enumerate(out):
                # Phase 1 fix — slice off input tokens before decoding
                pred = tokenizer.decode(
                    output[input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()
                label_ids = batch["labels"][i]
                label_ids = label_ids[label_ids != -100]
                ref = tokenizer.decode(label_ids, skip_special_tokens=True).strip()
                predictions.append(pred)
                references.append(ref)
    print(f"Sample pred: {repr(predictions[0])}")
    print(f"Sample ref:  {repr(references[0])}")
    scores = rouge.compute(predictions=predictions, references=references)
    model.train()
    return scores