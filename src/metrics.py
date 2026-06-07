# src/metrics.py
import torch
from evaluate import load as load_metric
from tqdm import tqdm

rouge = load_metric("rouge")

# Confirmed token ids for OPT tokenizer:
_NEWLINE_ID = 50118   # \n
_EOS_ID     = 2       # </s>

# After fine-tuning, BLIP-2 output format is:
#   [</s>, answer_tokens..., \n, Question: ...]
# Base model may generate conversational sentences instead.
# _decode_clean handles both cases.

def _decode_clean(tokenizer, output_ids):
    ids   = output_ids.tolist()

    # Skip leading EOS
    start = 1 if ids and ids[0] == _EOS_ID else 0

    # Stop at first \n (post-fine-tune format) or second EOS (base model stop)
    stop_tokens = {_NEWLINE_ID, _EOS_ID}
    end = len(ids)
    for idx in range(start, len(ids)):
        if ids[idx] in stop_tokens:
            end = idx
            break

    ids  = ids[start:end]
    pred = tokenizer.decode(ids, skip_special_tokens=True).strip()

    # Hard cap: VQA-RAD answers are never more than 8 words
    words = pred.split()
    if len(words) > 8:
        pred = " ".join(words[:8])

    return pred


def evaluate_model(model, processor, tokenizer, dataset_hf, device="cuda", batch_size=4):
    model.eval()
    predictions, references = [], []

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset_hf), batch_size), desc="Evaluating"):
            batch_samples = dataset_hf[i : i + batch_size]
            images    = [img.convert("RGB") for img in batch_samples["image"]]
            questions = [f"Question: {q} Answer:" for q in batch_samples["question"]]
            answers   = batch_samples["answer"]

            inputs = processor(
                images, questions,
                return_tensors="pt", padding=True,
                truncation=True, max_length=128,
            ).to(device, torch.float16)

            out = model.generate(
                **inputs,
                max_new_tokens=20,
                num_beams=4,
                repetition_penalty=1.5,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
            )

            for j, output_ids in enumerate(out):
                pred = _decode_clean(tokenizer, output_ids)
                predictions.append(pred)
                references.append(answers[j] if isinstance(answers, list) else answers)

    print("\n── Eval sample check ────────────────────────────────")
    for k in range(min(5, len(predictions))):
        print(f"  Pred [{k}]: {repr(predictions[k])}")
        print(f"  Ref  [{k}]: {repr(references[k])}")
    print("─────────────────────────────────────────────────────\n")

    scores = rouge.compute(predictions=predictions, references=references)
    model.train()
    return scores