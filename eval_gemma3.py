"""
Evaluate UniTime + Gemma3 LoRA on GTEA. One instance at a time, clear GPU between each.
"""
import os
os.environ.setdefault("CUDA_HOME", "/usr/local/apps/cuda/12.1")

import gc
import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoConfig

from models.gemma3_vl import Gemma3VLMRForConditionalGeneration, Gemma3VLMRProcessor
from collators.gemma3_vl import Gemma3DataCollator
from datasets_mr import VideoCentricDataset

PAD_IDX = -100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval_data_path", required=True)
    ap.add_argument("--feat_folder", required=True)
    ap.add_argument("--video_folder", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")

    base = Gemma3VLMRForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    model = PeftModel.from_pretrained(base, args.adapter).eval()

    processor = Gemma3VLMRProcessor.from_pretrained(args.base_model)
    tokenizer = processor.tokenizer
    config = AutoConfig.from_pretrained(args.base_model)

    eval_ds = VideoCentricDataset(
        data_path=args.eval_data_path,
        video_folder=args.video_folder,
        feat_folder=args.feat_folder,
        fps=2, split="val", num_clips=32, clip_length=-1,
        model_family_id="gemma3",
    )

    collator = Gemma3DataCollator(
        config=config, tokenizer=tokenizer, processor=processor,
        mask_question_tokens=True,
    )

    losses, predictions = [], []

    with torch.no_grad():
        for i in range(len(eval_ds)):
            torch.cuda.empty_cache()
            gc.collect()

            instance = eval_ds[i]
            try:
                batch = collator([instance])
            except Exception as ex:
                predictions.append({"qid": instance["qid"], "error": f"collator: {ex}"})
                continue

            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            try:
                out = model(**batch_gpu)
            except torch.cuda.OutOfMemoryError:
                predictions.append({"qid": instance["qid"], "error": "OOM"})
                torch.cuda.empty_cache()
                gc.collect()
                continue
            except Exception as ex:
                predictions.append({"qid": instance["qid"], "error": str(ex)})
                continue

            loss_val = out.loss.item() if out.loss is not None else None
            if loss_val is not None:
                losses.append(loss_val)

            labels = batch_gpu["labels"][0]
            logits = out.logits[0]
            answer_mask = labels != PAD_IDX
            pred_ids = logits[:-1].argmax(-1)[answer_mask[1:]]
            gt_ids = labels[1:][answer_mask[1:]]
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True)

            query_text = "?"
            for msg in instance["message"]:
                if msg.get("role") == "user":
                    for item in msg.get("content", []):
                        if isinstance(item, dict) and item.get("type") == "text":
                            txt = item.get("text", "")
                            if txt.startswith("Query:"):
                                query_text = txt.split("Query:", 1)[1].split("\nAnswer", 1)[0].strip()
                                break
                if query_text != "?":
                    break

            predictions.append({
                "qid": instance["qid"], "query": query_text,
                "loss": loss_val, "gold_text": gt_text, "pred_text": pred_text,
                "exact_match": pred_text.strip() == gt_text.strip(),
            })

            del batch, batch_gpu, out, logits
            torch.cuda.empty_cache()

            if (i + 1) % 10 == 0 or i == len(eval_ds) - 1:
                avg = sum(losses) / max(len(losses), 1)
                print(f"  [{i+1}/{len(eval_ds)}] avg loss = {avg:.4f}")

    n_ok = sum(1 for p in predictions if "error" not in p)
    n_oom = sum(1 for p in predictions if p.get("error") == "OOM")
    avg_loss = sum(losses) / max(len(losses), 1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": {"total": len(eval_ds), "n_ok": n_ok, "n_oom": n_oom,
                                "avg_loss": avg_loss},
                    "predictions": predictions}, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  {n_ok}/{len(eval_ds)} ok, {n_oom} OOM, avg_loss={avg_loss:.4f}")
    print(f"{'='*60}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
