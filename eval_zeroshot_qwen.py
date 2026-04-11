"""
Zero-shot evaluation: pre-trained UniTime LoRA (Qwen2-VL-7B) on GTEA.

No GTEA-specific training. Tests whether the universal VTG pre-training
(1.2M+ queries across 7 datasets) transfers to TAS on GTEA.

This is the KEY few-shot baseline:
  - 0-shot: this script (no GTEA training)
  - N-shot: fine-tune from UniTime LoRA with N GTEA videos
  - Full: fine-tune with all 21 GTEA train videos (what we've been doing)

Usage:
    conda activate UniTime
    cd experiments/unitime/UniTime
    python eval_zeroshot_qwen.py \
        --base_model /path/to/Qwen2-VL-7B-Instruct \
        --adapter /path/to/zeqianli-UniTime-LoRA \
        --eval_data_path .../data/gtea/annot/test.json \
        --feat_folder /path/to/Qwen2-VL-7B-features/gtea \
        --output ./results/zeroshot_unitime/eval.json
"""
import os
os.environ.setdefault("CUDA_HOME", "/usr/local/apps/cuda/12.1")

import faulthandler
faulthandler.enable()

import argparse
import json
import re
import time

import torch
from peft import PeftModel
from transformers import AutoConfig

from models.qwen2_vl import Qwen2VLMRForConditionalGeneration, Qwen2VLMRProcessor
from collators.qwen2_vl import Qwen2VLDataCollator
from collators.qwen_vision_process import process_vision_info
from datasets_mr import VideoCentricDataset

PAD_IDX = -100


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval_data_path", required=True)
    ap.add_argument("--feat_folder", required=True)
    ap.add_argument("--video_folder", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num_samples_to_print", default=10, type=int)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    print(f"loading base: {args.base_model}")
    base = Qwen2VLMRForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device).eval()

    print(f"loading LoRA: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter).eval()

    processor = Qwen2VLMRProcessor.from_pretrained(args.base_model)
    tokenizer = processor.tokenizer
    config = AutoConfig.from_pretrained(args.base_model)
    base.tokenizer = tokenizer

    print(f"loading test set: {args.eval_data_path}")
    eval_ds = VideoCentricDataset(
        data_path=args.eval_data_path,
        video_folder=args.video_folder,
        feat_folder=args.feat_folder,
        fps=2, split="val", num_clips=32, clip_length=32,
        model_family_id="qwen2-vl",
    )
    print(f"test set: {len(eval_ds)} entries")

    collator = Qwen2VLDataCollator(
        config=config, tokenizer=tokenizer, processor=processor,
        mask_question_tokens=True,
    )

    losses, predictions = [], []
    peak_mem = 0

    with torch.no_grad():
        for i in range(len(eval_ds)):
            instance = eval_ds[i]
            try:
                batch = collator([instance])
            except Exception as ex:
                print(f"  [{i}] collator FAIL: {ex}")
                predictions.append({"qid": instance["qid"], "error": f"collator: {ex}"})
                continue

            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            try:
                out = model(**batch_gpu)
            except Exception as ex:
                print(f"  [{i}] forward FAIL: {ex}")
                predictions.append({"qid": instance["qid"], "error": f"forward: {ex}"})
                continue

            loss_val = out.loss.item() if out.loss is not None else None
            if loss_val is not None:
                losses.append(loss_val)

            # Track peak memory
            mem = torch.cuda.max_memory_allocated() / 1024**3
            if mem > peak_mem:
                peak_mem = mem

            # Argmax predictions
            labels = batch_gpu.get("labels")
            if labels is not None:
                labels = labels[0]
                answer_mask = labels != PAD_IDX
                logits = out.logits[0]
                shifted_pred = logits[:-1].argmax(-1)
                mask_shifted = answer_mask[1:]
                pred_ids = shifted_pred[mask_shifted]
                gt_ids = labels[1:][mask_shifted]
                pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True)
            else:
                pred_text, gt_text = "", ""

            # Extract query
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

            if (i + 1) % 10 == 0 or i == len(eval_ds) - 1:
                avg = sum(losses) / max(len(losses), 1)
                print(f"  [{i+1}/{len(eval_ds)}] avg loss = {avg:.4f}")

    elapsed = time.time() - t0
    avg_loss = sum(losses) / max(len(losses), 1)
    n_ok = sum(1 for p in predictions if "error" not in p)
    exact = sum(1 for p in predictions if p.get("exact_match"))

    summary = {
        "experiment": "zero-shot UniTime (Qwen2-VL-7B)",
        "n_test": len(eval_ds), "n_ok": n_ok,
        "avg_eval_loss": round(avg_loss, 4),
        "exact_match_rate": round(exact / max(n_ok, 1), 4),
        "peak_gpu_memory_gb": round(peak_mem, 1),
        "inference_time_sec": round(elapsed, 1),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "predictions": predictions}, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Zero-shot UniTime eval")
    print(f"  {n_ok}/{len(eval_ds)} entries, avg_loss={avg_loss:.4f}")
    print(f"  exact_match: {exact}/{n_ok}")
    print(f"  peak_gpu: {peak_mem:.1f} GB, time: {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"saved {args.output}")

    for p in predictions[:args.num_samples_to_print]:
        if "error" in p:
            continue
        print(f"\n  qid={p['qid']}  '{p['query']}'  loss={p['loss']:.3f}")
        print(f"    gold: {p['gold_text']}")
        print(f"    pred: {p['pred_text']}")


if __name__ == "__main__":
    main()
