"""
Evaluate a UniTime + Qwen3-VL LoRA checkpoint on GTEA.
Pixel-values path (per-frame images), teacher-forced argmax.

Requires UniTime-gemma4 env (transformers >= 5.5.0).

Usage:
    conda activate UniTime-gemma4
    cd experiments/unitime/UniTime
    python eval_qwen3vl.py \
        --base_model /path/to/Qwen3-VL-2B-Instruct \
        --adapter ./checkpoints/qwen3vl_2b_gtea_run1 \
        --eval_data_path .../test.json \
        --video_folder /path/to/gtea \
        --output ./results/qwen3vl_2b_gtea_run1/eval.json
"""
import os
os.environ.setdefault("CUDA_HOME", "/usr/local/apps/cuda/12.1")

import faulthandler
faulthandler.enable()

import argparse
import json
import time

import torch
import decord
from PIL import Image
from peft import PeftModel
from transformers import AutoConfig, AutoProcessor

from models.qwen3_vl import Qwen3VLMRForConditionalGeneration

PAD_IDX = -100


def find_segments(sample_timestamps, gt_window):
    candidates_start = [x for x in sample_timestamps if x <= gt_window[0]]
    closest_start = max(candidates_start) if candidates_start else sample_timestamps[0]
    start_idx = sample_timestamps.index(closest_start)
    candidates_end = [x for x in sample_timestamps if x <= gt_window[1]]
    closest_end = max(candidates_end) if candidates_end else sample_timestamps[0]
    end_idx = sample_timestamps.index(closest_end)
    return start_idx, end_idx


def build_target_text(sampled_timestamps, windows):
    hit = []
    for window in windows:
        s_idx, e_idx = find_segments(sampled_timestamps, window)
        hit.extend(sampled_timestamps[s_idx:e_idx + 1])
    seen = set()
    ordered = []
    for t in hit:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    if not ordered:
        return "(no timestamps)"
    return ", ".join(f"{t} seconds" for t in ordered) + "."


def load_frames(video_path, num_frames=32):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total = len(vr)
    fps = vr.get_avg_fps()
    n = min(num_frames, total)
    idx = torch.linspace(0, total - 1, n).round().long().tolist()
    frames = vr.get_batch(idx).asnumpy()
    timestamps = [round(i / fps, 1) for i in idx]
    return [Image.fromarray(f) for f in frames], timestamps


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval_data_path", required=True)
    ap.add_argument("--video_folder", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num_samples_to_print", default=10, type=int)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    print(f"loading base model: {args.base_model}")
    base = Qwen3VLMRForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    print(f"loading LoRA: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter).eval()

    processor = AutoProcessor.from_pretrained(args.base_model)
    tokenizer = processor.tokenizer

    test_data = json.load(open(args.eval_data_path))
    print(f"test set: {len(test_data)} entries")

    losses, predictions = [], []
    peak_mem = 0

    with torch.no_grad():
        for i, entry in enumerate(test_data):
            qid = entry["qid"]
            vid = entry["id"]
            annos = entry["annos"]
            if len(annos) != 1:
                continue
            query = annos[0]["query"]
            windows = annos[0]["window"]

            video_path = os.path.join(args.video_folder, f"{vid}.mp4")
            if not os.path.exists(video_path):
                video_path = os.path.join(args.video_folder, f"{vid}.avi")

            pil_frames, sampled_timestamps = load_frames(video_path)
            target_text = build_target_text(sampled_timestamps, windows)

            user_content = [
                {"type": "text", "text": "This is a sequence interleaved with timestamps and frames. "
                 "Your task is to identify the specific timestamp(s) when the given query appears.\n"},
            ]
            for t in sampled_timestamps:
                user_content.append({"type": "text", "text": f"timestamp: {t} seconds "})
                user_content.append({"type": "image"})
            user_content.append({"type": "text", "text": f"\nQuery: {query}\nAnswer:"})

            full_messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
            ]
            prompt_messages = [{"role": "user", "content": user_content}]

            full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

            full_inputs = processor(text=[full_text], images=pil_frames, return_tensors="pt", padding=False)
            prompt_inputs = processor(text=[prompt_text], images=pil_frames, return_tensors="pt", padding=False)

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in full_inputs.items()}
            batch["labels"] = batch["input_ids"].clone()
            prompt_len = prompt_inputs["input_ids"].shape[1]
            batch["labels"][0, :prompt_len] = PAD_IDX

            try:
                out = model(**batch)
            except Exception as ex:
                print(f"  [{i}] FAIL: {ex}")
                predictions.append({"qid": qid, "error": str(ex)})
                continue

            loss_val = out.loss.item() if out.loss is not None else None
            if loss_val is not None:
                losses.append(loss_val)

            mem = torch.cuda.max_memory_allocated() / 1024**3
            if mem > peak_mem:
                peak_mem = mem

            logits = out.logits[0]
            labels = batch["labels"][0]
            answer_mask = labels != PAD_IDX
            shifted_pred_ids = logits[:-1].argmax(-1)
            answer_mask_shifted = answer_mask[1:]
            pred_answer_ids = shifted_pred_ids[answer_mask_shifted]
            gt_answer_ids = labels[1:][answer_mask_shifted]

            pred_text = tokenizer.decode(pred_answer_ids, skip_special_tokens=True)
            gt_text = tokenizer.decode(gt_answer_ids, skip_special_tokens=True)

            predictions.append({
                "qid": qid, "video_id": vid, "query": query,
                "loss": loss_val, "gold_text": gt_text, "pred_text": pred_text,
                "exact_match": pred_text.strip() == gt_text.strip(),
            })

            if (i + 1) % 10 == 0 or i == len(test_data) - 1:
                avg = sum(losses) / max(len(losses), 1)
                print(f"  [{i+1}/{len(test_data)}] avg loss = {avg:.4f}")

    elapsed = time.time() - t0
    avg_loss = sum(losses) / max(len(losses), 1)
    exact_matches = sum(1 for p in predictions if p.get("exact_match"))
    n_ok = sum(1 for p in predictions if "error" not in p)

    summary = {
        "experiment": "Qwen3-VL-2B (timestamp)",
        "n_test": len(test_data), "n_ok": n_ok,
        "avg_eval_loss": round(avg_loss, 4),
        "exact_match_rate": round(exact_matches / max(n_ok, 1), 4),
        "peak_gpu_memory_gb": round(peak_mem, 1),
        "inference_time_sec": round(elapsed, 1),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "predictions": predictions}, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  {summary['experiment']}")
    print(f"  {n_ok}/{len(test_data)} entries, avg_loss={avg_loss:.4f}")
    print(f"  exact_match: {exact_matches}/{n_ok}")
    print(f"  peak_gpu: {peak_mem:.1f} GB, time: {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"saved {args.output}")

    for p in predictions[:args.num_samples_to_print]:
        if "error" in p:
            continue
        print(f"\n  qid={p['qid']}  {p['video_id']}  '{p['query']}'  loss={p['loss']:.3f}")
        print(f"    gold: {p['gold_text']}")
        print(f"    pred: {p['pred_text']}")


if __name__ == "__main__":
    main()
