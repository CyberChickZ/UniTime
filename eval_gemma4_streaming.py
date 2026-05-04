"""
Streaming eval for UniTime + Gemma4: sliding-window inference over full video.

Instead of 32 uniform frames, slides a window of W frames with stride S across
all 2fps-sampled frames. Per-frame action labels are aggregated across
overlapping windows via majority vote. This gives finer temporal resolution
and scales to arbitrarily long videos.

Usage:
    python eval_gemma4_streaming.py \
        --base_model /path/to/Gemma4-E4B-it \
        --adapter ./checkpoints/gemma4_gtea_2gpu_e5 \
        --eval_data_path .../test.json \
        --video_folder /path/to/gtea \
        --window_size 32 --stride 16 \
        --output ./results/gemma4_streaming/eval.json
"""
import os
os.environ.setdefault("CUDA_HOME", "/usr/local/apps/cuda/12.1")

import faulthandler
faulthandler.enable()

import argparse
import json
import re
from collections import defaultdict

import torch
import torch.distributed as dist

for k, v in {"MASTER_ADDR": "localhost", "MASTER_PORT": "29501",
             "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}.items():
    os.environ.setdefault(k, v)
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")
torch.cuda.set_device(0)

import decord
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor

from models.gemma4_vl import Gemma4VLMRForConditionalGeneration

TS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*seconds")


def load_all_frames_2fps(video_path):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total = len(vr)
    fps = vr.get_avg_fps()
    sample_interval = max(round(fps / 2), 1)
    idx = list(range(0, total, sample_interval))
    if not idx:
        idx = [0]
    frames = vr.get_batch(idx).asnumpy()
    timestamps = [round(i / fps, 1) for i in idx]
    return [Image.fromarray(f) for f in frames], timestamps


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


def run_window(model, processor, pil_frames, timestamps, query, device):
    user_content = [
        {"type": "text", "text": "This is a sequence interleaved with timestamps and frames. "
         "Your task is to identify the specific timestamp(s) when the given query appears.\n"},
    ]
    for t, _ in zip(timestamps, pil_frames):
        user_content.append({"type": "text", "text": f"timestamp: {t} seconds "})
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": f"\nQuery: {query}\nAnswer:"})

    messages = [{"role": "user", "content": user_content}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=pil_frames, return_tensors="pt", padding=False)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **batch,
            max_new_tokens=512,
            do_sample=False,
        )
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = out[0][prompt_len:]
    pred_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return pred_text


def parse_timestamp_hits(pred_text, timestamps, tol=0.5):
    pred_ts = set()
    for x in TS_RE.findall(pred_text):
        pred_ts.add(round(float(x), 1))
    hits = set()
    for fi, t in enumerate(timestamps):
        if any(abs(t - pt) <= tol for pt in pred_ts):
            hits.add(fi)
    return hits


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval_data_path", required=True)
    ap.add_argument("--video_folder", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--window_size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    print(f"loading base model: {args.base_model}")
    base = Gemma4VLMRForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()
    print(f"loading LoRA: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter).eval()

    processor = AutoProcessor.from_pretrained(args.base_model)

    test_data = json.load(open(args.eval_data_path))
    print(f"test set: {len(test_data)} entries")
    print(f"streaming: window_size={args.window_size}, stride={args.stride}")

    video_cache = {}
    predictions = []

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

            if vid not in video_cache:
                all_frames, all_ts = load_all_frames_2fps(video_path)
                video_cache[vid] = (all_frames, all_ts)
            all_frames, all_ts = video_cache[vid]

            n_total = len(all_frames)
            W = args.window_size
            S = args.stride

            # Sliding window predictions
            frame_hits = defaultdict(int)
            frame_total = defaultdict(int)
            all_pred_texts = []

            for start in range(0, n_total, S):
                end = min(start + W, n_total)
                if end - start < 4:
                    continue
                win_frames = all_frames[start:end]
                win_ts = all_ts[start:end]

                try:
                    pred_text = run_window(model, processor, win_frames, win_ts, query, device)
                except Exception as ex:
                    print(f"  [{i}] window {start}-{end} FAIL: {ex}")
                    continue

                all_pred_texts.append(pred_text)
                hits = parse_timestamp_hits(pred_text, win_ts)
                for local_idx in range(len(win_ts)):
                    global_idx = start + local_idx
                    frame_total[global_idx] += 1
                    if local_idx in hits:
                        frame_hits[global_idx] += 1

            # Aggregate: frame is "hit" if majority of windows say so
            merged_pred_ts = []
            for fi in range(n_total):
                if frame_total.get(fi, 0) > 0:
                    ratio = frame_hits.get(fi, 0) / frame_total[fi]
                    if ratio > 0.5:
                        merged_pred_ts.append(all_ts[fi])

            merged_pred_text = ", ".join(f"{t} seconds" for t in merged_pred_ts) + "." if merged_pred_ts else "(no timestamps)"
            gt_text = build_target_text(all_ts, windows)

            predictions.append({
                "qid": qid, "video_id": vid, "query": query,
                "n_frames": n_total,
                "n_windows": len(all_pred_texts),
                "gold_text": gt_text,
                "pred_text": merged_pred_text,
                "exact_match": merged_pred_text.strip() == gt_text.strip(),
            })

            if (i + 1) % 5 == 0 or i == len(test_data) - 1:
                print(f"  [{i+1}/{len(test_data)}] {vid}/{query}: {n_total}fr, {len(all_pred_texts)} windows")

    n_ok = sum(1 for p in predictions if "error" not in p)
    exact = sum(1 for p in predictions if p.get("exact_match"))
    print(f"\nstreaming eval done: {n_ok} entries, exact_match={exact}/{n_ok}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": {"total": len(test_data), "n_ok": n_ok,
                                "exact_match_rate": exact / max(n_ok, 1),
                                "window_size": args.window_size, "stride": args.stride},
                    "predictions": predictions}, f, indent=2, default=str)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
