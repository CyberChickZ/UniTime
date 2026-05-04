"""
Gemma4 eval aligned with training pipeline.

Key differences from the old eval_gemma4.py:
  1. Uses cached features (same .pt files as training), not pixel-values
  2. Autoregressive generation (model.generate), not teacher-forced argmax
  3. Same combine_timestamps logic as training collator
  4. Outputs actual frame count for correct TAS metric computation

Usage:
    python eval_gemma4_official.py \
        --base_model /path/to/Gemma4-E4B-it \
        --adapter ./checkpoints/gemma4_gtea_run1 \
        --eval_data_path .../test.json \
        --feat_folder /path/to/features/gtea \
        --video_folder /path/to/gtea \
        --output ./results/gemma4_gtea_run1/eval_official.json
"""
import os
os.environ.setdefault("CUDA_HOME", "/usr/local/apps/cuda/12.1")

import faulthandler
faulthandler.enable()

import argparse
import gc
import json
import re

import torch
import torch.distributed as dist

for k, v in {"MASTER_ADDR": "localhost", "MASTER_PORT": "29501",
             "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}.items():
    os.environ.setdefault(k, v)
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")
torch.cuda.set_device(0)

import decord
from peft import PeftModel
from transformers import AutoProcessor

from models.gemma4_vl import Gemma4VLMRForConditionalGeneration
from collators.gemma_vision_process import _combine_timestamps_gemma

TS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*seconds")


def load_cached_features(feat_folder, vid):
    pt_path = os.path.join(feat_folder, f"{vid}.pt")
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    return payload["feature"], payload["frame_idx"], float(payload["sample_fps"])


def build_input_ids(tokenizer, config, sampled_timestamps, query,
                    combine_t_list, tokens_per_image):
    def tok(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    bos = tokenizer.bos_token or ""
    instruction = (
        "This is a sequence interleaved with timestamps and frames. "
        "Your task is to identify the specific timestamp(s) when the given query appears.\n"
    )
    header_ids = tok(f"{bos}<|turn>user\n{instruction}")

    boi_id = config.boi_token_id
    eoi_id = config.eoi_token_id
    img_tok = config.image_token_id
    newline_ids = tok("\n\n")

    body_ids = []
    if combine_t_list is None:
        combine_t_list = [1] * len(sampled_timestamps)
    for t, n_frames in zip(sampled_timestamps, combine_t_list):
        body_ids.extend(tok(f"timestamp: {t} seconds"))
        for _ in range(n_frames):
            body_ids.extend(newline_ids)
            body_ids.append(boi_id)
            body_ids.extend([img_tok] * tokens_per_image)
            body_ids.append(eoi_id)
            body_ids.extend(newline_ids)

    query_ids = tok(f"\nQuery: {query}\nAnswer:")
    turn_end_ids = tok("<turn|>\n<|turn>model\n")

    return header_ids + body_ids + query_ids + turn_end_ids


def find_segments(sample_timestamps, gt_window):
    candidates_start = [x for x in sample_timestamps if x <= gt_window[0]]
    closest_start = max(candidates_start) if candidates_start else sample_timestamps[0]
    start_idx = sample_timestamps.index(closest_start)
    candidates_end = [x for x in sample_timestamps if x <= gt_window[1]]
    closest_end = max(candidates_end) if candidates_end else sample_timestamps[0]
    end_idx = sample_timestamps.index(closest_end)
    return start_idx, end_idx


def build_gt_text(sampled_timestamps, windows):
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval_data_path", required=True)
    ap.add_argument("--feat_folder", required=True)
    ap.add_argument("--video_folder", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num_clips", type=int, default=32)
    ap.add_argument("--clip_length", type=int, default=-1)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0")

    print(f"loading base model: {args.base_model}")
    base = Gemma4VLMRForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()
    print(f"loading LoRA: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter).eval()

    processor = AutoProcessor.from_pretrained(args.base_model)
    tokenizer = processor.tokenizer
    config = base.config

    test_data = json.load(open(args.eval_data_path))
    print(f"test set: {len(test_data)} entries")

    # Pre-cache video fps
    video_fps_cache = {}
    for entry in test_data:
        vid = entry["id"]
        if vid not in video_fps_cache:
            for ext in [".mp4", ".avi"]:
                vpath = os.path.join(args.video_folder, f"{vid}{ext}")
                if os.path.exists(vpath):
                    break
            try:
                vr = decord.VideoReader(vpath, ctx=decord.cpu(0))
                video_fps_cache[vid] = vr.get_avg_fps()
            except Exception:
                video_fps_cache[vid] = 30.0

    predictions = []

    with torch.no_grad():
        for i, entry in enumerate(test_data):
            torch.cuda.empty_cache()
            gc.collect()

            qid = entry["qid"]
            vid = entry["id"]
            annos = entry["annos"]
            if len(annos) != 1:
                continue
            query = annos[0]["query"]
            windows = annos[0]["window"]
            duration = entry.get("duration", 0)

            try:
                feature, frame_idx, sample_fps = load_cached_features(args.feat_folder, vid)
            except Exception as ex:
                print(f"  [{i}] SKIP {vid}: {ex}")
                predictions.append({"qid": qid, "error": str(ex)})
                continue

            video_fps = video_fps_cache.get(vid, 30.0)
            T = feature.shape[0]
            sampled_timestamps = [round(idx.item() / video_fps, 1) for idx in frame_idx]

            cl = args.clip_length
            if cl > 0:
                cl = max(int(cl * sample_fps / 2), 1)
            else:
                cl = -1
            feat_combined, ts_combined, comb_t_list = _combine_timestamps_gemma(
                feature, sampled_timestamps, num_clips=args.num_clips, clip_length=cl
            )

            if feat_combined.dim() == 4:
                tokens_per_image = feat_combined.shape[1] * feat_combined.shape[2]
            else:
                tokens_per_image = feat_combined.shape[1]

            prompt_ids = build_input_ids(
                tokenizer, config, ts_combined, query,
                comb_t_list, tokens_per_image,
            )

            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            feature_flat = feat_combined.reshape(-1, feat_combined.shape[-1]).to(device, torch.bfloat16)

            n_slots = (input_ids == config.image_token_id).sum().item()
            if n_slots != feature_flat.shape[0]:
                print(f"  [{i}] SKIP {vid}: slots {n_slots} != features {feature_flat.shape[0]}")
                predictions.append({"qid": qid, "error": f"slot mismatch {n_slots} vs {feature_flat.shape[0]}"})
                continue

            try:
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    feature_inputs=feature_flat,
                    max_new_tokens=512,
                    do_sample=False,
                )
            except Exception as ex:
                print(f"  [{i}] FAIL {vid}/{query}: {ex}")
                predictions.append({"qid": qid, "error": str(ex)})
                continue

            generated_ids = out[0][input_ids.shape[1]:]
            pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            gold_text = build_gt_text(ts_combined, windows)

            predictions.append({
                "qid": qid, "video_id": vid, "query": query,
                "duration": duration,
                "n_frames": T,
                "n_timestamps": len(ts_combined),
                "sampled_timestamps": ts_combined,
                "gold_text": gold_text,
                "pred_text": pred_text,
                "exact_match": pred_text.strip() == gold_text.strip(),
            })

            if (i + 1) % 10 == 0 or i == len(test_data) - 1:
                n_ok = sum(1 for p in predictions if "error" not in p)
                n_em = sum(1 for p in predictions if p.get("exact_match"))
                print(f"  [{i+1}/{len(test_data)}] ok={n_ok} exact_match={n_em}")

    n_ok = sum(1 for p in predictions if "error" not in p)
    n_em = sum(1 for p in predictions if p.get("exact_match"))
    print(f"\neval done: {n_ok}/{len(predictions)}, exact_match={n_em}/{n_ok}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "summary": {
                "total": len(test_data), "n_ok": n_ok,
                "exact_match": n_em,
                "num_clips": args.num_clips,
                "clip_length": args.clip_length,
                "eval_mode": "generate_cached_features",
            },
            "predictions": predictions,
        }, f, indent=2, default=str)
    print(f"saved {args.output}")

    for p in predictions[:5]:
        if "error" in p:
            continue
        print(f"\n  qid={p['qid']} {p['video_id']}/{p['query']} ({p['n_frames']}fr, {p['n_timestamps']}ts)")
        print(f"    gold: {p['gold_text']}")
        print(f"    pred: {p['pred_text']}")


if __name__ == "__main__":
    main()
