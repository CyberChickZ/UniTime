"""
Extract Gemma4 vision features for UniTime training.

Aligned with UniTime upstream: 2fps dense + token compression.
Uses per-frame IMAGE processing (not video processor) to get features,
then bilinear resize to fit N_total budget.

Output schema (same as Qwen2-VL feature_offline.py):
    {
        "feature":     bf16 tensor [N, H', W', hidden_dim]
        "frame_idx":   int64 tensor [N]
        "sample_fps":  float
    }

Requires UniTime-gemma4 env (transformers >= 5.0).

Usage:
    conda activate UniTime-gemma4
    python extract_gemma4_features.py \\
        --video_root /nfs/hpc/dgx2-4/data/TAS_videos/gtea \\
        --feat_root  /nfs/hpc/dgx2-4/tmp/2026/4/6/feature/Gemma4-E4B-it \\
        --model_local_path /nfs/hpc/share/zhanhaoc/MODLE/Gemma4-E4B-it \\
        --dataset_name gtea --gpu 0
"""
import os
os.environ.setdefault("CUDA_HOME", "/usr/local/apps/cuda/12.1")

import faulthandler
faulthandler.enable()

import argparse
import math

import torch
import torch.nn.functional as F
import decord
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from models.gemma4_vl import Gemma4VLMRForConditionalGeneration

FRAME_FACTOR = 2
FPS = 2.0
N_TOTAL = 12288


def round_by_factor(number, factor):
    return round(number / factor) * factor


def resize_feature(feature, resize_h, resize_w):
    feature = feature.permute(0, 3, 1, 2)
    feature_resized = F.interpolate(feature, size=(resize_h, resize_w), mode='bilinear', align_corners=False)
    feature_resized = feature_resized.permute(0, 2, 3, 1)
    return feature_resized


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", required=True, type=str)
    ap.add_argument("--feat_root", required=True, type=str)
    ap.add_argument("--model_local_path", required=True, type=str)
    ap.add_argument("--dataset_name", default="gtea", type=str)
    ap.add_argument("--n_total", default=N_TOTAL, type=int)
    ap.add_argument("--part", default=0, type=int)
    ap.add_argument("--num_parts", default=1, type=int)
    ap.add_argument("--gpu", default=0, type=int)
    ap.add_argument("--batch_size", default=4, type=int)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    out_dir = os.path.join(args.feat_root, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"loading Gemma4 from {args.model_local_path}...")
    model = Gemma4VLMRForConditionalGeneration.from_pretrained(
        args.model_local_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_local_path)

    mm_tokens = getattr(model.config, "mm_tokens_per_image", 256)
    mm_side = int(math.sqrt(mm_tokens))
    print(f"mm_tokens_per_image={mm_tokens}, spatial={mm_side}x{mm_side}")

    valid_ext = (".mp4", ".avi", ".mkv", ".mov", ".webm")
    all_videos = sorted(
        f for f in os.listdir(args.video_root) if f.lower().endswith(valid_ext)
    )

    total = len(all_videos)
    part_size = total // args.num_parts
    s = args.part * part_size
    e = (args.part + 1) * part_size if args.part != args.num_parts - 1 else total
    subset = all_videos[s:e]
    print(f"part {args.part}/{args.num_parts}: {len(subset)} videos")

    with torch.no_grad():
        for filename in tqdm(subset):
            vid = os.path.splitext(filename)[0]
            video_path = os.path.join(args.video_root, filename)
            out_path = os.path.join(out_dir, f"{vid}.pt")
            if os.path.exists(out_path):
                continue

            try:
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            except Exception as ex:
                print(f"decord open failed for {video_path}: {ex}")
                continue

            total_frames_vid = len(vr)
            video_fps = vr.get_avg_fps()

            nframes = max(
                round_by_factor(int(total_frames_vid / video_fps * FPS), FRAME_FACTOR),
                FRAME_FACTOR
            )
            nframes = min(nframes, total_frames_vid)

            frame_idx = torch.linspace(0, total_frames_vid - 1, nframes).round().long().tolist()
            sample_fps = nframes / total_frames_vid * video_fps

            try:
                raw_frames = vr.get_batch(frame_idx).asnumpy()
            except Exception as ex:
                print(f"decord read failed for {vid}: {ex}")
                continue

            pil_frames = [Image.fromarray(f) for f in raw_frames]

            all_features = []
            bs = args.batch_size
            for bi in range(0, len(pil_frames), bs):
                batch_pil = pil_frames[bi:bi + bs]
                inputs = processor.image_processor(batch_pil, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
                if "image_position_ids" in inputs:
                    image_position_ids = inputs["image_position_ids"].to(device)
                else:
                    image_position_ids = None
                feats = model.get_image_features(pixel_values, image_position_ids).cpu()
                all_features.append(feats)

            features = torch.cat(all_features, dim=0)
            # features: [nframes, mm_tokens, hidden_dim]

            n_res = max(args.n_total // nframes, 4)
            res_side = max(int(math.sqrt(n_res)), 2)

            features_4d = features.reshape(features.shape[0], mm_side, mm_side, features.shape[-1])
            features_compressed = resize_feature(features_4d, res_side, res_side)

            print(f"  {vid}: {nframes} frames, {mm_tokens}→{res_side}x{res_side}={res_side**2}/frame, "
                  f"total={nframes * res_side**2}")

            torch.save(
                {
                    "feature": features_compressed,
                    "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
                    "sample_fps": float(sample_fps),
                },
                out_path,
            )


if __name__ == "__main__":
    main()
