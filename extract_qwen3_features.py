"""
Extract Qwen3-VL vision features for UniTime training.

Aligned with UniTime upstream: 2fps dense + token compression.
Per-frame image processing → get_image_features → resize_feature.

Requires UniTime-gemma4 env (transformers >= 5.5.0).

Usage:
    conda activate UniTime-gemma4
    python extract_qwen3_features.py \\
        --video_root /nfs/hpc/dgx2-4/data/TAS_videos/gtea \\
        --feat_root  /nfs/hpc/dgx2-4/tmp/2026/4/6/feature/Qwen3-VL-2B-Instruct \\
        --model_local_path /nfs/hpc/share/zhanhaoc/MODLE/Qwen3-VL-2B-Instruct \\
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

from models.qwen3_vl import Qwen3VLMRForConditionalGeneration

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

    print(f"loading Qwen3-VL from {args.model_local_path}...")
    model = Qwen3VLMRForConditionalGeneration.from_pretrained(
        args.model_local_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_local_path)

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

            # Qwen3-VL: process each frame as image, extract features
            all_features = []
            bs = args.batch_size
            for bi in range(0, len(pil_frames), bs):
                batch_pil = pil_frames[bi:bi + bs]
                inputs = processor.image_processor(batch_pil, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
                image_grid_thw = inputs["image_grid_thw"].to(device)

                feats = model.model.get_image_features(
                    pixel_values, image_grid_thw, return_dict=False
                )
                # feats: list of tensors or single tensor, one per image
                if isinstance(feats, (list, tuple)):
                    feats = torch.cat([f if f.dim() == 2 else f.squeeze(0) for f in feats], dim=0)
                all_features.append(feats.cpu())

            features_flat = torch.cat(all_features, dim=0)
            # features_flat: [total_tokens_all_frames, hidden_dim]

            # Split back to per-frame using image_grid_thw
            # Re-process all to get the thw info
            all_inputs = processor.image_processor(pil_frames, return_tensors="pt")
            all_thw = all_inputs["image_grid_thw"]
            merge_size = processor.image_processor.merge_size
            tokens_per_frame = []
            for thw in all_thw:
                t, h, w = thw[0].item(), thw[1].item(), thw[2].item()
                tokens_per_frame.append(t * (h // merge_size) * (w // merge_size))

            # Reshape each frame's flat features to 2D spatial
            per_frame_features = torch.split(features_flat, tokens_per_frame)
            # Find common spatial shape for all frames
            native_h = all_thw[0][1].item() // merge_size
            native_w = all_thw[0][2].item() // merge_size
            native_tokens = native_h * native_w

            # Stack into [nframes, H, W, D]
            features_4d = []
            for ff in per_frame_features:
                if ff.shape[0] == native_tokens:
                    features_4d.append(ff.reshape(native_h, native_w, -1))
                else:
                    # Different spatial size → resize to common
                    h_i = int(math.sqrt(ff.shape[0] * native_h / native_w))
                    w_i = ff.shape[0] // max(h_i, 1)
                    if h_i * w_i != ff.shape[0]:
                        h_i = int(math.sqrt(ff.shape[0]))
                        w_i = h_i
                    resized = F.interpolate(
                        ff.reshape(1, h_i, w_i, -1).permute(0, 3, 1, 2),
                        size=(native_h, native_w), mode='bilinear', align_corners=False
                    ).permute(0, 2, 3, 1).squeeze(0)
                    features_4d.append(resized)

            features_stacked = torch.stack(features_4d, dim=0)
            # features_stacked: [nframes, native_h, native_w, D]

            # Token compression to fit budget
            n_res = max(args.n_total // nframes, 4)
            res_side = max(int(math.sqrt(n_res)), 2)
            features_compressed = resize_feature(features_stacked, res_side, res_side)

            fps_sample_idx = [int((x + y) / 2) for x, y in zip(frame_idx[::2], frame_idx[1::2])]

            print(f"  {vid}: {nframes} frames, {native_tokens}→{res_side}x{res_side}={res_side**2}/frame, "
                  f"total={nframes * res_side**2}")

            torch.save(
                {
                    "feature": features_compressed,
                    "frame_idx": torch.tensor(fps_sample_idx, dtype=torch.long),
                    "sample_fps": float(sample_fps),
                },
                out_path,
            )


if __name__ == "__main__":
    main()
