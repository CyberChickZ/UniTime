"""
Extract Qwen3-VL vision features for UniTime training.

Pipeline: 2fps dense sampling → vision encoder → token compression → save.
Output: {feature: [N, H', W', D], frame_idx: [N], sample_fps: float}
feature.shape[0] == frame_idx.shape[0] always.

Requires UniTime-gemma4 env (transformers >= 5.5.0).
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


def resize_feature(feature, h, w):
    return F.interpolate(
        feature.permute(0, 3, 1, 2), size=(h, w), mode='bilinear', align_corners=False
    ).permute(0, 2, 3, 1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--feat_root", required=True)
    ap.add_argument("--model_local_path", required=True)
    ap.add_argument("--dataset_name", default="gtea")
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

    model = Qwen3VLMRForConditionalGeneration.from_pretrained(
        args.model_local_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_local_path)
    merge_size = processor.image_processor.merge_size

    videos = sorted(f for f in os.listdir(args.video_root)
                    if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov")))
    n = len(videos)
    s = args.part * (n // args.num_parts)
    e = n if args.part == args.num_parts - 1 else (args.part + 1) * (n // args.num_parts)
    videos = videos[s:e]

    with torch.no_grad():
        for fname in tqdm(videos):
            vid = os.path.splitext(fname)[0]
            out_path = os.path.join(out_dir, f"{vid}.pt")
            if os.path.exists(out_path):
                continue

            vr = decord.VideoReader(os.path.join(args.video_root, fname), ctx=decord.cpu(0))
            total_frames = len(vr)
            video_fps = vr.get_avg_fps()

            nframes = max(round(total_frames / video_fps * FPS / FRAME_FACTOR) * FRAME_FACTOR, FRAME_FACTOR)
            nframes = min(nframes, total_frames)
            frame_idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
            sample_fps = nframes / total_frames * video_fps

            frames = vr.get_batch(frame_idx).asnumpy()
            pil_frames = [Image.fromarray(f) for f in frames]

            # Extract per-frame features
            all_feats = []
            all_thw = []
            for i in range(0, len(pil_frames), args.batch_size):
                batch = pil_frames[i:i + args.batch_size]
                inputs = processor.image_processor(batch, return_tensors="pt")
                px = inputs["pixel_values"].to(device, torch.bfloat16)
                thw = inputs["image_grid_thw"].to(device)
                feat = model.model.get_image_features(px, thw, return_dict=False)
                if isinstance(feat, (list, tuple)):
                    feat = torch.cat([f.squeeze(0) if f.dim() == 3 else f for f in feat], dim=0)
                all_feats.append(feat.cpu())
                all_thw.append(thw.cpu())

            feats_flat = torch.cat(all_feats, dim=0)
            thw_all = torch.cat(all_thw, dim=0)

            # Split flat features back to per-frame by token count
            tokens_per_frame = []
            for thw in thw_all:
                t, h, w = thw.tolist()
                tokens_per_frame.append(int(t * (h // merge_size) * (w // merge_size)))
            per_frame = torch.split(feats_flat, tokens_per_frame)

            # Resize each frame to common spatial shape, then stack
            target_h = thw_all[0][1].item() // merge_size
            target_w = thw_all[0][2].item() // merge_size
            native_tokens = target_h * target_w

            frames_4d = []
            for ff in per_frame:
                if ff.shape[0] == native_tokens:
                    frames_4d.append(ff.reshape(target_h, target_w, -1))
                else:
                    side = int(math.sqrt(ff.shape[0]))
                    if side * side != ff.shape[0]:
                        side_h = int(math.sqrt(ff.shape[0] * target_h / target_w))
                        side_w = ff.shape[0] // max(side_h, 1)
                    else:
                        side_h = side_w = side
                    resized = F.interpolate(
                        ff.reshape(1, side_h, side_w, -1).permute(0, 3, 1, 2).float(),
                        size=(target_h, target_w), mode='bilinear', align_corners=False
                    ).permute(0, 2, 3, 1).squeeze(0).to(ff.dtype)
                    frames_4d.append(resized)

            feats_stacked = torch.stack(frames_4d, dim=0)  # [nframes, h, w, D]

            # Token compression
            res_side = max(int(math.sqrt(max(args.n_total // nframes, 4))), 2)
            feats_out = resize_feature(feats_stacked, res_side, res_side)

            assert feats_out.shape[0] == len(frame_idx), \
                f"feature {feats_out.shape[0]} != frame_idx {len(frame_idx)}"

            torch.save({
                "feature": feats_out,
                "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
                "sample_fps": float(sample_fps),
            }, out_path)

            print(f"  {vid}: {nframes}fr, {native_tokens}→{res_side**2}tok/fr, total={nframes*res_side**2}")


if __name__ == "__main__":
    main()
