"""
Extract Gemma4 vision features for UniTime training.
Pipeline: 2fps dense → vision encoder → token compression → save.
Requires UniTime-gemma4 env (transformers >= 5.0).
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


def resize_feature(feature, h, w):
    return F.interpolate(
        feature.permute(0, 3, 1, 2), size=(h, w), mode='bilinear', align_corners=False
    ).permute(0, 2, 3, 1)


def find_hw(n):
    """Find (h, w) factors of n closest to square."""
    best = (1, n)
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            best = (i, n // i)
    return best


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

    model = Gemma4VLMRForConditionalGeneration.from_pretrained(
        args.model_local_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_local_path)

    hidden_dim = model.config.text_config.hidden_size
    print(f"text hidden_dim={hidden_dim}")

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

            # Extract per-frame: pooler_output is [tokens, hidden_dim] (already
            # projected to text hidden_dim by embed_vision). Token count may vary
            # across images due to adaptive resolution, so process one at a time.
            per_frame_feats = []
            for img in pil_frames:
                inputs = processor.image_processor([img], return_tensors="pt")
                px = inputs["pixel_values"].to(device, torch.bfloat16)
                pos_ids = inputs.get("image_position_ids")
                if pos_ids is not None:
                    pos_ids = pos_ids.to(device)
                out = model.get_image_features(px, pos_ids)
                po = out.pooler_output.cpu()  # [tokens_this_image, hidden_dim]
                per_frame_feats.append(po)

            # Token compression: resize each frame's tokens to a fixed grid
            n_res = max(args.n_total // nframes, 4)
            res_h, res_w = find_hw(n_res) if n_res > 4 else (2, 2)

            compressed = []
            for po in per_frame_feats:
                n_tok = po.shape[0]
                src_h, src_w = find_hw(n_tok)
                frame_4d = po.reshape(1, src_h, src_w, -1)  # [1, h, w, D]
                frame_resized = resize_feature(frame_4d, res_h, res_w)  # [1, res_h, res_w, D]
                compressed.append(frame_resized.squeeze(0))
            feats_out = torch.stack(compressed, dim=0)  # [nframes, res_h, res_w, D]

            assert feats_out.shape[0] == len(frame_idx)
            assert feats_out.shape[-1] == hidden_dim, (
                f"expected D={hidden_dim}, got {feats_out.shape[-1]}"
            )

            torch.save({
                "feature": feats_out,
                "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
                "sample_fps": float(sample_fps),
            }, out_path)

            tok0 = per_frame_feats[0].shape[0]
            print(f"  {vid}: {nframes}fr, {tok0}->{res_h}x{res_w}={res_h*res_w}tok/fr, total={nframes*res_h*res_w}")


if __name__ == "__main__":
    main()
