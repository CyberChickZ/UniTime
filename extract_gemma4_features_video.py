"""
Extract Gemma4 vision features via the NATIVE VIDEO path.

Uses Gemma4VideoProcessor (max_soft_tokens=70/frame, num_frames=32) instead of
the image processor. This is the official Gemma 4 usage for video — each frame
gets 70 soft tokens with proper 2D position embeddings via the video tower,
rather than the image path which pools to 1 global token per frame.

Output schema (same as other extract scripts):
    {
        "feature":     bf16 tensor [total_video_tokens, hidden_dim]
                       where total_video_tokens ≈ 70 * num_frames
        "frame_idx":   int64 tensor [num_frames]
        "sample_fps":  float
        "num_soft_tokens_per_video": int (total soft tokens for this video)
        "num_frames":  int
    }

Requires UniTime-gemma4 env (transformers 5.x + torch 2.4+).

Usage:
    conda activate UniTime-gemma4
    cd .../experiments/unitime/UniTime
    python extract_gemma4_features_video.py \\
        --video_root /nfs/hpc/dgx2-4/data/TAS_videos/gtea \\
        --feat_root  /nfs/hpc/share/zhanhaoc/MODLE/Gemma4-E4B-it/features_video \\
        --model_local_path /nfs/hpc/share/zhanhaoc/MODLE/Gemma4-E4B-it \\
        --dataset_name gtea --num_frames 32 --gpu 0
"""
import os
os.environ.setdefault("CUDA_HOME", "/usr/local/apps/cuda/12.1")

import faulthandler
faulthandler.enable()

import argparse

import torch
import decord
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from models.gemma4_vl import Gemma4VLMRForConditionalGeneration


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", required=True, type=str)
    ap.add_argument("--feat_root", required=True, type=str)
    ap.add_argument("--model_local_path", required=True, type=str)
    ap.add_argument("--dataset_name", default="gtea", type=str)
    ap.add_argument("--num_frames", default=32, type=int)
    ap.add_argument("--part", default=0, type=int)
    ap.add_argument("--num_parts", default=1, type=int)
    ap.add_argument("--gpu", default=0, type=int)
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

            total_frames = len(vr)
            video_fps = vr.get_avg_fps()
            nframes = min(args.num_frames, total_frames)
            frame_idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
            sample_fps = nframes / total_frames * video_fps

            try:
                frames = vr.get_batch(frame_idx).asnumpy()
            except Exception as ex:
                print(f"decord read failed for {vid}: {ex}")
                continue

            pil_frames = [Image.fromarray(f) for f in frames]

            # Use VIDEO processor (not image processor).
            # do_sample_frames=False: we already sampled, don't re-sample.
            inputs = processor.video_processor(
                pil_frames, do_sample_frames=False, return_tensors="pt"
            )
            pixel_values_videos = inputs["pixel_values_videos"].to(device, dtype=torch.bfloat16)
            video_position_ids = inputs["video_position_ids"].to(device)
            num_soft_tokens = inputs.get("num_soft_tokens_per_video")

            # get_video_features is on Gemma4Model (self.model), not on the
            # ConditionalGeneration wrapper. Access it via model.model.
            out = model.model.get_video_features(
                pixel_values_videos,
                video_position_ids,
                return_dict=True,
            )
            features = out.pooler_output if hasattr(out, "pooler_output") else out
            features = features.cpu()
            # features expected shape: (total_soft_tokens, hidden_dim) or
            # (1, total_soft_tokens, hidden_dim)
            if features.dim() == 3 and features.shape[0] == 1:
                features = features.squeeze(0)

            nsft = int(num_soft_tokens[0].item()) if num_soft_tokens is not None else features.shape[0]
            print(f"  {vid}: features {tuple(features.shape)}, num_soft_tokens={nsft}")

            torch.save(
                {
                    "feature": features,
                    "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
                    "sample_fps": float(sample_fps),
                    "num_soft_tokens_per_video": nsft,
                    "num_frames": nframes,
                },
                out_path,
            )


if __name__ == "__main__":
    main()
