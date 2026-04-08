"""
Extract Gemma3 vision features for UniTime training.

Reads videos under --video_root, samples N uniform frames per video, runs them
through Gemma3's vision_tower + multi_modal_projector, saves the resulting
[N, mm_tokens_per_image, text_hidden_size] tensor as a .pt file.

Output schema (mirrors what `extract_qwen_embeddings.py` writes, so the
collator pipeline can interpret either):
    {
        "feature":     bf16 tensor [N, 256, hidden_dim]
        "frame_idx":   int64 tensor [N]   (raw video frame indices sampled)
        "sample_fps":  float              (effective sampling fps)
    }

Usage:
    python extract_gemma_features.py \\
        --video_root      /nfs/hpc/dgx2-4/data/TAS_videos/gtea \\
        --feat_root       /nfs/hpc/share/zhanhaoc/.../feature/Gemma3-4B \\
        --model_local_path /nfs/hpc/share/zhanhaoc/MODLE/Gemma3-4B-it \\
        --dataset_name gtea \\
        --num_frames 32 \\
        --gpu 0
"""
import argparse
import os

import decord
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

# Will fail if transformers < 4.50; we expect the env to be upgraded to 4.51.3+
from models.gemma3_vl import Gemma3VLMRForConditionalGeneration


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", required=True, type=str)
    ap.add_argument("--feat_root", required=True, type=str,
                    help="Output root; final dir is {feat_root}/{dataset_name}")
    ap.add_argument("--model_local_path", required=True, type=str)
    ap.add_argument("--dataset_name", default="gtea", type=str)
    ap.add_argument("--num_frames", default=32, type=int,
                    help="Uniform-sampled frames per video. Trade-off: more frames "
                         "= more image tokens = bigger LLM context. 32 frames at "
                         "256 tokens/frame = 8192 image tokens.")
    ap.add_argument("--part", default=0, type=int)
    ap.add_argument("--num_parts", default=1, type=int)
    ap.add_argument("--gpu", default=0, type=int)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    out_dir = os.path.join(args.feat_root, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"loading Gemma3 from {args.model_local_path}...")
    model = Gemma3VLMRForConditionalGeneration.from_pretrained(
        args.model_local_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
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

            # Convert to PIL → Gemma image processor → pixel_values
            pil_frames = [Image.fromarray(f) for f in frames]
            inputs = processor.image_processor(pil_frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
            # Gemma3 image processor may produce extra crops via pan_and_scan;
            # we keep do_pan_and_scan=False (default) so num_crops == num_images.

            # Run vision_tower + multi_modal_projector
            features = model.get_image_features(pixel_values).cpu()
            # features shape: (nframes, mm_tokens_per_image=256, text_hidden=2560)

            torch.save(
                {
                    "feature": features,
                    "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
                    "sample_fps": float(sample_fps),
                },
                out_path,
            )


if __name__ == "__main__":
    main()
