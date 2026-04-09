"""
Evaluate a UniTime + Gemma3 GTEA LoRA checkpoint.

Loads the trained adapter on top of the Gemma 3 4B base, runs the test set
through the same Gemma3DataCollator that training used, and reports:

  1. Per-instance eval loss (teacher-forced cross-entropy on the answer span)
  2. Average eval loss over the whole test set
  3. Argmax-decoded predictions for every test instance (saved to results.json)
  4. A few sample (gold vs predicted) tuples printed to stdout for sanity

Caveats:
- "Predictions" here are TEACHER-FORCED argmax over the answer span, not real
  autoregressive generation. This is a reasonable proxy for model quality
  (matches typical eval-loss conditions) but the real generation behavior may
  drift from this. To get true autoregressive output, use model.generate()
  with input_ids truncated at the start of the answer span — left for phase 2.
- mr_seg target is a list of timestamps, not interval pairs. We compare strings
  directly + give a "exact-match-per-timestamp-token" rate; full TAS-style
  F1@10/25/50 / Edit metrics need a separate post-processing pass.
- Pre-extracted features are required (same path as training).

Usage:
    conda activate UniTime
    cd .../experiments/unitime/UniTime
    python eval_gemma3.py \\
        --base_model      /nfs/hpc/share/zhanhaoc/MODLE/Gemma3-4B-it \\
        --adapter         ./checkpoints/gemma3_gtea_run1 \\
        --eval_data_path  /nfs/hpc/share/.../experiments/unitime/data/gtea/annot/test.json \\
        --feat_folder     /nfs/hpc/share/zhanhaoc/MODLE/Gemma3-4B-it/features/gtea \\
        --video_folder    /nfs/hpc/dgx2-4/data/TAS_videos/gtea \\
        --output          ./results/gemma3_gtea_run1/eval.json
"""
import os
# Same lesson as extract_gemma_features: CUDA_HOME + faulthandler + torch first.
os.environ.setdefault("CUDA_HOME", "/usr/local/apps/cuda/12.1")

import faulthandler
faulthandler.enable()

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoConfig

from models.gemma3_vl import Gemma3VLMRForConditionalGeneration, Gemma3VLMRProcessor
from collators.gemma3_vl import Gemma3DataCollator
from datasets_mr import VideoCentricDataset

PAD_IDX = -100


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, type=str,
                    help="Path to Gemma3 4B base model dir")
    ap.add_argument("--adapter", required=True, type=str,
                    help="Path to trained LoRA adapter dir (e.g. ./checkpoints/gemma3_gtea_run1)")
    ap.add_argument("--eval_data_path", required=True, type=str,
                    help="Path to test.json (mr_seg format)")
    ap.add_argument("--feat_folder", required=True, type=str,
                    help="Pre-extracted Gemma3 features dir")
    ap.add_argument("--video_folder", default=None, type=str,
                    help="Video folder (only used as fallback if feature path missing)")
    ap.add_argument("--output", required=True, type=str,
                    help="Output JSON path for predictions + metrics")
    ap.add_argument("--num_samples_to_print", default=10, type=int,
                    help="How many (gold, pred) pairs to print to stdout")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading base model from {args.base_model}...")
    base = Gemma3VLMRForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()

    print(f"loading LoRA adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(base, args.adapter).eval()

    print("loading processor + tokenizer...")
    processor = Gemma3VLMRProcessor.from_pretrained(args.base_model)
    tokenizer = processor.tokenizer
    config = AutoConfig.from_pretrained(args.base_model)
    base.tokenizer = tokenizer
    model.tokenizer = tokenizer

    print(f"loading test set from {args.eval_data_path}...")
    eval_ds = VideoCentricDataset(
        data_path=args.eval_data_path,
        video_folder=args.video_folder,
        feat_folder=args.feat_folder,
        fps=2,
        split="val",
        num_clips=32,
        clip_length=32,
        model_family_id="gemma3",
    )
    print(f"test set size: {len(eval_ds)} entries")

    collator = Gemma3DataCollator(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        mask_question_tokens=True,
    )

    losses = []
    predictions = []

    print()
    print("running eval...")
    with torch.no_grad():
        for i in range(len(eval_ds)):
            instance = eval_ds[i]
            batch = collator([instance])
            batch_gpu = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            try:
                out = model(**batch_gpu)
            except Exception as ex:
                print(f"  [{i:3d}] FAIL: {type(ex).__name__}: {ex}")
                predictions.append({
                    "qid": instance["qid"],
                    "id_in_dataset": i,
                    "error": str(ex),
                })
                continue

            loss_val = out.loss.item() if out.loss is not None else None
            if loss_val is not None:
                losses.append(loss_val)

            # Argmax over answer span (teacher-forced)
            labels = batch_gpu["labels"][0]
            input_ids = batch_gpu["input_ids"][0]
            answer_mask = labels != PAD_IDX
            # Logits at position t predict token t+1, so for the answer span at
            # positions [s, e), the predictions come from logits[s-1:e-1]. Shift.
            logits = out.logits[0]
            shifted_pred_ids = logits[:-1].argmax(-1)
            answer_mask_shifted = answer_mask[1:]
            pred_answer_ids = shifted_pred_ids[answer_mask_shifted]
            gt_answer_ids = labels[1:][answer_mask_shifted]

            pred_text = tokenizer.decode(pred_answer_ids, skip_special_tokens=True)
            gt_text = tokenizer.decode(gt_answer_ids, skip_special_tokens=True)

            # extract query for context
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

            # Extract vid from feature path (video field is None when --video_folder unset)
            feat_path = instance["message"][0]["content"][0].get("feature") or ""
            vid_str = os.path.splitext(os.path.basename(feat_path))[0] or "?"
            predictions.append({
                "qid": instance["qid"],
                "id_in_dataset": i,
                "video_id": vid_str,
                "query": query_text,
                "duration": instance["duration"],
                "loss": loss_val,
                "gold_text": gt_text,
                "pred_text": pred_text,
                "exact_match": pred_text.strip() == gt_text.strip(),
            })

            if (i + 1) % 10 == 0 or i == len(eval_ds) - 1:
                avg = sum(losses) / max(len(losses), 1)
                print(f"  [{i+1:3d}/{len(eval_ds)}] running avg loss = {avg:.4f}")

    avg_loss = sum(losses) / max(len(losses), 1)
    exact_matches = sum(1 for p in predictions if p.get("exact_match"))
    n_ok = sum(1 for p in predictions if "error" not in p)

    print()
    print("=" * 60)
    print(f"eval summary")
    print(f"  total entries:       {len(eval_ds)}")
    print(f"  successful forwards: {n_ok}")
    print(f"  avg eval loss:       {avg_loss:.4f}")
    print(f"  exact-match (text):  {exact_matches}/{n_ok} ({100*exact_matches/max(n_ok,1):.1f}%)")
    print("=" * 60)
    print()

    # Save JSON FIRST so a print bug can't lose the data
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "summary": {
                "total": len(eval_ds),
                "n_ok": n_ok,
                "avg_loss": avg_loss,
                "exact_match_rate": exact_matches / max(n_ok, 1),
            },
            "predictions": predictions,
        }, f, indent=2, default=str)
    print(f"saved {args.output}")
    print()

    print(f"sample predictions (first {args.num_samples_to_print}):")
    for p in predictions[: args.num_samples_to_print]:
        if "error" in p:
            print(f"  qid={p['qid']} ERROR: {p['error']}")
            continue
        loss_str = f"{p['loss']:.3f}" if p.get('loss') is not None else "?"
        print(f"  qid={p['qid']:3d}  video={p.get('video_id','?')}  query='{p['query']}'  loss={loss_str}")
        print(f"    gold: {p['gold_text']}")
        print(f"    pred: {p['pred_text']}")
        print()


if __name__ == "__main__":
    main()
