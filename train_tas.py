"""TAS 滑窗训练入口.

与 UniTime 原版 train.py 的区别:
  - Dataset: GTEAWindowDataset (随机 30s 窗口, 逐帧 GT)
  - Collator: Gemma4TASDataCollator (在线 pixel values, 交错 timestamp-image)
  - Model: Gemma4VLMRForConditionalGeneration + set_spatial_pool()
  - 训练方式 (loss, LoRA, Trainer) 与 UniTime 一致
"""
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml
import torch
import transformers
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from collators import COLLATORS
from datasets_tas import GTEAWindowDataset
from loaders import LOADERS
from supported_models import MODULE_KEYWORDS
from utils import rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer, TrainerWithCustomSampler


@dataclass
class ModelArguments:
    model_id: str = field(default="gemma4-e4b-it")
    model_local_path: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    video_folder: str = field(default=None)        # 视频目录
    gt_folder: str = field(default=None)            # groundTruth/*.txt 逐帧标注
    feat_folder: Optional[str] = field(default=None)  # 预提取 feature 目录 (.pt)
    train_split: str = field(default=None)          # split bundle 文件
    test_split: Optional[str] = field(default=None)

@dataclass
class TASArguments:
    spatial_pool_h: int = field(default=0)  # 0 表示不做 spatial pooling
    spatial_pool_w: int = field(default=0)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=16384)
    use_flash_attn: bool = field(default=False)
    train_vision_encoder: bool = field(default=False)
    train_vision_projector: bool = field(default=False)
    mask_question_tokens: bool = field(default=True)

    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False

@dataclass
class LoraArguments:
    use_lora: bool = field(default=True)
    use_vision_lora: bool = field(default=False)
    q_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=8)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = "none"


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TASArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, tas_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # 保存所有参数到 output_dir/arguments/
    output_dir = training_args.output_dir
    args_dir = Path(output_dir) / "arguments"
    args_dir.mkdir(parents=True, exist_ok=True)
    yaml.dump(asdict(model_args), open(args_dir / "model.yaml", "w"))
    yaml.dump(asdict(data_args), open(args_dir / "data.yaml", "w"))
    yaml.dump(asdict(tas_args), open(args_dir / "tas.yaml", "w"))
    yaml.dump(asdict(training_args), open(args_dir / "training.yaml", "w"))
    yaml.dump(asdict(lora_args), open(args_dir / "lora.yaml", "w"))

    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    model_family_id = "gemma4"

    # 加载模型
    rank0_print("Loading model...")
    loader = LOADERS[model_family_id](
        model_hf_path=model_args.model_id,
        model_local_path=model_args.model_local_path,
        compute_dtype=compute_dtype,
        bnb_config=None,
        use_flash_attn=training_args.use_flash_attn,
        device_map=None,
    )
    model, tokenizer, processor, config = loader.load()
    tokenizer.model_max_length = training_args.model_max_length

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # spatial pooling grid (稍后在 peft 包装之后再 set)
    spatial_pool_grid = None
    if tas_args.spatial_pool_h > 0 and tas_args.spatial_pool_w > 0:
        spatial_pool_grid = (tas_args.spatial_pool_h, tas_args.spatial_pool_w)

    # 冻结 vision encoder / projector
    vision_encoder_keys = MODULE_KEYWORDS[model_family_id]["vision_encoder"]
    if not training_args.train_vision_encoder:
        rank0_print("Freezing vision encoder:")
        for module in vision_encoder_keys:
            rank0_print(f"  {module}")
            eval(f"model.{module}").requires_grad_(False)

    vision_projector_keys = MODULE_KEYWORDS[model_family_id]["vision_projector"]
    if not training_args.train_vision_projector:
        rank0_print("Freezing vision projector:")
        for module in vision_projector_keys:
            rank0_print(f"  {module}")
            eval(f"model.{module}").requires_grad_(False)

    if "others" in MODULE_KEYWORDS[model_family_id]:
        for other_key in MODULE_KEYWORDS[model_family_id]["others"]:
            eval(f"model.{other_key}").requires_grad_(False)

    # LoRA (仅 LLM)
    llm_keys = MODULE_KEYWORDS[model_family_id]["llm"]
    if lora_args.use_lora:
        rank0_print("LoRA for LLM enabled")
        named_modules = {n: m for n, m in model.named_modules()}
        lora_modules = find_all_linear_names(named_modules, llm_keys)
        full_modules = []
        if training_args.train_vision_projector:
            full_modules.extend(vision_projector_keys)

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_modules,
            modules_to_save=full_modules if full_modules else None,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # 设置 spatial pooling (必须在 peft 包装之后, 否则 monkey-patch 会丢失)
    if spatial_pool_grid is not None:
        base = model.base_model.model if hasattr(model, "base_model") else model
        base.set_spatial_pool(spatial_pool_grid)
        rank0_print(f"Spatial pooling: {spatial_pool_grid}")

    # 加载数据
    rank0_print("Loading data...")
    train_dataset = GTEAWindowDataset(
        video_folder=data_args.video_folder,
        gt_folder=data_args.gt_folder,
        split_file=data_args.train_split,
        feat_folder=data_args.feat_folder,
        split="train",
    )
    rank0_print(f"Train: {len(train_dataset)} videos")

    val_dataset = None
    if data_args.test_split:
        val_dataset = GTEAWindowDataset(
            video_folder=data_args.video_folder,
            gt_folder=data_args.gt_folder,
            split_file=data_args.test_split,
            feat_folder=data_args.feat_folder,
            split="val",
        )
        rank0_print(f"Val: {len(val_dataset)} videos")
    else:
        training_args.eval_strategy = "no"

    # Collator
    data_collator = COLLATORS["gemma4_tas"](
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        mask_question_tokens=training_args.mask_question_tokens,
        spatial_pool_grid=spatial_pool_grid,
    )

    # Trainer (和 UniTime 一致: HF Trainer + DeepSpeed ZeRO-2)
    training_args.label_names = "labels"
    trainer = TrainerWithCustomSampler(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)


if __name__ == "__main__":
    train()
