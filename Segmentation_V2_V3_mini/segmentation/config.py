"""Segmentation_V2_V3_mini：相同 V2 backbone + 单个 128→368→128 IR/SE C3 head。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Literal

from .utils import resolve_path


@dataclass
class Config:
    # pretrain：联合训练 backbone 与 mini head。
    # head：冻结 CNN，只训练所选分割头。
    # custom：使用下面设置的训练范围。
    mode: Literal["pretrain", "head", "custom"] = "pretrain"

    data_root: str = "ADEChallengeData2016"
    output_root: str = "runs/ade20k_v2_v3_mini"
    output_dir: str = ""
    run_name: str = ""  # 空值自动分配 runN；pipeline 根据 backbone 绑定实验。

    # 空路径：新实验，stem 默认 ImageNet 初始化，其他模块随机初始化。
    backbone_checkpoint: str = ""
    model_checkpoint: str = ""  # 初始化完整模型；单独训练 backbone 时保留已训练 head。
    checkpoint_prefix: str = ""
    resume: str = ""
    allow_random_backbone: bool = True

    backbone_name: Literal["seaformer"] = "seaformer"
    head_name: Literal["mobile_c3_mini"] = "mobile_c3_mini"
    context_expanded_channels: int = 368  # PW expansion；默认配置全网 997,178 Params。

    stem_pretrained: bool = True
    seaformer_attention: Literal["original", "multiscale", "lepe"] = "original"
    seaformer_depths: tuple[int, int, int] = (2, 2, 2)
    seaformer_learnable_squeeze: bool = True

    # C3 单路；context、lateral/refine 与 classifier 输入共用此宽度。
    head_width: int = 128

    trainable_prefixes: tuple[str, ...] = (
        "backbone",
        "decode_head",
    )

    num_classes: int = 150
    ignore_index: int = 255

    # 保留原工程的快速验证训练预算。
    epochs: int = 5
    batch_size: int = 4
    num_workers: int = 0

    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    min_lr: float = 1e-6

    crop_size: int = 256
    scale_range: tuple[float, float] = (0.5, 2.0)
    val_short_side: int = 256
    val_max_side: int = 512

    # 0 表示使用全部样本。
    max_train_samples: int = 1000
    max_val_samples: int = 200

    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)

    save_every: int = 5
    amp: bool = True
    seed: int = 42
    device: str = "auto"


def validate_model_config(cfg: Config) -> None:
    if cfg.backbone_name != "seaformer" or cfg.head_name != "mobile_c3_mini":
        raise ValueError("Mini requires backbone_name='seaformer' and head_name='mobile_c3_mini'")
    width = cfg.head_width
    if not isinstance(width, int) or isinstance(width, bool) or width < 8 or width % 8:
        raise ValueError("head_width must be a positive integer divisible by 8")
    expanded = cfg.context_expanded_channels
    if not isinstance(expanded, int) or isinstance(expanded, bool) or expanded <= width or expanded % 8:
        raise ValueError("context_expanded_channels must be an integer >head_width and divisible by 8")
    if cfg.seaformer_attention not in ("original", "multiscale", "lepe"):
        raise ValueError("Unknown seaformer_attention")
    if tuple(cfg.seaformer_depths) != (2, 2, 2):
        raise ValueError("seaformer_depths must be (2, 2, 2) to match segmentation_V2")


def prepare_config(config: Config) -> Config:
    """生成实际配置，不修改调用者传入的对象。"""
    cfg = replace(config)

    if cfg.run_name and re.fullmatch(r"run[1-9][0-9]*", cfg.run_name) is None:
        raise ValueError("run_name must be empty or run followed by a positive integer")
    if cfg.backbone_checkpoint and cfg.model_checkpoint:
        raise ValueError("Choose backbone_checkpoint OR model_checkpoint")

    validate_model_config(cfg)

    if cfg.mode not in ("pretrain", "head", "custom"):
        raise ValueError(
            "mode must be pretrain, head or custom"
        )

    if cfg.mode == "pretrain":
        cfg.trainable_prefixes = (
            "backbone",
            "decode_head",
        )
        cfg.allow_random_backbone = True

    elif cfg.mode == "head":
        cfg.trainable_prefixes = ("decode_head",)
        cfg.allow_random_backbone = False

    cfg.data_root = str(resolve_path(cfg.data_root))
    cfg.output_root = str(resolve_path(cfg.output_root))

    cfg.output_dir = str(
        resolve_path(cfg.output_dir)
        if cfg.output_dir
        else resolve_path(cfg.output_root) / cfg.mode
    )

    if cfg.backbone_checkpoint:
        cfg.backbone_checkpoint = str(
            resolve_path(cfg.backbone_checkpoint)
        )

    if cfg.resume:
        cfg.resume = str(resolve_path(cfg.resume))

    if cfg.model_checkpoint:
        cfg.model_checkpoint = str(resolve_path(cfg.model_checkpoint))

    if (
        cfg.epochs < 1
        or cfg.save_every < 1
        or cfg.batch_size < 1
    ):
        raise ValueError(
            "epochs、save_every、batch_size must be positive"
        )

    if (
        cfg.num_workers < 0
        or min(
            cfg.max_train_samples,
            cfg.max_val_samples,
        ) < 0
    ):
        raise ValueError(
            "num_workers and sample limits cannot be negative"
        )

    if (
        cfg.crop_size < 32
        or cfg.val_short_side < 32
        or cfg.val_max_side < cfg.val_short_side
    ):
        raise ValueError(
            "crop_size/val_short_side must be at least 32; "
            "val_max_side must be >= val_short_side"
        )

    if not 0 < cfg.scale_range[0] <= cfg.scale_range[1]:
        raise ValueError(
            "scale_range must satisfy 0 < minimum <= maximum"
        )

    if cfg.num_classes != 150 or cfg.ignore_index != 255:
        raise ValueError(
            "ADE20K requires num_classes=150 and ignore_index=255"
        )

    if (
        len(cfg.mean) != 3
        or len(cfg.std) != 3
        or any(value <= 0 for value in cfg.std)
    ):
        raise ValueError(
            "mean/std must have three values; std must be positive"
        )

    if (
        not 0 <= cfg.min_lr <= cfg.learning_rate
        or cfg.learning_rate <= 0
        or cfg.weight_decay < 0
    ):
        raise ValueError(
            "Invalid learning rate or weight_decay"
        )

    return cfg
