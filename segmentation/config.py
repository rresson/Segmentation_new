"""集中配置：默认联合训练 MobileV3Backbone 和 MobileV3FPNHead。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .utils import resolve_path


@dataclass
class Config:
    # pretrain：CNN + SimpleHead。
    # head：冻结 CNN，只训练所选分割头。
    # custom：使用下面设置的训练范围。
    mode: Literal["pretrain", "head", "custom"] = "custom"

    data_root: str = "F:/Segmentation_Project/ADEChallengeData2016"
    output_root: str = "runs/ade20k_mobile_v3"
    output_dir: str = ""

    # 新架构默认从随机初始化开始。
    backbone_checkpoint: str = "runs/ade20k_mobile_v3/pretrain/backbone_best_miou.pth"
    checkpoint_prefix: str = ""
    resume: str = ""
    allow_random_backbone: bool = True

    backbone_name: Literal[
        "small_cnn",
        "demo",
        "mobile_v3",
    ] = "mobile_v3"

    head_name: Literal[
        "fpn",
        "simple",
        "mobile_fpn",
    ] = "mobile_fpn"

    # 控制 FPN 融合通道，必须为 8 的倍数。
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


def prepare_config(config: Config) -> Config:
    """生成实际配置，不修改调用者传入的对象。"""
    cfg = replace(config)

    if cfg.backbone_name not in (
        "small_cnn",
        "demo",
        "mobile_v3",
    ):
        raise ValueError("Unknown backbone_name")

    if cfg.head_name not in (
        "fpn",
        "simple",
        "mobile_fpn",
    ):
        raise ValueError("Unknown head_name")

    if cfg.mode not in ("pretrain", "head", "custom"):
        raise ValueError(
            "mode must be pretrain, head or custom"
        )

    if cfg.mode == "pretrain":
        cfg.head_name = "simple"
        cfg.trainable_prefixes = (
            "backbone",
            "decode_head",
        )
        cfg.allow_random_backbone = True

    elif cfg.mode == "head":
        cfg.trainable_prefixes = ("decode_head",)
        cfg.allow_random_backbone = False

        if not cfg.backbone_checkpoint:
            cfg.backbone_checkpoint = str(
                resolve_path(cfg.output_root)
                / "pretrain/backbone_best_miou.pth"
            )

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

    if cfg.head_width < 8 or cfg.head_width % 8:
        raise ValueError(
            "head_width must be a positive multiple of 8"
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