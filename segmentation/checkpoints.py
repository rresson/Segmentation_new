"""权重加载、独立 CNN 导出及断点续训兼容检查。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import Config
from .models.segmentor import Segmentor


def atomic_save(
    payload: dict[str, Any],
    path: Path,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def architecture_name(cfg: Config) -> str:
    return {
        "small_cnn": "SmallCNNBackbone_v1",
        "demo": "DemoBackbone_v1",
        "mobile_v3": "MobileV3Backbone_v1",
    }[cfg.backbone_name]


def export_backbone(
    model: Segmentor,
    cfg: Config,
    path: Path,
) -> None:
    atomic_save(
        {
            "state_dict": {
                name: value.detach().cpu()
                for name, value
                in model.backbone.state_dict().items()
            },
            "architecture": architecture_name(cfg),
            "out_channels": (32, 64, 128),
            "mean": cfg.mean,
            "std": cfg.std,
        },
        path,
    )


def load_backbone(
    model: Segmentor,
    cfg: Config,
) -> None:
    if not cfg.backbone_checkpoint:
        if not cfg.allow_random_backbone:
            raise ValueError(
                "Please set CNN weights or run mode='pretrain'."
            )

        print(
            "CNN weights randomly initialized.",
            flush=True,
        )
        return

    path = Path(cfg.backbone_checkpoint)

    if not path.is_file():
        raise FileNotFoundError(
            f"CNN weights not found: {path}"
        )

    checkpoint: dict[str, Any] = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if (
        "architecture" in checkpoint
        and checkpoint["architecture"] != architecture_name(cfg)
    ):
        raise ValueError(
            "CNN checkpoint architecture does not match "
            "the current backbone."
        )

    # 完整 checkpoint 的配置位于 config 内；
    # 独立 CNN 导出文件的配置位于顶层。
    metadata = checkpoint.get("config", checkpoint)

    if (
        "backbone_name" in metadata
        and metadata["backbone_name"] != cfg.backbone_name
    ):
        raise ValueError(
            "checkpoint backbone_name does not match current model"
        )

    for key in ("mean", "std"):
        if (
            key in metadata
            and tuple(metadata[key]) != getattr(cfg, key)
        ):
            raise ValueError(
                f"{key} differs from input normalization"
            )

    state = checkpoint.get(
        "state_dict",
        checkpoint.get("model", checkpoint),
    )

    state = {
        name.removeprefix("module."): value
        for name, value in state.items()
    }

    prefix = cfg.checkpoint_prefix

    if prefix:
        state = {
            name[len(prefix):]: value
            for name, value in state.items()
            if name.startswith(prefix)
        }

    if not state:
        raise ValueError(
            "No CNN parameters found; check checkpoint_prefix."
        )

    # 不忽略缺失或多余的参数。
    model.backbone.load_state_dict(
        state,
        strict=True,
    )

    print(
        f"CNN weights loaded: {path}",
        flush=True,
    )


def check_resume_config(
    checkpoint: dict[str, Any],
    cfg: Config,
    trainable_names: list[str],
) -> None:
    if checkpoint["trainable_names"] != trainable_names:
        raise ValueError(
            "The training parameter range changed. "
            "Please start a new experiment."
        )

    previous = checkpoint["config"]

    keys = (
        "epochs",
        "num_classes",
        "mean",
        "std",
        "crop_size",
        "scale_range",
        "val_short_side",
        "val_max_side",
        "batch_size",
        "amp",
        "seed",
        "ignore_index",
        "num_workers",
        "learning_rate",
        "weight_decay",
        "min_lr",
        "backbone_name",
        "head_name",
        "head_width",
        "max_train_samples",
        "max_val_samples",
    )

    legacy_defaults = {
        "backbone_name": "demo",
        "head_name": "fpn",
        "head_width": 128,
        "max_train_samples": 0,
        "max_val_samples": 0,
    }

    for key in keys:
        previous_value = previous.get(
            key,
            legacy_defaults.get(key),
        )

        if previous_value != getattr(cfg, key):
            raise ValueError(
                f"Resume config {key} differs from checkpoint"
            )

    if (
        checkpoint.get("format_version") == 2
        and previous["data_root"] != cfg.data_root
    ):
        raise ValueError(
            "data_root cannot be changed during resume"
        )