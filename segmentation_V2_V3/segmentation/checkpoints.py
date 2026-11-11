"""权重加载、独立 CNN 导出及断点续训兼容检查。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import Config
from .models.segmentor import Segmentor
from .models.seaformer import backbone_network_config


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
        "seaformer": "UploadedMobileV2SeaFormer_v1",
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
            "out_channels": tuple(model.backbone.out_channels),
            **({
                "network_config": backbone_network_config(cfg.seaformer_attention, cfg.seaformer_learnable_squeeze),
                "seaformer_attention": cfg.seaformer_attention,
                "seaformer_depths": cfg.seaformer_depths,
                "seaformer_learnable_squeeze": cfg.seaformer_learnable_squeeze,
            } if cfg.backbone_name == "seaformer" else {}),
            "mean": cfg.mean,
            "std": cfg.std,
        },
        path,
    )


def checkpoint_state(checkpoint):
    state = checkpoint
    for key in ("model_state_dict", "model", "state_dict"):
        if key in checkpoint:
            state = checkpoint[key]
            break
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a nonempty state_dict or model/model_state_dict/state_dict container")
    return {name.removeprefix("module."): value for name, value in state.items()}


def validate_metadata(checkpoint, cfg, *, full_model=False):
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

    if cfg.backbone_name == "seaformer":
        for key in ("seaformer_attention", "seaformer_depths", "seaformer_learnable_squeeze"):
            if key in metadata and metadata[key] != getattr(cfg, key):
                raise ValueError(f"CNN checkpoint {key} differs from current model")

    if (
        "backbone_name" in metadata
        and metadata["backbone_name"] not in (
            (cfg.backbone_name, "uploaded_mobile_v2") if cfg.backbone_name == "seaformer"
            else (cfg.backbone_name,)
        )
    ):
        raise ValueError(
            "checkpoint backbone_name does not match current model"
        )

    if cfg.backbone_name == "seaformer":
        network = checkpoint.get("network_config", checkpoint.get("model_config", {}))
        if metadata.get("backbone_name") == "uploaded_mobile_v2":
            network = metadata
        expected = backbone_network_config(cfg.seaformer_attention, cfg.seaformer_learnable_squeeze)
        for key in ("dim", "key_dim", "num_heads", "attn_ratio", "pos_shape",
                    "use_learnable_squeeze", "ffn_ratio", "expansion_ratio1", "expansion_ratio2",
                    "attention_type", "fusion_type"):
            if key in network and network[key] != expected[key]:
                raise ValueError(f"CNN checkpoint {key} differs from current model")

    for key in ("mean", "std"):
        if (
            key in metadata
            and tuple(metadata[key]) != getattr(cfg, key)
        ):
            raise ValueError(
                f"{key} differs from input normalization"
            )

    if full_model:
        for key in ("backbone_name", "head_name", "head_width", "num_classes"):
            if key in metadata and metadata[key] != getattr(cfg, key):
                raise ValueError(f"Full checkpoint {key} differs from current model")


def load_full_model(model, checkpoint, cfg):
    validate_metadata(checkpoint, cfg, full_model=True)
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)


def load_backbone(
    model: Segmentor,
    cfg: Config,
) -> None:
    if cfg.model_checkpoint:
        checkpoint = torch.load(cfg.model_checkpoint, map_location="cpu", weights_only=True)
        load_full_model(model, checkpoint, cfg)
        print(f"Full model initialized: {cfg.model_checkpoint}", flush=True)
        return
    if not cfg.backbone_checkpoint:
        if not cfg.allow_random_backbone:
            raise ValueError(
                "Please set CNN weights or run mode='pretrain'."
            )

        print(
            ("SeaFormer: ImageNet stem; context/fusion/head require training."
             if cfg.backbone_name == "seaformer" and cfg.stem_pretrained
             else "CNN weights randomly initialized."),
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

    validate_metadata(checkpoint, cfg)

    state = checkpoint_state(checkpoint)

    prefix = cfg.checkpoint_prefix

    if prefix:
        state = {
            name[len(prefix):]: value
            for name, value in state.items()
            if name.startswith(prefix)
        }
    elif any(name.startswith("backbone.") for name in state):
        state = {name.removeprefix("backbone."): value for name, value in state.items()
                 if name.startswith("backbone.")}

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

    if cfg.backbone_name == "seaformer":
        keys += ("seaformer_attention", "seaformer_depths", "seaformer_learnable_squeeze")

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
