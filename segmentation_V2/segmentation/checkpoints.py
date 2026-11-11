"""新训练器状态和上传网络 model_state_dict 的显式兼容适配。"""
from dataclasses import fields
from pathlib import Path
import torch
from .config import Config, network_kwargs
from .models.segmentor import BACKBONE_MODULES

ARCHITECTURE = "UploadedMobileV2SeaFormer_v1"
BACKBONE_OPTIONS = ("dim", "key_dim", "num_heads", "attn_ratio", "pos_shape",
                    "use_learnable_squeeze", "ffn_ratio", "expansion_ratio1",
                    "expansion_ratio2", "attention_type", "fusion_type")


def atomic_save(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def export_backbone(model, cfg, path):
    atomic_save(dict(state_dict={k: v.detach().cpu() for k, v in model.backbone.state_dict().items()},
                     architecture=ARCHITECTURE, out_channels=model.backbone.out_channels,
                     network_config=network_kwargs(cfg), mean=cfg.mean, std=cfg.std), path)


def checkpoint_state(checkpoint):
    state = checkpoint
    for key in ("model_state_dict", "model", "state_dict"):
        if key in checkpoint:
            state = checkpoint[key]
            break
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a nonempty state_dict or model/model_state_dict/state_dict container")
    return {k.removeprefix("module."): v for k, v in state.items()}


def adapt_full_state(state, use_aux):
    """原参数名 → 适配器参数名；张量不裁剪、不拼接，最终严格加载。"""
    converted = {}
    for name, value in state.items():
        first = name.split(".", 1)[0]
        if first in BACKBONE_MODULES:
            name = "backbone." + name
        elif first in ("segment_head", "auxiliary_head"):
            name = "decode_head." + name
        if not use_aux and name.startswith("decode_head.auxiliary_head."):
            continue  # 用户明确关闭辅助分支时允许舍弃旧辅助头
        converted[name] = value
    return converted


def validate_metadata(checkpoint, cfg, full_model=False):
    if "architecture" in checkpoint and checkpoint["architecture"] != ARCHITECTURE:
        raise ValueError("Checkpoint belongs to a different backbone architecture")
    if "config" in checkpoint:
        previous = checkpoint["config"]
        if previous.get("backbone_name") != cfg.backbone_name:
            raise ValueError("Checkpoint backbone_name differs from uploaded MobileV2")
        network = network_kwargs(Config(**previous))
        normalization = previous
    else:
        network = checkpoint.get("network_config", checkpoint.get("model_config", {})).copy()
        normalization = checkpoint
        if "model_config" in checkpoint:
            # 这些选项在上传 main_network 中有默认值。
            for key, value in dict(attention_type="original", fusion_type="original",
                                   segment_type="original", use_aux=True).items():
                network.setdefault(key, value)
        if "num_classes" in checkpoint:
            network["num_class"] = checkpoint["num_classes"]
    wanted = network_kwargs(cfg)
    options = BACKBONE_OPTIONS + (("hidden_channel", "num_class", "segment_type") if full_model else ())
    for key in options:
        if key in network and network[key] != wanted[key]:
            raise ValueError(f"Checkpoint {key}={network[key]!r} differs from Config {wanted[key]!r}")
    for key in ("mean", "std"):
        if key in normalization and tuple(normalization[key]) != getattr(cfg, key):
            raise ValueError(f"Checkpoint {key} differs from input normalization")


def load_full_model(model, checkpoint, cfg):
    validate_metadata(checkpoint, cfg, full_model=True)
    state = adapt_full_state(checkpoint_state(checkpoint), cfg.use_aux)
    model.load_state_dict(state, strict=True)


def load_backbone(model, cfg):
    """保留 trainer.py 的函数接口；model_checkpoint 优先初始化完整模型。"""
    if cfg.model_checkpoint:
        checkpoint = torch.load(cfg.model_checkpoint, map_location="cpu", weights_only=True)
        load_full_model(model, checkpoint, cfg)
        print(f"Full model initialized: {cfg.model_checkpoint}", flush=True)
        return
    if not cfg.backbone_checkpoint:
        if not cfg.allow_random_backbone:
            raise ValueError("A trained backbone or full model checkpoint is required")
        print("New experiment: ImageNet stem; remaining modules random."
              if cfg.stem_pretrained else "New experiment: random initialization.", flush=True)
        return
    checkpoint = torch.load(cfg.backbone_checkpoint, map_location="cpu", weights_only=True)
    validate_metadata(checkpoint, cfg)
    state = checkpoint_state(checkpoint)
    if cfg.checkpoint_prefix:
        state = {k[len(cfg.checkpoint_prefix):]: v for k, v in state.items()
                 if k.startswith(cfg.checkpoint_prefix)}
    elif any(k.startswith("backbone.") for k in state):
        state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
    else:
        # 独立 backbone 和原 main_network 使用相同的 stem/context/fusion 前缀。
        state = {k: v for k, v in state.items() if k.split(".", 1)[0] in BACKBONE_MODULES}
    model.backbone.load_state_dict(state, strict=True)
    print(f"Backbone initialized: {cfg.backbone_checkpoint}", flush=True)


def check_resume_config(checkpoint, cfg, trainable_names):
    required = {"model", "optimizer", "scheduler", "scaler", "epoch", "config",
                "trainable_names", "best_miou", "best_accuracy"}
    if not required.issubset(checkpoint):
        raise ValueError("resume requires this trainer's complete checkpoint; use model_checkpoint for uploaded weights")
    if checkpoint["trainable_names"] != trainable_names:
        raise ValueError("Training parameter range changed; start a new experiment")
    # 初始化路径和输出位置不影响已恢复的权重；其余实验设置保持一致。
    initialization = {"output_root", "output_dir", "backbone_checkpoint", "model_checkpoint",
                      "checkpoint_prefix", "resume", "stem_pretrained", "allow_random_backbone",
                      "save_every", "device", "trainable_prefixes", "run_name"}
    previous = checkpoint["config"]
    for field in fields(Config):
        if field.name not in initialization and previous.get(field.name) != getattr(cfg, field.name):
            raise ValueError(f"Resume config {field.name} differs from checkpoint")
