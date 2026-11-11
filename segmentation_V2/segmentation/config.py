"""上传 MobileNetV2-stem 网络的配置；不依赖 MobileV3 网络或分割头。"""
from dataclasses import dataclass, replace
import re
from typing import Literal
from .utils import resolve_path


@dataclass
class Config:
    mode: Literal["pretrain", "head", "custom"] = "pretrain"
    data_root: str = "F:/Segmentation_Project/ADEChallengeData2016"
    output_root: str = "runs/ade20k"
    output_dir: str = ""
    run_name: str = ""            # 如 run3；空值：pretrain 新建，custom/head 选择最近的 pretrain run
    backbone_checkpoint: str = "auto"  # custom/head 自动读取同一 run/pretrain 的 backbone 权重
    model_checkpoint: str = ""     # 初始化完整模型，支持上传网络的 model_state_dict
    checkpoint_prefix: str = ""
    resume: str = ""               # 仅用于本训练器保存的完整训练状态
    allow_random_backbone: bool = True
    backbone_name: str = "uploaded_mobile_v2"
    head_name: Literal["original", "multilevel"] = "multilevel"
    segment_type = "original"
    head_width: int = 128          # 对应 main_network.hidden_channel
    stem_pretrained: bool = True
    attention_type: Literal["original", "multiscale", "lepe"] = "original"
    fusion_type: Literal["original", "enhanced"] = "enhanced"
    dim: int = 64                 # 原始 stem 输出 64，不改变网络结构
    key_dim: int = 16
    num_heads: int = 4
    attn_ratio: float = 2.0
    pos_shape: int = 16
    use_learnable_squeeze: bool = True
    ffn_ratio: float = 2.0
    expansion_ratio1: float = 2.0
    expansion_ratio2: float = 2.0
    use_aux: bool = False          # 原网络可选辅助头；False 时只有主头
    aux_loss_weight: float = 0.4
    trainable_prefixes: tuple[str, ...] = ("backbone", "decode_head")
    num_classes: int = 150
    ignore_index: int = 255
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
    max_train_samples: int = 1000
    max_val_samples: int = 200
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    save_every: int = 5
    amp: bool = True
    seed: int = 42
    device: str = "auto"

    def __post_init__(self):
        # auto 是配置标记，不是真实文件路径；实际权重在训练启动时解析。
        if self.backbone_checkpoint == "auto":
            self.backbone_checkpoint = ""


def network_kwargs(cfg: Config) -> dict:
    """与上传 main_network 的参数一一对应，用于构造和权重兼容检查。"""
    return dict(dim=cfg.dim, key_dim=cfg.key_dim, num_heads=cfg.num_heads,
                attn_ratio=cfg.attn_ratio, pos_shape=cfg.pos_shape,
                use_learnable_squeeze=cfg.use_learnable_squeeze,
                ffn_ratio=cfg.ffn_ratio, hidden_channel=cfg.head_width,
                num_class=cfg.num_classes, expansion_ratio1=cfg.expansion_ratio1,
                expansion_ratio2=cfg.expansion_ratio2, use_aux=cfg.use_aux,
                attention_type=cfg.attention_type, fusion_type=cfg.fusion_type,
                segment_type=cfg.head_name)


def prepare_config(config: Config) -> Config:
    cfg = replace(config)
    # auto 在训练启动时解析；pretrain 和独立结构统计不读取预训练实验。
    if cfg.backbone_checkpoint == "auto":
        cfg.backbone_checkpoint = ""
    if cfg.run_name and re.fullmatch(r"run[1-9][0-9]*", cfg.run_name) is None:
        raise ValueError("run_name must be empty or run followed by a positive integer, e.g. run3")
    if cfg.backbone_name != "uploaded_mobile_v2":
        raise ValueError("This adapter requires backbone_name='uploaded_mobile_v2'")
    if cfg.mode not in ("pretrain", "head", "custom"):
        raise ValueError("mode must be pretrain, head or custom")
    if cfg.attention_type not in ("original", "multiscale", "lepe"):
        raise ValueError("Unknown attention_type")
    if cfg.fusion_type not in ("original", "enhanced"):
        raise ValueError("Unknown fusion_type")
    if cfg.head_name not in ("original", "multilevel"):
        raise ValueError("head_name must be original or multilevel")
    if cfg.dim != 64:
        raise ValueError("Original MobileNetV2 stem outputs 64 channels; dim must be 64")
    if min(cfg.key_dim, cfg.num_heads, cfg.pos_shape, cfg.head_width) < 1:
        raise ValueError("Channel dimensions and pos_shape must be positive")
    if (cfg.attn_ratio <= 0 or cfg.ffn_ratio <= 0
            or cfg.expansion_ratio1 <= 0 or cfg.expansion_ratio2 <= 0
            or min(int(cfg.attn_ratio * cfg.key_dim), int(cfg.ffn_ratio * cfg.dim),
                   int(cfg.expansion_ratio1 * cfg.dim), int(cfg.expansion_ratio2 * cfg.dim)) < 1):
        raise ValueError("Expansion ratios must produce positive channels")
    if cfg.use_aux and cfg.aux_loss_weight <= 0:
        raise ValueError("use_aux=True requires aux_loss_weight > 0")
    if cfg.backbone_checkpoint and cfg.model_checkpoint:
        raise ValueError("Choose backbone_checkpoint OR model_checkpoint")
    if cfg.mode == "pretrain":
        # 沿用上传网络选定的主/辅助分割头，不换成 SimpleHead。
        cfg.trainable_prefixes = ("backbone", "decode_head")
        cfg.allow_random_backbone = True
    elif cfg.mode == "head":
        cfg.trainable_prefixes = ("decode_head",)
        cfg.allow_random_backbone = False
    cfg.data_root = str(resolve_path(cfg.data_root))
    cfg.output_root = str(resolve_path(cfg.output_root))
    cfg.output_dir = str(resolve_path(cfg.output_dir) if cfg.output_dir
                         else resolve_path(cfg.output_root) / cfg.mode)
    for key in ("backbone_checkpoint", "model_checkpoint", "resume"):
        if getattr(cfg, key):
            setattr(cfg, key, str(resolve_path(getattr(cfg, key))))
    if min(cfg.epochs, cfg.save_every, cfg.batch_size) < 1:
        raise ValueError("epochs/save_every/batch_size must be positive")
    if min(cfg.num_workers, cfg.max_train_samples, cfg.max_val_samples) < 0:
        raise ValueError("num_workers and sample limits cannot be negative")
    if min(cfg.crop_size, cfg.val_short_side) < 32 or cfg.val_max_side < cfg.val_short_side:
        raise ValueError("Invalid crop/validation sizes")
    if not 0 < cfg.scale_range[0] <= cfg.scale_range[1]:
        raise ValueError("Invalid scale_range")
    if cfg.num_classes != 150 or cfg.ignore_index != 255:
        raise ValueError("ADE20K requires num_classes=150 and ignore_index=255")
    if len(cfg.mean) != 3 or len(cfg.std) != 3 or any(x <= 0 for x in cfg.std):
        raise ValueError("Invalid mean/std")
    if not 0 <= cfg.min_lr <= cfg.learning_rate or cfg.learning_rate <= 0 or cfg.weight_decay < 0:
        raise ValueError("Invalid learning rate/weight_decay")
    return cfg
