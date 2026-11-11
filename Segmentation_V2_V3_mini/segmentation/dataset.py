"""ADE20K 原始标签映射与同步增强；保留上传脚本的原图标签验证方式。"""

from __future__ import annotations

from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import Tensor
from torch.utils.data import Dataset
from .config import Config


def encode_mask(raw: np.ndarray, cfg: Config) -> np.ndarray:
    """输入为 ADEChallengeData2016 原始单通道标注：0 忽略，1..150 为类别。"""
    raw = raw.astype(np.int64)
    if raw.ndim != 2:
        raise ValueError("需要单通道类别索引 PNG，不能使用 RGB 可视化标签。")
    valid = (raw >= 1) & (raw <= cfg.num_classes)
    if not np.all(valid | (raw == 0) | (raw == cfg.ignore_index)):
        raise ValueError("标签不符合原始 ADE20K 的 0..150/255 编码。")
    label = np.full(raw.shape, cfg.ignore_index, dtype=np.int64)
    label[valid] = raw[valid] - 1
    return label


class ADE20KDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, cfg: Config, split: str) -> None:
        self.cfg = cfg
        if split not in ("training", "validation"):
            raise ValueError(split)
        self.training = split == "training"
        root = Path(cfg.data_root)
        self.images = sorted((root / "images" / split).glob("*.jpg"))
        # 固定随机子集；0 表示全部。两个阶段相同种子会选择相同图片。
        limit = cfg.max_train_samples if self.training else cfg.max_val_samples
        if limit > 0 and len(self.images) > limit:
            rng = random.Random(cfg.seed + (0 if self.training else 10000))
            indices = sorted(rng.sample(range(len(self.images)), limit))
            self.images = [self.images[index] for index in indices]
        self.masks = [
            root / "annotations" / split / f"{p.stem}.png" for p in self.images
        ]
        if not self.images:
            raise FileNotFoundError(f"没有找到图片：{root / 'images' / split}")
        missing = [str(p) for p in self.masks if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"缺失标注，例如：{missing[0]}")
        self.mean = torch.tensor(cfg.mean).view(3, 1, 1)
        self.std = torch.tensor(cfg.std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        cfg = self.cfg
        with Image.open(self.images[index]) as src:
            image = src.convert("RGB")
        with Image.open(self.masks[index]) as src:
            mask = src.copy()  # 保留索引值；不要 convert('RGB')。
        if image.size != mask.size:
            raise ValueError(f"图片和标注尺寸不一致：{self.images[index]}")
        width, height = image.size
        if self.training:
            # 图像和标签使用完全相同的缩放、裁剪与翻转参数。
            scale = (
                cfg.crop_size * random.uniform(*cfg.scale_range) / min(width, height)
            )
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = image.resize(size, Image.Resampling.BILINEAR)
            mask = mask.resize(size, Image.Resampling.NEAREST)
            pad_w, pad_h = max(0, cfg.crop_size - size[0]), max(
                0, cfg.crop_size - size[1]
            )
            image = ImageOps.expand(
                image,
                (0, 0, pad_w, pad_h),
                fill=tuple(round(255 * x) for x in cfg.mean),
            )
            mask = ImageOps.expand(mask, (0, 0, pad_w, pad_h), fill=0)
            left = random.randint(0, image.width - cfg.crop_size)
            top = random.randint(0, image.height - cfg.crop_size)
            box = (left, top, left + cfg.crop_size, top + cfg.crop_size)
            image, mask = image.crop(box), mask.crop(box)
            if random.random() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
        else:
            # 验证仅对输入图像作确定性等比例缩放；标签保留原始分辨率。
            scale = min(
                cfg.val_short_side / min(width, height),
                cfg.val_max_side / max(width, height),
            )
            image = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.BILINEAR,
            )
        array = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        tensor = (tensor - self.mean) / self.std
        label = torch.from_numpy(encode_mask(np.array(mask), cfg))
        return tensor, label
