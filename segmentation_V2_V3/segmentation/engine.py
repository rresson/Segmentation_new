"""单个 epoch 的训练与验证，保留上传脚本的损失和原图标签指标计算。"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader
from .config import Config
from .freezing import set_training_mode
from .metrics import SegmentationMetrics


def resize_logits(logits: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
        raise ValueError(
            "分割头需输出 [B,C,H,W] logits；dict/tuple 输出请先在模型中适配。"
        )
    if logits.shape[1] != num_classes:
        raise ValueError("输出通道数必须等于类别数。")
    # 先插值 logits 再 argmax，而非插值离散类别。
    return functional.interpolate(
        logits, labels.shape[-2:], mode="bilinear", align_corners=False
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: Config,
    device: torch.device,
) -> float:
    set_training_mode(model, cfg.trainable_prefixes)
    loss_sum, pixel_count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        count = int((labels != cfg.ignore_index).sum())
        if count == 0:  # 全忽略裁剪不可计算 mean CE，否则会产生 NaN。
            continue
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, enabled=cfg.amp and device.type == "cuda"
        ):
            logits = resize_logits(model(images), labels, cfg.num_classes)
            loss = functional.cross_entropy(
                logits, labels, ignore_index=cfg.ignore_index
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("训练 loss 为 NaN/Inf，请检查输入或学习率。")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.detach().item() * count
        pixel_count += count
    if pixel_count == 0:
        raise ValueError("本 epoch 无有效训练像素。")
    return loss_sum / pixel_count


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    metric = SegmentationMetrics(cfg.num_classes, cfg.ignore_index, device)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(
            device_type=device.type, enabled=cfg.amp and device.type == "cuda"
        ):
            output = model(images)
        # 验证标签保持原图尺寸；上采样 logits 到标签尺寸再计算指标。
        logits = resize_logits(output.float(), labels, cfg.num_classes)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("验证 logits 包含 NaN/Inf。")
        metric.update(logits.argmax(1), labels)
    return metric.compute()
