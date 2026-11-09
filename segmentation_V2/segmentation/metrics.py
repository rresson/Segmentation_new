"""累计验证集混淆矩阵，计算 mIoU 与像素 accuracy。"""

from __future__ import annotations

import torch
from torch import Tensor


class SegmentationMetrics:
    def __init__(
        self, num_classes: int, ignore_index: int, device: torch.device
    ) -> None:
        self.n, self.ignore = num_classes, ignore_index
        self.matrix = torch.zeros((self.n, self.n), dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, prediction: Tensor, target: Tensor) -> None:
        # 行是真值、列是预测；忽略像素既不计入分子，也不计入分母。
        valid = target != self.ignore
        target, prediction = target[valid], prediction[valid]
        index = target * self.n + prediction
        self.matrix += torch.bincount(index, minlength=self.n**2).reshape(
            self.n, self.n
        )

    def compute(self) -> tuple[float, float]:
        matrix = self.matrix.double()
        if matrix.sum() == 0:
            raise ValueError("验证集没有有效标注像素。")
        intersection = matrix.diag()
        union = matrix.sum(0) + matrix.sum(1) - intersection
        # 真值/预测均不存在的类别 union=0，排除在均值外；误预测出的类别计为 0 IoU。
        present = union > 0
        miou = (intersection[present] / union[present]).mean().item()
        accuracy = (intersection.sum() / matrix.sum()).item()
        return miou, accuracy
