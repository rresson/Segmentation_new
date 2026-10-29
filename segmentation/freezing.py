"""按模块名/参数名前缀控制梯度及训练状态，避免冻结区域的 BatchNorm 继续更新。"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def name_matches(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def configure_trainable(model: nn.Module, prefixes: tuple[str, ...]) -> list[str]:
    prefixes = tuple(p.rstrip(".") for p in prefixes)
    if not prefixes or any(not p for p in prefixes):
        raise ValueError("训练前缀不能为空。")
    named = list(model.named_parameters())
    for prefix in prefixes:
        if not any(name_matches(name, prefix) for name, _ in named):
            raise ValueError(
                f"没有参数匹配 {prefix!r}，请检查 model.named_parameters()。"
            )
    for name, parameter in named:
        parameter.requires_grad_(any(name_matches(name, prefix) for prefix in prefixes))
        parameter.grad = None
    names = [n for n, p in named if p.requires_grad]
    total = sum(p.numel() for _, p in named)
    active = sum(p.numel() for _, p in named if p.requires_grad)
    print(f"trained parameters：{active:,}/{total:,}；range：{prefixes}", flush=True)
    return names


def set_training_mode(model: nn.Module, prefixes: tuple[str, ...]) -> None:
    """每个 epoch 重设：冻结区域保持 eval，选中区域才进入 train。

    处理 BatchNorm buffers 与 Dropout。父模块若包含可训练后代，保留训练标志；
    纯冻结子树设为 eval；选中模块内无参数的 Dropout 也会正确启用。
    若只选某个 weight 而没有选整个模块，它所属模块仍进入训练模式。
    自定义模块若在 eval 下仍修改 buffer，需要额外检查该模块实现。
    """
    model.train()
    prefixes = tuple(p.rstrip(".") for p in prefixes)
    for name, module in model.named_modules():
        selected = any(name_matches(name, p) for p in prefixes)
        has_trainable = any(p.requires_grad for p in module.parameters())
        if not selected and not has_trainable:
            module.eval()


def snapshot_frozen_backbone(model: nn.Module) -> dict[str, Tensor] | None:
    """全部冻结时记录参数和 BN buffers；部分微调时不对整个 CNN 做不变性检查。"""
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        return None
    return {
        name: value.detach().cpu().clone()
        for name, value in model.backbone.state_dict().items()
    }


def check_frozen_backbone(
    model: nn.Module, reference: dict[str, Tensor] | None
) -> None:
    if reference is None:
        return
    for name, value in model.backbone.state_dict().items():
        if not torch.equal(value.detach().cpu(), reference[name]):
            raise RuntimeError(f"冻结检查失败：backbone.{name} 发生改变。")
