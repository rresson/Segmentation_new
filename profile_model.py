"""原版分割网络的参数量与前向 MACs 统计。

统计范围：
1. nn.Conv2d 和 nn.Linear，1 次乘加记为 1 MAC。
2. Params 包含冻结参数，不包含 BatchNorm 等模块的 buffers。
3. 不统计 bias 加法、池化、归一化、激活、插值、
   损失、反向传播和优化器计算。

模型要求：
- model.backbone：CNN backbone。
- model.decode_head：分割头。
- model(images)：返回分割 logits。
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset


@contextmanager
def preserve_rng(loader):
    """保存并恢复本工程使用的随机状态和 DataLoader generator。"""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    cuda_state = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_initialized()
        else None
    )

    generators = {}
    objects = (
        loader,
        getattr(loader, "sampler", None),
        getattr(
            getattr(loader, "batch_sampler", None),
            "sampler",
            None,
        ),
    )

    for obj in objects:
        generator = getattr(obj, "generator", None)
        if isinstance(generator, torch.Generator):
            generators[id(generator)] = (
                generator,
                generator.get_state(),
            )

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)

        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

        for generator, state in generators.values():
            generator.set_state(state)


def profile_model(
    model: nn.Module,
    inputs: torch.Tensor,
) -> dict:
    """统计一次前向计算，返回报告；不执行反向传播。"""
    if inputs.ndim != 4 or min(inputs.shape) < 1:
        raise ValueError("input must be non-blank tensor [B,C,H,W]")

    if not inputs.is_floating_point():
        raise ValueError("Please pass in the preprocessed floating-point image tensor.")

    first_conv = next(
        (
            module
            for module in model.backbone.modules()
            if isinstance(module, nn.Conv2d)
        ),
        None,
    )

    if (
        first_conv is not None
        and inputs.shape[1] != first_conv.in_channels
    ):
        raise ValueError(
            f"image has {inputs.shape[1]} channels，"
            f"CNN needs {first_conv.in_channels} channels"
        )

    rows = []
    handles = []
    feature_shapes = []

    states = {
        module: module.training
        for module in model.modules()
    }

    batch_size = inputs.shape[0]
    cost_keys = ("convolution_macs", "linear_macs")

    def record(name, layer, args, output):
        convolution_macs = 0
        linear_macs = 0

        if isinstance(layer, nn.Conv2d):
            kernel_elements = (
                layer.kernel_size[0] * layer.kernel_size[1]
            )

            # 输出元素数包含 batch、输出通道和输出空间尺寸。
            # 分组/深度卷积通过 in_channels / groups 处理。
            convolution_macs = (
                output.numel()
                * (layer.in_channels // layer.groups)
                * kernel_elements
            )

        elif isinstance(layer, nn.Linear):
            linear_macs = output.numel() * layer.in_features

        if name.startswith("backbone."):
            component = "backbone"
        elif name.startswith("decode_head."):
            component = "decode_head"
        else:
            component = "other"

        rows.append(
            {
                "name": name,
                "component": component,
                "input_shape": list(args[0].shape),
                "output_shape": list(output.shape),
                "convolution_macs": convolution_macs,
                "linear_macs": linear_macs,
            }
        )

    def record_features(module, args, outputs):
        values = (
            [outputs]
            if isinstance(outputs, torch.Tensor)
            else outputs
        )
        feature_shapes.extend(
            list(value.shape) for value in values
        )

    try:
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                handles.append(
                    layer.register_forward_hook(
                        lambda module, args, output, name=name:
                        record(name, module, args, output)
                    )
                )

        handles.append(
            model.backbone.register_forward_hook(
                record_features
            )
        )

        # 避免统计时更新 BatchNorm 或启用 Dropout。
        model.eval()

        with torch.inference_mode():
            output = model(inputs)

    finally:
        # 即使 forward 报错，也移除 hook 并恢复各模块状态。
        for handle in handles:
            handle.remove()

        for module, training in states.items():
            module.training = training

    def summarize(module, selected):
        costs = {
            key: sum(row[key] for row in selected)
            for key in cost_keys
        }
        macs = sum(costs.values())

        return {
            "parameters": sum(
                parameter.numel()
                for parameter in module.parameters()
            ),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            ),
            **costs,
            "total_counted_macs": macs,
            "macs_per_image": macs / batch_size,
        }

    parts = {
        "backbone": summarize(
            model.backbone,
            [
                row for row in rows
                if row["component"] == "backbone"
            ],
        ),
        "decode_head": summarize(
            model.decode_head,
            [
                row for row in rows
                if row["component"] == "decode_head"
            ],
        ),
        "total": summarize(model, rows),
    }

    return {
        "input_shape": list(inputs.shape),
        "output_shape": list(output.shape),
        "backbone_feature_shapes": feature_shapes,
        "components": parts,
        **parts["total"],
        "head_parameters": parts["decode_head"]["parameters"],
        "backbone_parameters": parts["backbone"]["parameters"],
        "layers": rows,
    }


def print_profile(report: dict) -> None:
    """打印输入尺寸和各模块统计结果。"""
    batch, channels, height, width = report["input_shape"]

    print(
        f"\n[Model] Input："
        f"B={batch}, C={channels}, H={height}, W={width}",
        flush=True,
    )
    print(
        "CNN output/head input："
        f"{report['backbone_feature_shapes']}",
        flush=True,
    )
    print(
        f"{'module':<16} "
        f"{'Params':>14} "
        f"{'trainable Params':>14} "
        f"{'MACs/image':>18} "
        f"{'MACs/batch':>18}",
        flush=True,
    )

    for key, label in (
        ("backbone", "CNN backbone"),
        ("decode_head", "segmentation head"),
        ("total", "whole net"),
    ):
        row = report["components"][key]
        print(
            f"{label:<16} "
            f"{row['parameters']:>14,} "
            f"{row['trainable_parameters']:>14,} "
            f"{row['macs_per_image']:>18,.0f} "
            f"{row['total_counted_macs']:>18,}",
            flush=True,
        )

    print(
        "MACs contains only Conv2d/Linear forward multiplication;"
        "does not include interpolation, normalization, or backpropagation.\n",
        flush=True,
    )


def profile_training(
    model: nn.Module,
    train_loader,
    output_dir=None,
) -> dict:
    """从真实训练 DataLoader 取一个 batch，统计、打印并可选保存。"""
    if (
        isinstance(train_loader.dataset, IterableDataset)
        or train_loader.persistent_workers
    ):
        raise ValueError(
            "Pre-training statistical requires map-style dataset "
            "and persistent_workers=False"
        )

    with preserve_rng(train_loader):
        iterator = iter(train_loader)

        try:
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise ValueError(
                    "trainable DataLoader does not contain batch，"
                    "please check dataset or drop_last"
                ) from exc
        finally:
            del iterator

        images = (
            batch[0]
            if isinstance(batch, (tuple, list))
            else batch
        )

        if not isinstance(images, torch.Tensor):
            raise TypeError(
                "trainable batch should be image tensor or (images, labels)"
            )

        if not images.is_floating_point():
            raise ValueError("please input the processed floating point image tensor")

        parameter = next(model.parameters())
        images = images.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )

        report = profile_model(model, images)

    report["counting"] = __doc__
    report["input_source"] = (
        "one actual training batch after preprocessing"
    )

    print(
        "pre-training detection: The dimensions are derived from the actual preprocessed training batch,"
        "not the size of original image.",
        flush=True,
    )
    print_profile(report)

    if output_dir is not None:
        path = Path(output_dir) / "model_profile.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Model saved：{path}", flush=True)

    return report


def main() -> None:
    """独立运行：按 Config 构建模型，使用模拟 RGB 输入统计。"""
    from segmentation.config import Config, prepare_config
    from segmentation.freezing import configure_trainable
    from segmentation.models import build_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Simulates the input side length, using Config.crop_size by default.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output available",
    )
    args = parser.parse_args()

    cfg = prepare_config(Config())
    size = cfg.crop_size if args.size is None else args.size

    if size < 32 or args.batch_size < 1:
        parser.error("size must be >=32 and batch-size >=1")

    torch.set_num_threads(2)

    # 结构统计不需要加载 checkpoint。
    # 由 prepare_config 应用 pretrain/head/custom 模式设置。
    model = build_model(cfg)
    configure_trainable(model, cfg.trainable_prefixes)

    inputs = torch.zeros(
        args.batch_size,
        3,
        size,
        size,
    )

    report = profile_model(model, inputs)
    report["counting"] = __doc__
    report["input_source"] = "synthetic RGB input"

    print(
        f"individual statistic："
        f"mode={cfg.mode}, head={cfg.head_name}"
    )
    print_profile(report)

    if args.output:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()