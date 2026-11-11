"""训练管理器，包含训练前 Params/MACs 统计。"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from profile_model import profile_training

from .checkpoints import (
    atomic_save,
    check_resume_config,
    export_backbone,
    load_backbone,
)
from .config import Config, prepare_config
from .dataset import ADE20KDataset
from .engine import train_one_epoch, validate
from .freezing import (
    check_frozen_backbone,
    check_frozen_head,
    configure_trainable,
    snapshot_frozen_backbone,
    snapshot_frozen_head,
)
from .models import build_model
from .run_paths import prepare_output_dir
from .utils import seed_everything, seed_worker


def run_experiment(config: Config) -> dict:
    cfg = prepare_config(config)
    seed_everything(cfg.seed)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else cfg.device
    )

    if device.type not in ("cpu", "cuda"):
        raise ValueError(
            "This project supports CPU or CUDA only"
        )

    output = prepare_output_dir(cfg, explicit_output_dir=bool(config.output_dir))
    cfg.output_dir = str(output)
    print(f"Output directory: {output}", flush=True)

    train_set = ADE20KDataset(cfg, "training")
    val_set = ADE20KDataset(cfg, "validation")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )

    # 验证标签保持原图尺寸，因此每批一张。
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg)

    if not cfg.resume:
        load_backbone(model, cfg)

    model.to(device)

    trainable_names = configure_trainable(
        model,
        cfg.trainable_prefixes,
    )

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.min_lr,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=cfg.amp and device.type == "cuda",
    )

    start_epoch = 0
    best_miou = -math.inf
    best_accuracy = -math.inf

    if cfg.resume:
        checkpoint = torch.load(
            cfg.resume,
            map_location="cpu",
            weights_only=True,
        )

        check_resume_config(
            checkpoint,
            cfg,
            trainable_names,
        )

        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])

        start_epoch = checkpoint["epoch"]
        best_miou = checkpoint["best_miou"]
        best_accuracy = checkpoint["best_accuracy"]

        print(
            f"恢复第 {start_epoch} 轮结束时的训练状态",
            flush=True,
        )

    frozen_reference = snapshot_frozen_backbone(model)
    frozen_head_reference = snapshot_frozen_head(model)
    output.mkdir(parents=True, exist_ok=True)

    # 使用真实预处理后的训练 batch。
    # 包含卷积/线性层和 SeaFormer 的 QK^T、AV，训练开始前打印并保存。
    profile = profile_training(
        model,
        train_loader,
        output_dir=output,
    )

    (output / "config.json").write_text(
        json.dumps(
            asdict(cfg),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (output / "trainable_parameters.txt").write_text(
        "\n".join(trainable_names),
        encoding="utf-8",
    )

    (output / "samples.json").write_text(
        json.dumps(
            {
                "training": [
                    path.name for path in train_set.images
                ],
                "validation": [
                    path.name for path in val_set.images
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"mode={cfg.mode} | "
        f"CNN={cfg.backbone_name} | "
        f"head={cfg.head_name} | "
        f"device={device}\n"
        f"train={len(train_set)} | "
        f"val={len(val_set)} | "
        f"output={output}",
        flush=True,
    )

    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        seed_everything(cfg.seed + epoch)

        started = time.time()
        used_lr = optimizer.param_groups[0]["lr"]

        loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
        )

        miou, accuracy = validate(
            model=model,
            loader=val_loader,
            cfg=cfg,
            device=device,
        )

        check_frozen_backbone(
            model,
            frozen_reference,
        )
        check_frozen_head(model, frozen_head_reference)

        improved_miou = miou > best_miou
        improved_accuracy = accuracy > best_accuracy

        best_miou = max(best_miou, miou)
        best_accuracy = max(best_accuracy, accuracy)

        scheduler.step()

        payload = {
            "format_version": 2,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_miou": best_miou,
            "best_accuracy": best_accuracy,
            "val_miou": miou,
            "val_accuracy": accuracy,
            "trainable_names": trainable_names,
            "config": asdict(cfg),
        }

        if improved_miou:
            atomic_save(
                payload,
                output / "best_miou.pth",
            )
            export_backbone(
                model,
                cfg,
                output / "backbone_best_miou.pth",
            )

        if improved_accuracy:
            atomic_save(
                payload,
                output / "best_accuracy.pth",
            )

        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            atomic_save(
                payload,
                output / f"epoch_{epoch:03d}.pth",
            )

        export_backbone(
            model,
            cfg,
            output / "backbone_last.pth",
        )

        atomic_save(
            payload,
            output / "last.pth",
        )

        log_path = output / "history.csv"
        new_log = (
            not log_path.exists()
            or log_path.stat().st_size == 0
        )

        with log_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)

            if new_log:
                writer.writerow(
                    [
                        "epoch",
                        "loss",
                        "lr",
                        "val_miou",
                        "val_accuracy",
                    ]
                )

            writer.writerow(
                [
                    epoch,
                    loss,
                    used_lr,
                    miou,
                    accuracy,
                ]
            )

        frozen_message = (
            " | CNN冻结检查=通过"
            if frozen_reference is not None
            else ""
        )
        if frozen_head_reference is not None:
            frozen_message += " | head冻结检查=通过"

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"val mIoU={miou:.2%} | "
            f"val accuracy={accuracy:.2%} | "
            f"best mIoU={best_miou:.2%} | "
            f"best accuracy={best_accuracy:.2%} | "
            f"loss={loss:.4f}"
            f"{frozen_message} | "
            f"{time.time() - started:.1f}s",
            flush=True,
        )

    return {**profile, "output_dir": str(output), "run_name": cfg.run_name}
