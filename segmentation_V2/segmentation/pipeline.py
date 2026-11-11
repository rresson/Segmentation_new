"""三种训练流程，以及由用户选择的 checkpoint 决定的实验归属。"""
from __future__ import annotations

from dataclasses import replace
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import torch

from .checkpoints import checkpoint_state
from .config import Config, prepare_config
from .models.segmentor import BACKBONE_MODULES
from .run_paths import _new_run
from .trainer import run_experiment
from .utils import resolve_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_run(checkpoint: Path, output_root: Path) -> Path:
    """标准 checkpoint 归原 run；外部文件以路径及内容指纹持久化绑定。"""
    checkpoint = checkpoint.resolve()
    owner = checkpoint.parent.parent
    if re.fullmatch(r"run[1-9][0-9]*", owner.name):
        return owner

    output_root.mkdir(parents=True, exist_ok=True)
    identity = {"checkpoint": str(checkpoint), "sha256": _sha256(checkpoint)}
    source_key = json.dumps(identity, sort_keys=True)
    # 跨进程保留外部权重的归属；事务避免同时启动时创建两个归属目录。
    with closing(sqlite3.connect(output_root / ".pipeline_sources.sqlite3", timeout=30)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS sources (source TEXT PRIMARY KEY, run_name TEXT NOT NULL)")
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT run_name FROM sources WHERE source = ?", (source_key,)).fetchone()
        if row:
            owner = output_root / row[0]
            manifest = owner / "backbone_source.json"
            if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")) == identity:
                return owner

        owner = _new_run(output_root)
        # 用户删除实验目录后，允许重新使用目录编号而不继承旧绑定。
        db.execute("DELETE FROM sources WHERE run_name = ? OR source = ?", (owner.name, source_key))
        (owner / "backbone_source.json").write_text(
            json.dumps(identity, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        db.execute("INSERT INTO sources VALUES (?, ?)", (source_key, owner.name))
    return owner


def _backbone_state(state):
    if any(name.startswith("backbone.") for name in state):
        return {name.removeprefix("backbone."): value for name, value in state.items()
                if name.startswith("backbone.")}
    return {name: value for name, value in state.items()
            if name.split(".", 1)[0] in BACKBONE_MODULES}


def _has_head(state) -> bool:
    return any(name.startswith(("decode_head.", "segment_head.")) for name in state)


def backbone_training_checkpoint(selected: Path) -> Path:
    """训练 backbone 时必须有已训练的 head；校验导出 backbone 与完整权重配对。"""
    state = checkpoint_state(torch.load(selected, map_location="cpu", weights_only=True))
    if _has_head(state):
        if not _backbone_state(state):
            raise ValueError("所选 checkpoint 只有 head，需要同时包含 backbone 的完整模型。")
        return selected
    paired_names = {"backbone_best_miou.pth": "best_miou.pth", "backbone_last.pth": "last.pth"}
    paired = selected.with_name(paired_names.get(selected.name, "best_miou.pth"))
    if selected.name not in paired_names or not paired.is_file():
        raise ValueError(
            "单独训练 backbone 需要冻结已有 head。请选择完整 best_miou.pth/last.pth，"
            "或保留 backbone_best_miou.pth/backbone_last.pth 同目录的配套完整 checkpoint。"
        )
    full = checkpoint_state(torch.load(paired, map_location="cpu", weights_only=True))
    left, right = _backbone_state(state), _backbone_state(full)
    if not _has_head(full) or not left or left.keys() != right.keys() or any(
        not torch.equal(value, right[name]) for name, value in left.items()
    ):
        raise ValueError(f"所选 backbone 与配套完整 checkpoint 不匹配：{paired}")
    return paired


def _record(report: dict, choice: int, selected: Path | None, loaded: Path | None) -> None:
    (Path(report["output_dir"]) / "pipeline.json").write_text(
        json.dumps({"choice": choice, "run_name": report["run_name"],
                    "selected_checkpoint": str(selected) if selected else "",
                    "loaded_checkpoint": str(loaded) if loaded else "",
                    "output_dir": report["output_dir"]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_pipeline(config: Config, choice: int, *, checkpoint: str = "",
                 pretrain_epochs: int | None = None, head_epochs: int | None = None) -> dict:
    """1: pretrain→head；2: head-only；3: backbone-only（custom）。"""
    if choice not in (1, 2, 3):
        raise ValueError("choice must be 1, 2 or 3")
    if config.resume:
        raise ValueError("断点续训请使用 main.py --resume；pipeline 用于启动新训练。")
    if any(value is not None and value < 1 for value in (pretrain_epochs, head_epochs)):
        raise ValueError("训练轮数必须大于 0")
    if choice == 1 and checkpoint:
        raise ValueError("选项 1 自动运行新的 pretrain；手动权重用于选项 2 或 3。")
    selected = resolve_path(checkpoint).resolve() if checkpoint else None
    if choice in (2, 3) and (selected is None or not selected.is_file()):
        raise FileNotFoundError("请手动指定存在的 checkpoint 文件（--checkpoint）。")

    # 路径、初始化和训练范围由所选流程确定；保留数据、网络和优化超参数。
    base = replace(config, output_dir="", run_name="", resume="",
                   backbone_checkpoint="", model_checkpoint="", checkpoint_prefix="")
    prepare_config(base)  # 创建目录前检查常规配置。
    root = resolve_path(base.output_root).resolve()

    if choice == 1:
        pretrain = run_experiment(replace(
            base, mode="pretrain", epochs=pretrain_epochs or base.epochs,
        ))
        _record(pretrain, choice, None, None)
        owner = Path(pretrain["output_dir"]).parent
        source = Path(pretrain["output_dir"]) / "backbone_best_miou.pth"
        if not source.is_file():
            raise FileNotFoundError(f"pretrain 未生成最佳 backbone，head 阶段未启动：{source}")
        head = run_experiment(replace(
            base, mode="head", output_root=str(owner.parent), run_name=owner.name,
            backbone_checkpoint=str(source), epochs=head_epochs or base.epochs,
        ))
        _record(head, choice, source, source)
        return {"choice": choice, "run_dir": str(owner), "pretrain": pretrain, "head": head}

    if choice == 2:
        # 提前确认文件是可读的模型权重，避免给无效文件分配实验编号。
        state = checkpoint_state(torch.load(selected, map_location="cpu", weights_only=True))
        if not _backbone_state(state):
            raise ValueError("所选 checkpoint 不包含本网络的 backbone 权重。")
        owner = checkpoint_run(selected, root)
        head = run_experiment(replace(
            base, mode="head", output_root=str(owner.parent), run_name=owner.name,
            backbone_checkpoint=str(selected), epochs=head_epochs or base.epochs,
        ))
        _record(head, choice, selected, selected)
        return {"choice": choice, "run_dir": str(owner), "head": head}

    full_checkpoint = backbone_training_checkpoint(selected)
    # 更新 backbone 后是一份新的 backbone 实验，另建 run，保留来源信息。
    owner = _new_run(root)
    backbone = run_experiment(replace(
        base, mode="custom", run_name=owner.name,
        model_checkpoint=str(full_checkpoint), trainable_prefixes=("backbone",),
        allow_random_backbone=False,
    ))
    _record(backbone, choice, selected, full_checkpoint)
    return {"choice": choice, "run_dir": str(owner), "backbone": backbone}
