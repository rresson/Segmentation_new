"""将预训练权重与后续训练绑定到同一 run；只在训练启动时创建目录。"""
from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config


def _run_number(path: Path) -> int:
    match = re.fullmatch(r"run([1-9][0-9]*)", path.name)
    return int(match.group(1)) if match else 0


def _reserve_directory(requested: Path) -> Path:
    requested.parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        output = (requested if index == 1 else
                  requested.with_name(f"{requested.name}_{index}"))
        try:
            output.mkdir(exist_ok=False)
            return output
        except FileExistsError:
            index += 1


def _new_run(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    index = max((_run_number(path) for path in root.iterdir()), default=0) + 1
    while True:
        run_dir = root / f"run{index}"
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            index += 1


def reserve_head_directory(run_dir: Path) -> Path:
    """从磁盘读取编号，重启程序后继续使用 head1、head2……。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in run_dir.iterdir():
        # 兼容先前版本的 head、head_2 命名。
        match = re.fullmatch(r"head(?:_?([1-9][0-9]*))?", path.name)
        if match:
            numbers.append(int(match.group(1) or 1))
    index = max(numbers, default=0) + 1
    while True:
        output = run_dir / f"head{index}"
        try:
            output.mkdir(exist_ok=False)
            return output
        except FileExistsError:
            index += 1


def _run_for_stage(root: Path, stage: Path) -> Path | None:
    run_dir = stage.resolve().parent
    return run_dir if run_dir.parent == root and _run_number(run_dir) else None


def prepare_output_dir(cfg: Config, *, explicit_output_dir: bool) -> Path:
    """接收 prepare_config 的结果，解析实际 run/权重并独占输出目录。"""
    root = Path(cfg.output_root).resolve()

    if cfg.resume:
        checkpoint = Path(cfg.resume).resolve()
        output = Path(cfg.output_dir).resolve() if explicit_output_dir else checkpoint.parent
        if output != checkpoint.parent:
            raise ValueError("output_dir must match the resume file directory")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        resumed_run = output.parent.name if _run_number(output.parent) else ""
        if cfg.run_name and cfg.run_name != resumed_run:
            raise ValueError("run_name differs from the resume checkpoint directory")
        cfg.run_name = resumed_run
        return output

    run_dir = root / cfg.run_name if cfg.run_name else None
    if explicit_output_dir:
        output_run = _run_for_stage(root, Path(cfg.output_dir))
        if output_run is not None:
            if run_dir is not None and run_dir != output_run:
                raise ValueError("run_name and output_dir refer to different runs")
            run_dir = output_run

    # 手动权重优先；auto 已在 prepare_config 中转为空字符串。
    explicit_checkpoint = cfg.backbone_checkpoint or cfg.model_checkpoint
    if explicit_checkpoint and not Path(explicit_checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {explicit_checkpoint}")

    # 保留原 V2_V3 custom 的随机初始化行为；pipeline 选项 3 显式给出完整模型。
    if cfg.mode == "head" or (cfg.mode == "custom" and explicit_checkpoint):
        if explicit_checkpoint:
            if run_dir is None:
                run_dir = _run_for_stage(root, Path(explicit_checkpoint).parent)
        else:
            if run_dir is None:
                runs = [path for path in root.glob("run*")
                        if _run_number(path) and path.is_dir()
                        and (path / "pretrain").is_dir()]
                if not runs:
                    raise FileNotFoundError(
                        f"No pretrain run under {root}; run pretrain first or specify a checkpoint."
                    )
                run_dir = max(runs, key=_run_number)

            checkpoint = run_dir / "pretrain/backbone_best_miou.pth"
            if not checkpoint.is_file():
                # 当前轮没有权重时明确报错，不悄悄退回更早的实验。
                raise FileNotFoundError(
                    f"Pretrain checkpoint not found: {checkpoint}. "
                    "Finish this pretrain run or explicitly choose run_name/backbone_checkpoint."
                )
            cfg.backbone_checkpoint = str(checkpoint)
            cfg.allow_random_backbone = False
            print(f"Auto backbone checkpoint: {checkpoint}", flush=True)

    if explicit_output_dir:
        output = _reserve_directory(Path(cfg.output_dir))
    else:
        if run_dir is None:
            run_dir = _new_run(root)
        elif cfg.mode == "pretrain":
            # 显式编号对应一个预训练实验；已有编号请使用 resume 或换新编号。
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                run_dir.mkdir(exist_ok=False)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Explicit run already exists: {run_dir}; clear run_name for a new run or use resume."
                ) from exc
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
        output = (reserve_head_directory(run_dir) if cfg.mode == "head"
                  else _reserve_directory(run_dir / cfg.mode))

    cfg.run_name = run_dir.name if run_dir is not None else ""
    return output
