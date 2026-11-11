"""公共命令行入口：默认 pipeline，显式 --mode/--resume 时执行单阶段训练。"""
import argparse
from dataclasses import replace

from . import Config, run_experiment


def add_config_arguments(parser, *, single_stage=True):
    """训练入口和 pipeline 共享数据、模型与优化配置参数。"""
    if single_stage:
        parser.add_argument("--mode", choices=("pretrain", "head", "custom"))
    paths = ("data-root", "output-root", "device")
    if single_stage:
        paths += ("output-dir", "run-name", "backbone-checkpoint", "model-checkpoint", "resume")
    for name in paths:
        parser.add_argument(f"--{name}")
    for name in ("epochs", "batch-size", "num-workers", "crop-size", "val-short-side",
                 "val-max-side", "max-train-samples", "max-val-samples"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--seaformer-attention", choices=("original", "multiscale", "lepe"))
    parser.add_argument("--pretrained-stem", dest="stem_pretrained",
                        action=argparse.BooleanOptionalAction, default=None)


def main(argv=None, *, pipeline_only=False):
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser, single_stage=not pipeline_only)
    parser.add_argument("--choice", type=int, choices=(1, 2, 3),
                        help="pipeline 选项；省略时显示交互菜单")
    parser.add_argument("--checkpoint", help="pipeline 选项 2/3 手动选择的权重")
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--head-epochs", type=int)
    args = vars(parser.parse_args(argv))
    pipeline_args = {key: args.pop(key) for key in
                     ("choice", "checkpoint", "pretrain_epochs", "head_epochs")}
    cfg = replace(Config(), **{key: value for key, value in args.items() if value is not None})
    single_stage = not pipeline_only and (args.get("mode") is not None or bool(args.get("resume")))
    if single_stage:
        if any(value is not None for value in pipeline_args.values()):
            parser.error("--mode/--resume 不能与 --choice/--checkpoint/阶段轮数混用。")
        return run_experiment(cfg)
    for key in ("output_dir", "run_name", "backbone_checkpoint", "model_checkpoint"):
        if args.get(key) is not None:
            parser.error(f"--{key.replace('_', '-')} 用于单阶段训练，请同时指定 --mode；"
                         "pipeline 手动权重请用 --checkpoint。")
    return launch_pipeline(parser, cfg, **pipeline_args)


def launch_pipeline(parser, cfg, *, choice, checkpoint, pretrain_epochs, head_epochs):
    from .pipeline import run_pipeline

    interactive = choice is None
    if interactive:
        print("1. 联合训练 backbone + head：pretrain → 冻结 backbone → 训练 head")
        print("2. 单独训练 head：手动选择 backbone，冻结 backbone")
        print("3. 单独训练 backbone（custom）：手动选择权重，冻结已有 head")
        while True:
            try:
                answer = input("请选择 [1/2/3]：").strip()
            except EOFError:
                parser.error("未读取到选项；命令行运行请指定 --choice 1/2/3。")
            if answer in ("1", "2", "3"):
                choice = int(answer)
                break
            print("请输入 1、2 或 3。")

    if choice in (2, 3) and not checkpoint:
        if not interactive:
            parser.error("单独训练请手动指定 --checkpoint PATH。")
        try:
            checkpoint = input("请输入 checkpoint 文件路径：").strip().strip('"').strip("'")
        except EOFError:
            parser.error("未读取到 checkpoint 路径。")

    result = run_pipeline(cfg, choice, checkpoint=checkpoint or "",
                          pretrain_epochs=pretrain_epochs, head_epochs=head_epochs)
    print(f"\n训练完成，实验目录：{result['run_dir']}", flush=True)
    for stage in ("pretrain", "head", "backbone"):
        if stage in result:
            print(f"{stage}: {result[stage]['output_dir']}", flush=True)
    return result
