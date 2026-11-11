"""上传 MobileNetV2-stem 网络的训练入口；PyCharm 可直接使用 Config。"""
import argparse
from dataclasses import replace
from segmentation import Config, run_experiment


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
    parser.add_argument("--attention-type", choices=("original", "multiscale", "lepe"))
    parser.add_argument("--fusion-type", choices=("original", "enhanced"))
    parser.add_argument("--segment-type", dest="head_name", choices=("original", "multilevel"))
    parser.add_argument("--aux", dest="use_aux", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--learnable-squeeze", dest="use_learnable_squeeze",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pretrained-stem", dest="stem_pretrained",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--aux-loss-weight", type=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    overrides = {key: value for key, value in vars(parser.parse_args()).items() if value is not None}
    run_experiment(replace(Config(), **overrides))


if __name__ == "__main__":
    main()
