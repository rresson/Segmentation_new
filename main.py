"""统一运行入口：支持 Config 默认值和命令行覆盖。"""

import argparse
from dataclasses import replace
from segmentation import Config, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pretrain", "head", "custom"))
    for name in ("data-root", "output-dir", "backbone-checkpoint", "resume", "device"):
        parser.add_argument(f"--{name}")
    for name in ("epochs", "batch-size", "num-workers", "crop-size",
                 "val-short-side", "val-max-side", "max-train-samples", "max-val-samples"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--seaformer-attention", choices=("original", "multiscale", "lepe"))
    parser.add_argument("--no-pretrained-stem", action="store_true",
                        help="Disable ImageNet stem initialization for offline/debug runs")
    args = vars(parser.parse_args())
    no_pretrained = args.pop("no_pretrained_stem")
    overrides = {key: value for key, value in args.items() if value is not None}
    if no_pretrained:
        overrides["stem_pretrained"] = False
    run_experiment(replace(Config(), **overrides))


if __name__ == "__main__":
    # Windows 的多进程 DataLoader 需要入口保护。
    main()
