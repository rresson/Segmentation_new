"""交互训练入口：联合两阶段 / 仅 head / 仅 backbone。"""
import argparse
from dataclasses import replace

from main import add_config_arguments
from segmentation import Config
from segmentation.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser, single_stage=False)
    parser.add_argument("--choice", type=int, choices=(1, 2, 3))
    parser.add_argument("--checkpoint", help="选项 2/3 手动选择的 backbone 或完整模型权重")
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--head-epochs", type=int)
    args = vars(parser.parse_args())
    choice = args.pop("choice")
    checkpoint = args.pop("checkpoint")
    pretrain_epochs = args.pop("pretrain_epochs")
    head_epochs = args.pop("head_epochs")
    cfg = replace(Config(), **{name: value for name, value in args.items() if value is not None})

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


if __name__ == "__main__":
    main()
