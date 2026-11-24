"""运行 main.py 启动 pipeline；--mode/--resume 保留单阶段训练入口。"""
from segmentation.cli import main as run_cli


def main(argv=None):
    return run_cli(argv)


if __name__ == "__main__":
    # Windows 多进程 DataLoader 需要入口保护。
    main()
