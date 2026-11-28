"""运行 main.py 启动 pipeline；--mode/--resume 保留单阶段训练入口。"""
from bootstrap_dependencies import ensure_dependencies


def main(argv=None):
    ensure_dependencies()
    from segmentation.cli import main as run_cli

    return run_cli(argv)


if __name__ == "__main__":
    # Windows 多进程 DataLoader 需要入口保护。
    main()
