"""兼容独立 pipeline 入口，也可直接运行 main.py。"""
from segmentation.cli import main as run_cli


def main(argv=None):
    return run_cli(argv, pipeline_only=True)


if __name__ == "__main__":
    main()
