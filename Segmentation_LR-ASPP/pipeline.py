"""兼容独立 pipeline 入口，也可直接运行 main.py。"""
from bootstrap_dependencies import ensure_dependencies


def main(argv=None):
    ensure_dependencies()
    from segmentation.cli import main as run_cli

    return run_cli(argv, pipeline_only=True)


if __name__ == "__main__":
    main()
