"""PyCharm 统一运行入口；请在 segmentation/config.py 中修改配置。"""

from segmentation import Config, run_experiment


def main() -> None:
    # main 只创建配置并调用训练管理器，具体功能由独立模块负责。
    run_experiment(Config())


if __name__ == "__main__":
    # Windows 的多进程 DataLoader 需要入口保护。
    main()
