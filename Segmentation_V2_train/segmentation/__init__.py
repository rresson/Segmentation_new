"""对 main.py 暴露稳定的公共接口，导入不会启动训练。"""

from .config import Config


def run_experiment(config: Config) -> dict:
    # 模型及统计工具可独立导入；实际训练时才加载 trainer，避免循环导入。
    from .trainer import run_experiment as run
    return run(config)

__all__ = ["Config", "run_experiment"]
