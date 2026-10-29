"""对 main.py 暴露稳定的公共接口，导入不会启动训练。"""

from .config import Config
from .trainer import run_experiment

__all__ = ["Config", "run_experiment"]
