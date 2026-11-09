"""随机种子与路径管理。"""
from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import torch


def seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(value: str) -> Path:
    """所有相对路径按工程根目录解析，避免 PyCharm 工作目录改变路径含义。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path
