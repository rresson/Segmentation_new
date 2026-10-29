"""模型组件与统一组装函数。"""

from .cnn import DemoBackbone, SmallCNNBackbone
from .heads import FPNHead, SimpleHead
from .segmentor import Segmentor, build_model

__all__ = [
    "DemoBackbone",
    "SmallCNNBackbone",
    "FPNHead",
    "SimpleHead",
    "Segmentor",
    "build_model",
]
