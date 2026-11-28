"""Project model and its sole decoder."""
from .lr_aspp import LRASPPHead
from .segmentor import Segmentor, build_model

__all__ = ["LRASPPHead", "Segmentor", "build_model"]
