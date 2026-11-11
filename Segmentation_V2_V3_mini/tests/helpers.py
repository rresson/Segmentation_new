import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


def make_dataset(root):
    rng = np.random.default_rng(4)
    for split, count in (("training", 2), ("validation", 1)):
        images = root / "images" / split
        masks = root / "annotations" / split
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        for i in range(count):
            Image.fromarray(rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)).save(images / f"{i}.jpg")
            mask = rng.integers(1, 151, (32, 40), dtype=np.uint8)
            mask[:2] = 0
            Image.fromarray(mask).save(masks / f"{i}.png")


def reference_modules(project):
    path = Path(__file__).resolve().parents[2] / project / 'segmentation'
    if not path.is_dir():
        pytest.skip(f'Optional cross-project reference unavailable: {project}')
    alias = '_reference_' + project.lower()
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(alias, path / '__init__.py',
                                                    submodule_search_locations=[str(path)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    return (importlib.import_module(alias + '.config'), importlib.import_module(alias + '.models'),
            importlib.import_module(alias + '.checkpoints'))
