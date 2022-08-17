
import importlib
import sys
from pathlib import Path


parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


if "torchreid" in sys.modules:
    importlib.reload(sys.modules["torchreid"])

from utils.reid_predictor import ReIdPredictor
from utils.box_extract import BoxExtractor, BoxExtractorIdentity
from utils.reid import ReIdGallery

del sys.path[0]

# with ImportEnv() as env:
"""
env = ImportEnv()
env.__enter__()

parent_dir = Path(__file__).parent.parent

env.add_path(parent_dir)
env.unload_module("utils")
env.unload_module("torchreid")
env.unload_module("models.common")
env.unload_module("models")

from utils.box_extract import BoxExtractor, BoxExtractorIdentity
from utils.reid import ReIdGallery
from utils.reid_predictor import ReIdPredictor

env.__exit__(None, None, None)
"""

# ReIdPredictor = importlib.import_module(".torchreid.utils.reid_predictor", "torchreid").ReIdPredictor