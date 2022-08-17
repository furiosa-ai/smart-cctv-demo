
from pathlib import Path
import importlib
import sys

parent_dir = Path(__file__).parent.parent
arcface_dir = parent_dir / "recognition" / "arcface_torch"

sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(arcface_dir))

if "utils" in sys.modules:
    importlib.reload(sys.modules["utils"])

from recognition.arcface_torch.utils.arcface_predictor import ArcFacePredictor
from recognition.arcface_torch.utils.face_extract import FaceExtractor
from recognition.arcface_torch.utils.face_gallery import FaceGallery


del sys.path[0]
del sys.path[0]


if "utils" in sys.modules:
    importlib.reload(sys.modules["utils"])