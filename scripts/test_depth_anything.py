import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.depth_anything_v2.dpt import DepthAnythingV2

print("Import successful!")
