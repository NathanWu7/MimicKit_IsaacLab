from . import *
import os

# Conveniences to other module directories via relative paths
ASSET_DIR = os.path.abspath(os.path.dirname(__file__))

# Import robot configurations
from .Config.robots.smpl import SMPL_HUMANOID
from .Config.robots.g1 import G1_CFG

# Alias: Humanoid 使用 SMPL
HUMANOID_CFG = SMPL_HUMANOID

__all__ = ["ASSET_DIR", "HUMANOID_CFG", "SMPL_HUMANOID", "G1_CFG"]
