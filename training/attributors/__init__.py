import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import ATTRIBUTOR
from .attributor_resnet50 import Resnet50Attributor
from .attributor_clip_lr import ClipLrAttributor
from .attributor_repmix import RepmixAttributor
from .attributor_hifi_net import HiFiNetAttributor
from .attributor_defl import DEFLAttributor
from .attributor_ssp import SSPAttributor
from .attributor_patchcraft import PatchCraftAttributor
from .attributor_dct import DCTAttributor
from .attributor_dna import DNAAttributor
from .attributor_ucf import UCFAttributor
from .attributor_patch import PatchAttributor
from .attributor_gfd import GFDAttributor
from .attributor_pose import POSEAttributor
