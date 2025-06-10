import os
import sys

current_file_path = os.path.abspath(__file__)  
parent_dir = os.path.dirname(current_file_path)          # ImageAttributionDataset 
dataset_root_dir = os.path.dirname(parent_dir)           # dataset
project_root_dir = os.path.dirname(dataset_root_dir)     # project-root
sys.path.append(parent_dir)
sys.path.append(dataset_root_dir)
sys.path.append(project_root_dir)


from training.metrics.registry import DATASET
from .dataset_resnet50 import Resnet50Dataset
from .dataset_clip_lr import ClipLrDataset
from .dataset_repmix import RepmixDataset
from .dataset_hifi_net import HiFiNetDataset
from .dataset_defl import DEFLDataset
from .dataset_ssp import SSPDataset
from .dataset_patchcraft import PatchCraftDataset
from .dataset_dct import DCTDataset
from .dataset_dna import DNADataset
from .dataset_ucf import UCFDataset
from .dataset_patch import PatchDataset
from .dataset_gfd import GFDDataset 
from .dataset_pose import POSEDataset 
