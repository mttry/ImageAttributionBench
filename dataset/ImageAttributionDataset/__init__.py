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
from .resnet50_dataset import Resnet50Dataset
