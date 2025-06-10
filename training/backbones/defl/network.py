import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import convolve2d
from itertools import combinations

import torchvision.models as models

def partition_mhfs(mhf_filters):
    np.random.shuffle(mhf_filters)
    partitioned_filters = [
        mhf_filters[:64],
        mhf_filters[64:128],
        mhf_filters[128:192],
        np.concatenate((mhf_filters[192:254], mhf_filters[192:194]))
    ]
    return [torch.tensor(np.array(filters), dtype=torch.float32).unsqueeze(1) for filters in partitioned_filters]


class DirectionalConvolutionalBlock(nn.Module):
    def __init__(self, filters, in_channels=64):
        super(DirectionalConvolutionalBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1, bias=False)
        
        if filters.shape == (64, 1, 3, 3): 
            filters = filters.repeat(1, in_channels, 1, 1)  
        
        with torch.no_grad():
            self.conv.weight = nn.Parameter(filters)

    def forward(self, x):
        return F.relu(self.conv(x))


class StandardConvolutionalBlock(nn.Module):
    def __init__(self, in_channels=64):
        super(StandardConvolutionalBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class DEFLNetwork(nn.Module):
    def __init__(self, partitioned_filters=None):
        super(DEFLNetwork, self).__init__()
        if partitioned_filters is None:
                from .filter import composite_filters
                mhf_filters = composite_filters  
                partitioned_filters = partition_mhfs(mhf_filters)
        self.level1_dcb = DirectionalConvolutionalBlock(partitioned_filters[0], in_channels=3)
        self.level1_scb = StandardConvolutionalBlock(in_channels=3)
        
        self.level2_dcb = DirectionalConvolutionalBlock(partitioned_filters[1], in_channels=128)
        self.level2_scb = StandardConvolutionalBlock(in_channels=128)
        
        self.level3_dcb = DirectionalConvolutionalBlock(partitioned_filters[2], in_channels=128)
        self.level3_scb = StandardConvolutionalBlock(in_channels=128)
        
        self.level4_dcb = DirectionalConvolutionalBlock(partitioned_filters[3], in_channels=128)
        self.level4_scb = StandardConvolutionalBlock(in_channels=128)

    def forward(self, x):
        x_dcb = self.level1_dcb(x)
        x_scb = self.level1_scb(x)
        x = torch.cat((x_dcb, x_scb), dim=1)
        
        x_dcb = self.level2_dcb(x)
        x_scb = self.level2_scb(x)
        x = torch.cat((x_dcb, x_scb), dim=1)
        
        x_dcb = self.level3_dcb(x)
        x_scb = self.level3_scb(x)
        x = torch.cat((x_dcb, x_scb), dim=1)
        
        x_dcb = self.level4_dcb(x)
        x_scb = self.level4_scb(x)
        x = torch.cat((x_dcb, x_scb), dim=1)
        
        return x

import clip
from torchvision.models import resnet50
import torch
import torch.nn as nn

class SemanticFeatureExtractor(nn.Module):
    def __init__(self, defl_model, embed_dim=768):
        super(SemanticFeatureExtractor, self).__init__()
        self.defl_model = defl_model

        self.clip_model, self.clip_preprocess = clip.load("RN50x16", device="cuda")
        self.clip_model.eval()
        self.clip_model_visual = self.clip_model.visual

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.adjust_channels = nn.Conv2d(128, 64, kernel_size=1)

        self.resnet_for_defl = resnet50(pretrained=True)
        self.resnet_for_defl.fc = nn.Identity() 
        self.resnet_for_defl = nn.Sequential(
            *list(self.resnet_for_defl.children())[4:],  
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten() 
        )

        fusion_dim = 2048 + embed_dim  
        self.fusion_layer = nn.Linear(fusion_dim, 2048)

    def forward(self, x, x_clip):

        defl_features = self.defl_model(x)  

        adjusted_defl_features = self.adjust_channels(defl_features)

        resnet_features = self.resnet_for_defl(adjusted_defl_features) 
        with torch.no_grad():  
            clip_features = self.clip_model_visual(x_clip)  

        combined_features = torch.cat((resnet_features, clip_features), dim=1)  

        fused_features = self.fusion_layer(combined_features)
        
        return fused_features






