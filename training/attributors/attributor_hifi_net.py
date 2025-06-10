'''
Reference:
@inproceedings{guo2023hierarchical,
  title={Hierarchical fine-grained image forgery detection and localization},
  author={Guo, Xiao and Liu, Xiaohong and Ren, Zhiyuan and Grosz, Steven and Masi, Iacopo and Liu, Xiaoming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3155--3165},
  year={2023}
}
'''

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np  
from .attributor_base import AbstractAttributor  
from attributors import ATTRIBUTOR  
from metrics.base_metrics_class import calculate_metrics_for_train, calculate_metrics_for_test

from backbones.hifi_net.seg_hrnet import get_seg_model  
from backbones.hifi_net.seg_hrnet_config import get_cfg_defaults  
from backbones.hifi_net.NLCDetection_api import NLCDetection 

@ATTRIBUTOR.register_module(module_name='hifi_net')  
class HiFiNetAttributor(AbstractAttributor):  
    def __init__(self, config=None, load_param=False):  
        super().__init__(config, load_param)  
        self.config = config or {}  

        self.device = "cuda" if torch.cuda.is_available() else "cpu"  

        self.build_model(self.config)  
        self.build_loss(self.config)  

        if load_param:  
            self.load_parameters(load_param)  

    def build_model(self, config=None):  
        # Load configuration and models  
        FENet_cfg = get_cfg_defaults()  
        self.FENet = get_seg_model(FENet_cfg).to(self.device)  
        self.SegNet = NLCDetection().to(self.device)  


    def build_loss(self, config=None):  
        self.loss_fn = nn.CrossEntropyLoss()  

    def extract_features(self, input_data: dict) -> torch.Tensor:  
        images = input_data['image'].to(self.device)  
        features = self.FENet(images)
        return features  
 

    def classifier(self, features: torch.Tensor, images) -> torch.Tensor:  
        _, _, out0, out1, out2, out3 = self.SegNet(features, images)  
        return out0, out1, out2, out3

    def forward(self, input_data: dict, inference=False) -> dict:  
        features = self.extract_features(input_data)  
        out0, out1, out2, out3 = self.classifier(features,input_data['image'])  
        return {'logits': out3, "hierachi_logits": [out0, out1, out2, out3]}  

    def compute_losses(self, input_data: dict, pred_dict: dict) -> dict: 
        total_loss = 0
        for i in range(4):
            labels = input_data['hierachi_label'][i].to(self.device)  
            logits = pred_dict['hierachi_logits'][i].to(self.device)    
            loss = self.loss_fn(logits, labels)  
            total_loss += loss
              
        return {'overall': total_loss}  

    def compute_metrics(self, input_data: dict, pred_dict: dict, test=False) -> dict:  
        label = input_data['label']  
        semantic_label = input_data.get('semantic_label', None)  
        pred = pred_dict['logits'].to(label.device)   

        if test:  
            auc, acc, ap, conf_matrix, semantic_acc = calculate_metrics_for_test(  
                label.detach(), pred.detach(), semantic_label  
            )  
            return {  
                'acc': float(acc),  
                'auc': float(auc),  
                'ap': float(ap),  
                'conf_matrix': conf_matrix,  
                'semantic_acc': semantic_acc  
            }  
        else:  
            auc, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())  
            return {'acc': float(acc), 'auc': float(auc), 'ap': float(ap)}  

    def load_parameters(self, load_param):  
        pass  