'''
Reference:
@inproceedings{li2024handcrafted,
  title={Are handcrafted filters helpful for attributing AI-generated images?},
  author={Li, Jialiang and Wang, Haoyue and Li, Sheng and Qian, Zhenxing and Zhang, Xinpeng and Vasilakos, Athanasios V},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={10698--10706},
  year={2024}
}
'''

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np  
from .attributor_base import AbstractAttributor  
from attributors import ATTRIBUTOR  
from metrics.base_metrics_class import calculate_metrics_for_train, calculate_metrics_for_test
from loss.loss_dual_margin_contrastive import DualMarginContrastiveLoss
from backbones.defl.classifier import SimpleNNClassifier
from backbones.defl.network import DEFLNetwork, SemanticFeatureExtractor


@ATTRIBUTOR.register_module(module_name='defl')  
class DEFLAttributor(AbstractAttributor):  
    def __init__(self, config=None, load_param=False):  
        super().__init__(config, load_param)  
        self.config = config or {}  

        self.device = "cuda" if torch.cuda.is_available() else "cpu"  

        self.build_model(self.config)  
        self.build_loss(self.config)  

        if load_param:  
            self.load_parameters(load_param)  

    def build_model(self, config=None):  
        self.defl_model = DEFLNetwork().to(self.device)
        self.semantic_extractor = SemanticFeatureExtractor(self.defl_model).to(self.device)
        self.nn_classifier = SimpleNNClassifier(num_classes=23).to(self.device) 


    def build_loss(self, config=None):  
        self.criterion = DualMarginContrastiveLoss(margin1=5.0, margin2=10.0).to(self.device)  
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def extract_features(self, input_data: dict) -> torch.Tensor: 
        image = input_data["image"].to(self.device) 
        clip_image = input_data["clip_image"].to(self.device) 
        return self.semantic_extractor(image,clip_image)
 

    def classifier(self, features: torch.Tensor) -> torch.Tensor:  
        result = self.nn_classifier(features)
        return result

    def forward(self, input_data: dict, inference=False) -> dict:  
        fingerprint = self.extract_features(input_data)
        result = self.classifier(fingerprint)
        return {'logits': result, "feat":fingerprint}

    def compute_losses(self, input_data: dict, pred_dict: dict) -> dict: 
        res = pred_dict["logits"].to(self.device)
        feat = pred_dict["feat"].to(self.device)
        label = input_data["label"].to(self.device)
        method_label = input_data["method_label"].to(self.device)
        loss1 = self.criterion(feat, label, method_label)
        loss2 = self.cross_entropy_loss(res, label)  
              
        return {"dual_contrastive_loss": loss1,
                "ce_loss": loss2,
                'overall': loss1+loss2
                # 'overall': loss1 # FIXME: only dual?
                }  

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