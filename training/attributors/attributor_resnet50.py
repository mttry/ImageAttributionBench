'''
Reference:
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
'''

import torch  
import torch.nn as nn  
import torchvision.models as models  
from .attributor_base import AbstractAttributor  
import torch.nn.functional as F  
from attributors import ATTRIBUTOR
from metrics.base_metrics_class import calculate_metrics_for_train, calculate_metrics_for_test

@ATTRIBUTOR.register_module(module_name='resnet50')
class Resnet50Attributor(AbstractAttributor):  
    def __init__(self, config=None, load_param=False):  
        super().__init__(config, load_param)  
        self.config = config or {}  

        self.build_model(self.config)  
        self.build_loss(self.config)  

        if load_param:  
            self.load_parameters(load_param)  

    def build_model(self, config):  
        self.backbone = models.resnet50(pretrained=config.get('pretrained', True))  
        self.backbone.fc = nn.Identity()  

        num_classes = config.get('num_classes', 23)  
        self.fc = nn.Linear(2048, num_classes)  

    def build_loss(self, config):  
        self.loss_fn = nn.CrossEntropyLoss()  

    def extract_features(self, input_data: dict) -> torch.Tensor:  
        x = input_data['image']  # Tensor (B,C,H,W)  
        features = self.backbone(x)  # (B, 2048)  
        return features  

    def classifier(self, features: torch.Tensor) -> torch.Tensor:  
        logits = self.fc(features)  # (B, num_classes)  
        return logits  

    def forward(self, input_data: dict, inference=False) -> dict:  
        features = self.extract_features(input_data)  
        logits = self.classifier(features)  

        out = {'logits': logits, 
               'features': features,
               'cls': torch.argmax(logits, dim=1)  }  

        if inference:  
            probs = F.softmax(logits, dim=1)  
            out['probs'] = probs  

        return out  

    def compute_losses(self, input_data: dict, pred_dict: dict) -> dict:  
        labels = input_data['label'] 
        logits = pred_dict['logits']  
        loss = self.loss_fn(logits, labels)  
        return {'overall': loss}  

    def compute_metrics(self, input_data: dict, pred_dict: dict, test = False) -> dict:  
        label = input_data['label']
        semantic_label = input_data.get('semantic_label', None)  
        pred = pred_dict['logits']
        # compute metrics for batch data
        if test:
            auc, acc, ap, conf_matrix, semantic_acc= calculate_metrics_for_test(label.detach(), pred.detach(),semantic_label)
            metric_batch_dict = {'acc': float(acc), 'auc': float(auc), 'ap': float(ap),
                                 "conf_matrix": conf_matrix,
                                 'semantic_acc':semantic_acc # acc of each semantic
                                 }
            return metric_batch_dict
        else:
            # train and val
            auc, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
            metric_batch_dict = {'acc': float(acc), 'auc': float(auc), 'ap': float(ap)}
            return metric_batch_dict

    def load_parameters(self, load_param):  
        if isinstance(load_param, str):  
            path = load_param  
        elif load_param is True:  
            path = self.config.get('pretrained_path', None)  
            if path is None:  
                raise ValueError("No default pretrained path set in config.")  
        else:  
            return   

        state_dict = torch.load(path, map_location='cpu')  
        self.load_state_dict(state_dict)  
        print(f"Loaded parameters from {path}")  