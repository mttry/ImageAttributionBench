'''
Reference:
@inproceedings{bui2022repmix,
  title={Repmix: Representation mixing for robust attribution of synthesized images},
  author={Bui, Tu and Yu, Ning and Collomosse, John},
  booktitle={European Conference on Computer Vision},
  pages={146--163},
  year={2022},
  organization={Springer}
}
'''

import abc  
from typing import Union  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import torchvision  
from backbones.repmix import torch_layers as tl  
from .attributor_base import AbstractAttributor  
from attributors import ATTRIBUTOR  
from metrics.base_metrics_class import calculate_metrics_for_train, calculate_metrics_for_test

@ATTRIBUTOR.register_module(module_name='repmix')
class RepmixAttributor(AbstractAttributor):  
    def __init__(self, config=None, load_param: Union[bool, str] = False):  
        super().__init__()  
        default_config = {  
            'd_embed': 256,  
            'num_classes': 23,  
            'mixup_samples': 2,  
            'mixup_beta': 0.4,  
            'mixup_level': 5,  
            'inference': False,  
            'img_mean': [0.5, 0.5, 0.5],  
            'img_std': [0.5, 0.5, 0.5],  
            'img_rsize': 256,  
            'img_csize': 224,  
            'pertubation': True,  
        }  
        self.config = default_config  
        if config is not None:  
            self.config.update(config)  
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.build_model(self.config)  
        self.build_loss(self.config)  
        
    
    def build_model(self, config):  
        kwargs = {  
            'mix_level': config['mixup_level'],  
            'nmix': config['mixup_samples'],  
            'beta': config['mixup_beta'],  
            'inference': config['inference'],  
        }  
        self.backbone = ResnetMixup(config['d_embed'], **kwargs).to(self.device)  
        d_embed = config['d_embed']  
        num_classes = config['num_classes']  
        self.attribution = nn.Linear(d_embed, num_classes).to(self.device) 
        self.detection = nn.Linear(d_embed, 2).to(self.device)  

    def build_loss(self, config):  
        num_classes = config['num_classes']  
        self.attr_ce = nn.CrossEntropyLoss(reduction='none')  
        self.act = nn.Softmax(dim=1)  
        self.det_ce = nn.CrossEntropyLoss(torch.tensor([1, 1.0 / (num_classes - 1)], device=self.device), reduction='none')  

    def extract_features(self,  input_data: dict, inference=False) -> torch.Tensor:  
        self.backbone.inference = inference  
        x = input_data['x'].to(self.device)  
        beta = input_data.get('beta', None)
        feats = self.backbone(x, beta)  
        return feats  

    def classifier(self, features: torch.Tensor) -> torch.Tensor:  
        attribution = self.attribution(features)  
        detection = self.detection(features)  
        detection_prob = self.act(detection)  
        attribution0 = attribution[:, 0] * detection_prob[:, 0]  
        attribution1 = attribution[:, 1:] * detection_prob[:, 1].unsqueeze(1)  
        out_attribution = torch.cat([attribution0.unsqueeze(1), attribution1], dim=1)  
        return out_attribution, detection  

    def forward(self, input_data: dict, inference=False) -> dict:   
        features = self.extract_features(input_data, inference)  
        attribution, detection = self.classifier(features)  
        return {'embedding': features, 'attribution': attribution, 'detection': detection,
        "logits": attribution}  

    def compute_losses(self, input_data: dict, pred_dict: dict) -> dict:  
        y_gan = input_data['y_gan'].to(self.device)  
        y_det = input_data['y_det'].to(self.device)  
        beta = input_data['beta'].to(self.device)  
        out = {}  
        if self.config['inference']:  
            out['attribution_loss'] = torch.mean(self.attr_ce(pred_dict['attribution'], y_gan))  
            out['detection_loss'] = torch.mean(self.det_ce(pred_dict['detection'], y_det))  
        else:  
            mnum = self.config['mixup_samples']  
            if mnum == 1:  
                y_gan = y_gan.unsqueeze(1)  
                beta = beta.unsqueeze(1)  
                y_det = y_det.unsqueeze(1)  
            out['attribution_loss'] = torch.mean(  
                sum([self.attr_ce(pred_dict['attribution'], y_gan[:, i]) * beta[:, i] for i in range(mnum)])  
            )  
            out['detection_loss'] = torch.mean(  
                sum([self.det_ce(pred_dict['detection'], y_det[:, i]) * beta[:, i] for i in range(mnum)])  
            )  
        out['overall'] = sum(out.values())  
        return out  

    def compute_metrics(self, input_data: dict, pred_dict: dict, test=False) -> dict:  
        pred = F.softmax(pred_dict['logits'], dim=1)  
        label = input_data['label'][:,0]  
        semantic_label = input_data.get('semantic_label', None)
        semantic_label = semantic_label[:,0] if semantic_label is not None else None
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


class ResnetMixup(nn.Module):  
    def __init__(self, d_embed, mix_level=0, nmix=2, beta=0.4, inference=False):  
        super().__init__()  
        self.inference = inference  
        model = torchvision.models.resnet50(pretrained=True, progress=False)  
        model.fc = nn.Linear(model.fc.in_features, d_embed)  
        model = list(model.children())  
        model = [nn.Sequential(*model[:4])] + model[4:-2] + [nn.Sequential(model[-2], nn.Flatten(1), model[-1])]  
        assert mix_level <= len(model)  
        mx_layer = tl.MixupLayer(nmix, beta)  
        model.insert(mix_level, mx_layer)  
        self.mix_level = mix_level  
        self.model = nn.ModuleList(model)  

    def forward(self, x, ratio=None):  
        for i, layer in enumerate(self.model):  
            if i == self.mix_level:  
                if not self.inference:  
                    x = layer(x, ratio)  
            else:  
                x = layer(x)  
        return x  