'''
Reference:
@inproceedings{chai2020makes,
  title={What makes fake images detectable? understanding properties that generalize},
  author={Chai, Lucy and Bau, David and Lim, Ser-Nam and Isola, Phillip},
  booktitle={Computer vision--ECCV 2020: 16th European conference, Glasgow, UK, August 23--28, 2020, proceedings, part XXVI 16},
  pages={103--120},
  year={2020},
  organization={Springer}
}
'''

import torch  
import torch.nn as nn  
import torchvision.models as models  
from .attributor_base import AbstractAttributor  
import torch.nn.functional as F  
from attributors import ATTRIBUTOR
from metrics.base_metrics_class import calculate_metrics_for_train, calculate_metrics_for_test
from backbones.patch_forensics import networks
from utils.dataset_util import ConfigToAttr

@ATTRIBUTOR.register_module(module_name='patch')
class PatchAttributor(AbstractAttributor):  
    def __init__(self, config=None, load_param=False):  
        super().__init__(config, load_param)  
        self.config = config or {}  
        self.config = ConfigToAttr(self.config)
        self.device = "cuda"
        self.gpu_ids = [0]

        self.build_model(self.config)  
        self.build_loss(self.config)  
        if load_param:  
            self.load_parameters(load_param)  

    def build_model(self, config):  
        num_classes = config.get('num_classes', 23)  
        self.net_D = networks.define_patch_D(config.which_model_netD,
                                             config.init_type, self.gpu_ids, num_class=num_classes)

    def build_loss(self, config):  
        self.criterionCE = nn.CrossEntropyLoss().to(self.device)
        self.softmax = torch.nn.Softmax(dim=1)

    def extract_features(self, input_data: dict) -> torch.Tensor:  
        x = input_data['image']  # Tensor (B,C,H,W)  
        features = self.net_D(x)  # (B, 2048)  
        return features  

    def classifier(self, features: torch.Tensor) -> torch.Tensor: 
        return features # FIXME:  features are logits

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
        pred_logit = pred_dict['logits'] 
        labels = input_data['label']
        assert(len(pred_logit.shape) == 4)
        n, c, h, w = pred_logit.shape
        labels = labels.view(-1, 1, 1).expand(n, h, w)
        loss_D = self.criterionCE(pred_logit, labels)



        return {'overall': loss_D}  

    def compute_metrics(self, input_data: dict, pred_dict: dict, test = False) -> dict:  
        label = input_data['label']
        pred = pred_dict['logits']
        n, c, h, w = pred.shape
        label_exp = label.view(-1, 1, 1).expand(n, h, w)
        semantic_label = input_data.get('semantic_label', None)  
        
        # compute metrics for batch data
        # voted acc is forcing each patch into a 0/1 decision,
        # and taking the average
        acc_D_raw = torch.mean(torch.eq(label_exp, torch.argmax(
            pred, dim=1)).float())
        votes = torch.mode(torch.argmax(pred, dim=1).view(n, -1))[0]
        acc_D_voted = torch.mean(torch.eq(label, votes).float())
        avg_preds = torch.argmax(self.softmax(pred)
                                 .mean(dim=(2,3)), dim=1)
        acc_D_avg = torch.mean(torch.eq(label,
                                             avg_preds).float()) 
        pred = self.softmax(pred).mean(dim=(2,3))
        # print(pred.shape)
        if test:
            auc, acc, ap, conf_matrix, semantic_acc= calculate_metrics_for_test(label.detach(), pred.detach(),semantic_label,need_softmax=False)
            metric_batch_dict = {'acc': float(acc), 'auc': float(auc), 'ap': float(ap),
                                 'acc_D_raw': float(acc_D_raw),  'acc_D_voted': float(acc_D_voted), "acc_D_avg": float(acc_D_avg),
                                 "conf_matrix": conf_matrix,
                                 'semantic_acc':semantic_acc # acc of each semantic
                                
                                 }
            return metric_batch_dict
        else:
            # train and val
            auc, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach(),need_softmax=False)
            metric_batch_dict = {'acc': float(acc), 'auc': float(auc), 'ap': float(ap),
                                    'acc_D_raw': float(acc_D_raw),  'acc_D_voted': float(acc_D_voted), "acc_D_avg": float(acc_D_avg),}
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