'''
Reference:
@inproceedings{yang2022deepfake,
  title={Deepfake network architecture attribution},
  author={Yang, Tianyun and Huang, Ziyao and Cao, Juan and Li, Lei and Li, Xirong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={4},
  pages={4662--4670},
  year={2022}
}
'''

import os
import torch  
import torch.nn as nn  
import torchvision.models as models  
from .attributor_base import AbstractAttributor  
import torch.nn.functional as F  
from attributors import ATTRIBUTOR
from metrics.base_metrics_class import calculate_metrics_for_train, calculate_metrics_for_test
from utils.dataset_util import ConfigToAttr
@ATTRIBUTOR.register_module(module_name='dna')
class DNAAttributor(AbstractAttributor):  
    def __init__(self, config=None, load_param=False):  
        super().__init__(config, load_param)  
        self.config = config or {}  
        self.config = ConfigToAttr(self.config)
        self.device = "cuda"
        self.build_model(self.config)  
        self.build_loss(self.config)  
        load_param = config.get("load_param", False)
        if load_param:  
            self.load_parameters(config.get("pretrained_path", None))  

    def build_model(self, config):  
        from backbones.dna.models import Simple_CNN
        num_classes = config.get('num_classes', 23)  
        self.backbone = Simple_CNN(num_classes)

        # feature head
        head = self.config.get("head", "mlp")
        dim_in = self.config.get("dim_in",512)
        feat_dim = self.config.get("feat_dim",128)
        if head=='linear':
            self.head=nn.Linear(dim_in, feat_dim)
        elif head=='mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def build_loss(self, config):  
        from loss.dna.loss import AutomaticWeightedLoss, SupConLoss
        self.awl = AutomaticWeightedLoss(2).to(self.device)  
        self.criterionCE = nn.CrossEntropyLoss()  
        self.criterionCon = SupConLoss(temperature=config.temperature)  

    def extract_features(self, input_data):  
        cls_out, embedding = self.backbone(input_data)
        feat = self.backbone.pool(embedding)
        feat = feat.view(feat.shape[0], -1)
        feat = F.normalize(self.head(feat), dim=1)
        return cls_out, feat
         

    def classifier(self, features: torch.Tensor) -> torch.Tensor:  
        cls_out, embedding = self.backbone(features)
        return cls_out  

    def forward(self, input_data: dict, inference=False) -> dict: 
        if not inference:
            crops_batch, label_batch = input_data['crops'], input_data['label']
            crops = [crop_batch.reshape((-1, 3, crop_batch.size(-2), crop_batch.size(-1))).to(self.device) for crop_batch in crops_batch]
            labels = label_batch.reshape((-1)).to(self.device)
            
            # predict on crops
            crops_result = [self.extract_features(crop) for crop in crops]
            # classification probs on crops
            crops_prob = torch.cat([result[0] for result in crops_result], dim=0)
            crops_prob_stack = torch.stack([result[0] for result in crops_result])
            prob = torch.mean(crops_prob_stack, dim=0)
            # features on crops
            crops_feat = torch.cat([result[1].unsqueeze(1) for result in crops_result], dim=1)
            out = {
                'logits': prob, 
                'crop_logits': crops_prob, 
                'features': crops_feat,
                'cls': torch.argmax(crops_prob, dim=1)
            }  

        if inference:  
            input_img_batch, label_batch = input_data['image'], input_data['label']
            input_img = input_img_batch.reshape((-1, 3, input_img_batch.size(-2), input_img_batch.size(-1))).to(self.device)
            prob, _ = self.extract_features(input_img)
            out = {
                'logits': prob, 
                'cls': torch.argmax(prob, dim=1)
            }    

        return out  

    def compute_losses(self, input_data: dict, pred_dict: dict) -> dict:  
        labels = input_data['label']
        crops_label = torch.cat([labels]*len(self.config.multi_size), dim=0) 
        crops_prob = pred_dict['crop_logits']  
        crops_feat = pred_dict['features']  
         # calculate classification loss
        loss_cls = self.criterionCE(crops_prob, crops_label)
        # calculate contrastive loss
        loss_contra = self.criterionCon(crops_feat, labels)
        # calculate total loss
        loss_total = self.awl(loss_cls, loss_contra)
        return {'overall': loss_total, 'loss_cls': loss_cls.item(), 'loss_contra': loss_contra.item()}  

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

    def load_parameters(self, path):

        if path is not None and os.path.exists(path):
            print(f'Resuming from pretrained model {path}')
            netE_dict = self.backbone.state_dict()
            
            pretrained_full = torch.load(path, map_location='cpu')
            
            pretrained_full = pretrained_full['model_state_dict']
        
            pretrained_dict = {}
            for k, v in pretrained_full.items():
                if k.startswith('backbone.'):
                    new_key = k[len('backbone.'):]
                else:
                    new_key = k
                pretrained_dict[new_key] = v
            
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in netE_dict and netE_dict[k].shape == v.shape}
            
            netE_dict.update(pretrained_dict)
            
            self.backbone.load_state_dict(netE_dict)
            print(f"Loaded {len(pretrained_dict)} parameters into backbone.")
        else:
            print("Checkpoint path not found or None, no weights loaded.")