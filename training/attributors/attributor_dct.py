'''
Reference:
@inproceedings{frank2020leveraging,
  title={Leveraging frequency analysis for deep fake image recognition},
  author={Frank, Joel and Eisenhofer, Thorsten and Sch{\"o}nherr, Lea and Fischer, Asja and Kolossa, Dorothea and Holz, Thorsten},
  booktitle={International conference on machine learning},
  pages={3247--3258},
  year={2020},
  organization={PMLR}
}
'''

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from .attributor_base import AbstractAttributor  
from attributors import ATTRIBUTOR  
from metrics.base_metrics_class import calculate_metrics_for_train, calculate_metrics_for_test  
from scipy.fftpack import dct
from backbones.dct.simple_cnn import SimpleCNN

@ATTRIBUTOR.register_module(module_name='dct')
class DCTAttributor(AbstractAttributor):  
    def __init__(self, config=None, load_param=False):  
        super().__init__(config, load_param)  
        self.config = config or {}  

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.build_model(self.config)   
        self.build_loss(self.config)

    def build_model(self, config):  
        self.num_classes = config.get('num_classes', 23)
        self.input_channels = config.get('input_channels', 3)  
        self.input_size = config.get('input_size', (256, 256))  
        
        self.feature_extractor = SimpleCNN(self.input_channels, self.num_classes, self.input_size)

    def build_loss(self, config):  
        self.loss_fn = nn.CrossEntropyLoss()



    def extract_features(self, input_data: dict) -> torch.Tensor:   
        images = input_data['image']
        feature = self.feature_extractor.extract_feature(images)
        return feature

    def classifier(self, features: torch.Tensor) -> torch.Tensor:  
        logits = self.feature_extractor.classify(features)
        return logits

    def forward(self, input_data: dict, inference=False) -> dict:  
        features = self.extract_features(input_data)
        logits = self.classifier(features)

        outputs = {}
        outputs['logits'] = logits
        outputs['pred'] = torch.argmax(logits, dim=1)
        return outputs

    def compute_losses(self, input_data: dict, pred_dict: dict) -> dict:  
        target = input_data['label'].long()
        logits = pred_dict['logits']
        loss = self.loss_fn(logits, target)
        return {'overall': loss}

    def compute_metrics(self, input_data: dict, pred_dict: dict, test=False) -> dict:  
        label = input_data['label']
        semantic_label = input_data.get('semantic_label', None)  
        pred = pred_dict['logits']

        if isinstance(label, torch.Tensor):
            label = label.detach().cpu()
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu()

        if test:
            auc, acc, ap, conf_matrix, semantic_acc = calculate_metrics_for_test(label, pred, semantic_label)
            return {'acc': float(acc), 'auc': float(auc), 'ap': float(ap),
                    "conf_matrix": conf_matrix,
                    'semantic_acc': semantic_acc}
        else:
            auc, acc, ap = calculate_metrics_for_train(label, pred)
            return {'acc': float(acc), 'auc': float(auc), 'ap': float(ap)}

    def load_parameters(self, load_param):  
        pass 