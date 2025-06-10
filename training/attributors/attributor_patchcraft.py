'''
Reference:
@article{zhong2023patchcraft,
  title={Patchcraft: Exploring texture patch for efficient ai-generated image detection},
  author={Zhong, Nan and Xu, Yiran and Li, Sheng and Qian, Zhenxing and Zhang, Xinpeng},
  journal={arXiv preprint arXiv:2311.12397},
  year={2023}
}
'''

import torch  
import torch.nn as nn  
import torchvision.models as models  
from .attributor_base import AbstractAttributor  
import torch.nn.functional as F  
from attributors import ATTRIBUTOR
from metrics.base_metrics_class import calculate_metrics_for_train, calculate_metrics_for_test

@ATTRIBUTOR.register_module(module_name='patchcraft')
class PatchCraftAttributor(AbstractAttributor):  
    def __init__(self, config=None, load_param=False):  
        super().__init__(config, load_param)  
        self.config = config or {}  

        self.build_model(self.config)  
        self.build_loss(self.config)  

        if load_param:  
            self.load_parameters(load_param)  

    def build_model(self, config):  
        from backbones.patchcraft.filter import FilterLayer 
        num_classes = config.get('num_classes', 23)  
        in_channels = config.get('in_channels', 1)
        out_channels = config.get('out_channels', 32)
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  
        self.batch_norm = nn.BatchNorm2d(out_channels)  
        self.hardtanh = nn.Hardtanh(inplace=True)  
        self.filter = FilterLayer()
        for param in self.filter.parameters():  
            param.requires_grad = False  
        self.nn_classifier = Classifier(num_classes) 

    def build_loss(self, config):  
        self.loss_fn = nn.CrossEntropyLoss()  

    def extract_features(self, input_data: dict) -> torch.Tensor:  
        img1, img2 = input_data["img1"], input_data["img2"]
        filtered_img1 = self.filter(img1)  
        filtered_img2 = self.filter(img2)  

        filtered_img1 = filtered_img1.unsqueeze(1)
        filtered_img2 = filtered_img2.unsqueeze(1)

        out1 = self.conv_layer(filtered_img1)  
        out1 = self.batch_norm(out1)  
        out1 = self.hardtanh(out1)  

        out2 = self.conv_layer(filtered_img2)  
        out2 = self.batch_norm(out2)  
        out2 = self.hardtanh(out2)  

        fingerprint = out1 - out2  

        return fingerprint

    def classifier(self, features: torch.Tensor) -> torch.Tensor:  
        logits = self.nn_classifier(features)  # (B, num_classes)  
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


class Classifier(nn.Module):  
    def __init__(self, num_classes):  
        super(Classifier, self).__init__()  
        self.model = nn.Sequential(  
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),  
            nn.ReLU(),  
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),  
            nn.ReLU(),  
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),  
            nn.ReLU(),  
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),  
            nn.ReLU(),  
            nn.AvgPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),  
            nn.ReLU(),  
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),  
            nn.ReLU(),  
            nn.AvgPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),  
            nn.ReLU(),  
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),  
            nn.ReLU(),  
            nn.AvgPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),  
            nn.ReLU(),  
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),  
            nn.ReLU(),  
            nn.AdaptiveAvgPool2d((1, 1)),  
            nn.Flatten(),  
            nn.Linear(32, num_classes)  
        )  

    def forward(self, x):  
        return self.model(x)  