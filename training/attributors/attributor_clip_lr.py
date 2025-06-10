'''
Reference:
@article{cioni2024clip,
  title={Are CLIP features all you need for Universal Synthetic Image Origin Attribution?},
  author={Cioni, Dario and Tzelepis, Christos and Seidenari, Lorenzo and Patras, Ioannis},
  journal={arXiv preprint arXiv:2408.09153},
  year={2024}
}
'''

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np  
from transformers import CLIPModel  
from sklearn.linear_model import LogisticRegression  
from sklearn.preprocessing import StandardScaler  
from .attributor_base import AbstractAttributor  
from attributors import ATTRIBUTOR  
from metrics.base_metrics_class import calculate_metrics_for_train, calculate_metrics_for_test  


@ATTRIBUTOR.register_module(module_name='clip_lr')  
class ClipLrAttributor(AbstractAttributor):  
    def __init__(self, config=None, load_param=False):  
        super().__init__(config, load_param)  
        self.config = config or {}  

        self.device = "cuda" if torch.cuda.is_available() else "cpu"  

        self.build_model(self.config)  
        self.build_loss(self.config)  

        if load_param:  
            self.load_parameters(load_param)  

    def build_model(self, config=None):  
        self.clip_model = CLIPModel.from_pretrained(  
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" ,
            local_files_only=True  
        ).to(self.device)  
        self.clip_model.eval()  

        self.scaler = StandardScaler()  
        solver = self.config.get('lr_solver', 'lbfgs')  
        penalty = self.config.get('lr_penalty', 'l2')  
        max_iter = self.config.get('lr_max_iter', 1000)  
        multi_class = self.config.get('lr_multi_class', 'auto')  

        self.linear_probe = LogisticRegression(  
            solver=solver,  
            penalty=penalty,  
            max_iter=max_iter,  
            multi_class=multi_class  
        )  

    def build_loss(self, config=None):  
        self.loss_fn = nn.CrossEntropyLoss()  

    def extract_features(self, input_data: dict) -> torch.Tensor:  
        images = input_data['image'].to(self.device)  
        with torch.no_grad():  
            features = self.clip_model.get_image_features(images)  
        return features  

    def train_linear_probe(self, train_loader):  
        from tqdm import tqdm  
        feature_list = []  
        label_list = []  
        for batch in tqdm(train_loader, desc="Training linear probe"):  
            images = batch["image"].to(self.device)  
            labels = batch["label"].numpy()  

            with torch.no_grad():  
                feats = self.clip_model.get_image_features(images)  

            feature_list.append(feats.cpu().numpy())  
            label_list.extend(labels)  

        features = np.concatenate(feature_list, axis=0)  
        labels = np.array(label_list)  

        features = self.scaler.fit_transform(features)  

        self.linear_probe.fit(features, labels)  
        print("Logistic Regression training completed.")    

    def classifier(self, features: torch.Tensor) -> torch.Tensor:  
        features_np = features.detach().cpu().numpy()  
        features_np = self.scaler.transform(features_np)  
        probs_np = self.linear_probe.predict_proba(features_np)  
        probs_tensor = torch.from_numpy(probs_np).to(features.device).float()  
        return probs_tensor  

    def forward(self, input_data: dict, inference=False) -> dict:  
        images = input_data['image'].to(self.device)  
        with torch.no_grad():
            features = self.clip_model.get_image_features(images)  
        logits = self.classifier(features)  

        out = {  
            'logits': logits,  
            'features': features,  
            'cls': torch.argmax(logits, dim=1),  
        }  

        if inference:  
            out['probs'] = F.softmax(logits, dim=1)  

        return out  

    def compute_losses(self, input_data: dict, pred_dict: dict) -> dict:  
        labels = input_data['label'].to(self.device)  
        logits = pred_dict['logits']  
        loss = self.loss_fn(logits, labels)  
        return {'overall': loss}  

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