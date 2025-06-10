'''
Reference:
@article{yang2021learning,
  title={Learning to disentangle gan fingerprint for fake image attribution},
  author={Yang, Tianyun and Cao, Juan and Sheng, Qiang and Li, Lei and Ji, Jiaqi and Li, Xirong and Tang, Sheng},
  journal={arXiv preprint arXiv:2106.08749},
  year={2021}
}
'''

import torch  
import torch.nn as nn  
import torchvision.models as models  
from .attributor_base import AbstractAttributor  
import torch.nn.functional as F  
from attributors import ATTRIBUTOR
from metrics.base_metrics_class import calculate_metrics_for_train, calculate_metrics_for_test
from backbones.gfd.CD import NLayerDiscriminator,AuxiliaryClassifier
from backbones.gfd.G import UNetGenerator
from loss.gfd.loss import CombinedLoss

@ATTRIBUTOR.register_module(module_name='gfd')
class GFDAttributor(AbstractAttributor):  
    def __init__(self, config=None, load_param=False):  
        super().__init__(config, load_param)  
        self.config = config or {}  
        self.device = "cuda"

        self.build_model(self.config)  
        self.build_loss(self.config)  

        if load_param:  
            self.load_parameters(load_param)  

    def set_stage_1(self):
        for p in self.generator.parameters():
            p.requires_grad = True
        for p in self.discriminator.parameters():
            p.requires_grad = False
        for p in self.auxiliary_classifier.parameters():
            p.requires_grad = False

    def set_stage_2(self):
        for p in self.generator.parameters():
            p.requires_grad = False
        for p in self.discriminator.parameters():
            p.requires_grad = True
        for p in self.auxiliary_classifier.parameters():
            p.requires_grad = True


    def build_model(self, config):  
        in_channels = config.get("in_channels", 3)
        base_channels = config.get("base_channels", 64)
        num_classes = config.get("num_classes", 23)
        self.generator = UNetGenerator(in_channels, base_channels, num_classes) 
        self.discriminator = NLayerDiscriminator()  
        self.auxiliary_classifier = AuxiliaryClassifier(num_classes) 

    def build_loss(self, config):  
        self.loss_fn = CombinedLoss(self.device)  

    def extract_features(self, input_data: dict) -> torch.Tensor:  
        pass 

    def classifier(self, features: torch.Tensor) -> torch.Tensor:  
        pass

    def forward(self, input_data, stage_no=0, inference=False):  
        if not inference:
            if stage_no == 1:
                self.set_stage_1()
            elif stage_no == 2:
                self.set_stage_2()
            real_image, fake_image = input_data['real_img'], input_data['fake_img']
            gen_image, fake_pred_logits = self.generator(fake_image)  
            real_pred_logits = self.generator.get_classifer_logits(real_image)  
            
            fingerprinted_image = real_image + gen_image   

            D_real = self.discriminator(real_image)  
            D_fingerprint = self.discriminator(fingerprinted_image) 

            auxiliary_prediction_xf = self.auxiliary_classifier(fingerprinted_image)  
            auxiliary_prediction_xy = self.auxiliary_classifier(fake_image)  

            out = {'logits': fake_pred_logits, 
                #    'features': features,
                'real_pred_prob': real_pred_logits,
                'fake_pred_prob': fake_pred_logits,
                'fingerprinted_image': fingerprinted_image,
                'D_real': D_real,
                'D_fingerprint': D_fingerprint,
                'auxiliary_prediction_xf': auxiliary_prediction_xf,
                'auxiliary_prediction_xy': auxiliary_prediction_xy
                }   

        if inference:  
            image = input_data['image']
            pred_logits = self.generator.get_classifer_logits(image)
            out = {"logits":pred_logits,
                   "probs": torch.softmax(pred_logits,dim=1)}

        return out  

    def compute_losses(self, input_data: dict, pred_dict: dict, stage_no: int) -> dict:  
        real_images = input_data['real_img']
        real_labels = input_data['real_label']
        gen_labels = input_data['fake_label']

        fingerprinted_image = pred_dict['fingerprinted_image']
        real_pred_prob = pred_dict['real_pred_prob']
        fake_pred_prob = pred_dict['fake_pred_prob']
        D_real = pred_dict['D_real']
        D_fingerprint = pred_dict['D_fingerprint']
        auxiliary_prediction_xy = pred_dict['auxiliary_prediction_xy']
        auxiliary_prediction_xf = pred_dict['auxiliary_prediction_xf']
        if stage_no == 1: # train generator
            loss = self.loss_fn.calculate_generator_loss(real_images, fingerprinted_image, real_labels, gen_labels, real_pred_prob, fake_pred_prob, D_real, D_fingerprint, auxiliary_prediction_xy)
        elif stage_no == 2: # train discriminator and classifier
            loss = self.loss_fn.calculate_discriminator_loss(D_real, D_fingerprint, auxiliary_prediction_xf, gen_labels)    
        return loss 

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
        pass