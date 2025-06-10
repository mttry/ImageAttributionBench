'''
Reference:
@inproceedings{yang2023progressive,
  title={Progressive open space expansion for open-set model attribution},
  author={Yang, Tianyun and Wang, Danding and Tang, Fan and Zhao, Xinying and Cao, Juan and Tang, Sheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15856--15865},
  year={2023}
}
'''
# FIXME: Training failed due to poor fitting.

from collections import defaultdict
import os
import torch  
import torch.nn as nn  
import torchvision.models as models  
from .attributor_base import AbstractAttributor  
import torch.nn.functional as F  
from attributors import ATTRIBUTOR
from metrics.base_metrics_class import calculate_metrics_for_train, calculate_metrics_for_test
from backbones.pose.augment_network import SingleLayer
from utils.pose.logger import Progbar, AverageMeter
from utils.pose.common import plot_ROC_curve, plot_hist, tsne_analyze, set_requires_grad
from loss.pose.loss import TripletLoss
from backbones.pose.models import Simple_CNN
import numpy as np
import time
from metrics.registry import LOSSFUNC
from utils.dataset_util import ConfigToAttr

@ATTRIBUTOR.register_module(module_name='pose')
class POSEAttributor(AbstractAttributor):  
    def __init__(self, config=None, load_param=False):  
        super().__init__(config, load_param)  
        self.config = config or {}  
        self.device = "cuda"
        self.config = ConfigToAttr(self.config)
        self.build_model(self.config)  
        self.build_loss(self.config)  

        if load_param:  
            self.load_parameters(load_param)  

    def build_model(self, config):  
        num_classes = config.get('class_num', 23)  
        self.augnets = []
        self.model=Simple_CNN(class_num=num_classes, out_feature_result=True).to(self.device)
        self.optimizer = torch.optim.Adam([{'params':self.model.parameters(), 'lr':config.init_lr_E},])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.step_size, gamma=config.gamma)

    def build_loss(self, config):  
        self.criterionMSE = torch.nn.MSELoss()
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.criterionMetric = TripletLoss()
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def extract_features(self, input_data: dict) -> torch.Tensor:  
        x = input_data['image']  # Tensor (B,C,H,W)  
        cls_out, features = self.model(x)  # (B, 2048)  
        return cls_out, features  

    def classifier(self, features: torch.Tensor) -> torch.Tensor:  
        pass

    def train_epoch_POSE_backup(self, epoch, train_loader, optimizer, scheduler, logger):
        # for i, param_group in enumerate(optimizer.param_groups):
        #     print(f"Param group {i}:")
        # for j, param in enumerate(param_group['params']):
        #     print(f"  Param {j} - shape: {param.shape}, requires_grad: {param.requires_grad}")
        logger.info(f"===> Epoch[{epoch}] start!")  
        self.model.train()  

        train_loss_recorder = defaultdict(float)  
        train_metric_recorder = defaultdict(float)  
        val_metric = None  
        val_metrics = None  
        num_batches = len(train_loader)  
        global_step = epoch * num_batches 

        augnet = SingleLayer(inc=self.config.inc, kernel_size=self.config.kernel_size).to(self.device)
        optimizerA = torch.optim.Adam([{'params':augnet.parameters(), 'lr':self.config.augnet_lr}])
        schedulerA = torch.optim.lr_scheduler.StepLR(optimizerA, step_size=self.config.step_size, gamma=self.config.gamma)

        progbar = Progbar(len(train_loader), stateful_metrics=['epoch'])
        batch_time = AverageMeter()
        end = time.time()
        for batch_idx, batch in enumerate(train_loader):
            i = batch_idx
            input_img_batch, label_batch = batch["image"], batch["label"]
            input_img = input_img_batch.reshape((-1, 3, input_img_batch.size(-2), input_img_batch.size(-1))).to(self.device)
            label = label_batch.reshape((-1)).to(self.device)

            # -------- step1 :train augnet start -------- 
            optimizerA.zero_grad()
            set_requires_grad([augnet], True)
            set_requires_grad([self.model], False)

            # reconstruction loss
            aug_img = augnet(input_img.detach())
            loss_mse = self.criterionMSE(aug_img, input_img)
            loss_A = torch.clamp(loss_mse, self.config.mse_lowbound)
            train_loss_recorder.update({'loss_mse': loss_A.item()})

            _, aug_fea = self.model(aug_img, data=self.config.input_data)
            input_prob, input_fea = self.model(input_img, data=self.config.input_data)

            if len(self.augnets) >= 1:
                # close to known samples
                loss_close_known = torch.clamp(1 - torch.mean(self.cos(input_fea, aug_fea)), 1 - self.config.known_sim_limit) # larger similarity 
                train_loss_recorder.update({'loss_close_known': loss_close_known.item()})
                loss_A += self.config.w_close_known * loss_close_known

                # distant from previous augnets
                idx = np.random.randint(0, len(self.augnets))
                aug_img_pre = self.augnets[idx](input_img.detach())
                _, aug_fea_pre = self.model(aug_img_pre, data=self.config.input_data)
                loss_distant_preaug = torch.mean(self.cos(aug_fea, aug_fea_pre)) # smaller similarity 
                train_loss_recorder.update({'loss_distant_preaug': loss_distant_preaug.item()})
                loss_A += self.config.w_dist_pre * loss_distant_preaug

            loss_A.backward()
            optimizerA.step()
            schedulerA.step() 
            # -------- train augnet end -------- 

            # -------- step2 :train classifier start -------- 
            self.optimizer.zero_grad()
            set_requires_grad([self.model], True)
            set_requires_grad([augnet], False)
            
            input_prob, input_fea = self.model(input_img, data=self.config.input_data)
            # print(label)
            loss_cls = self.criterion(input_prob, label)

            # get augnet data and labels
            aug_img = augnet(input_img) 
            augnet_label = self.config.class_num + label    
            augnet_label = augnet_label.to(self.device)            
            _, aug_fea = self.model(aug_img, data=self.config.input_data)
            merged_label = torch.cat([label, augnet_label])
            merged_fea = torch.cat([input_fea, aug_fea])

            # metric loss
            loss_metric = self.criterionMetric(merged_fea, merged_label)    
            train_loss_recorder.update({'loss_cls':loss_cls.item(), 'loss_metric':loss_metric.item()})        
            loss = loss_cls + loss_metric

            # classification on previous aug data
            if self.config.cls_pre and epoch > self.config.start_cls_pre_epoch:
                idx = np.random.randint(0, len(self.augnets))
                aug_img_pre = self.augnets[idx](input_img)
                augnet_label_pre = self.config.class_num + label  
                _, aug_fea_pre = self.model(aug_img_pre, data=self.config.input_data)
                merged_label = torch.cat([label, augnet_label_pre])
                merged_fea = torch.cat([input_fea, aug_fea_pre])
                loss_metric_pre = self.criterionMetric(merged_fea, merged_label) 
                train_loss_recorder.update({'loss_metric_pre':loss_metric_pre.item()})
                loss += loss_metric_pre

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()                 
            # -------- train classifier end -------- 

            # -------- log and visualize -------- 
            progbar.add(1,values=[('epoch', epoch)]+[(loss_key,train_loss_recorder[loss_key]) for loss_key in train_loss_recorder.keys()]+[('lr', scheduler.get_lr()[0])])
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 300 == 0:   
                loss_str = " | ".join([f"{k}: {train_loss_recorder[k]/(i+1):.4f}" for k in train_loss_recorder])  
                metric_str = " | ".join([f"{k}: {train_metric_recorder[k]/(i+1):.4f}" for k in train_metric_recorder])  
                logger.info(f"Step {global_step} - Losses: {loss_str}")  
                logger.info(f"Step {global_step} - Metrics: {metric_str}")

        self.augnets.append(augnet)

    def train_epoch_POSE(self, epoch, train_loader, optimizer, scheduler, logger):
        logger.info(f"===> Epoch[{epoch}] start!")  
        self.model.train()  

        train_loss_recorder = defaultdict(float)  
        train_metric_recorder = defaultdict(float)  

        num_batches = len(train_loader)  
        global_step = epoch * num_batches 

        progbar = Progbar(num_batches, stateful_metrics=['epoch'])
        batch_time = AverageMeter()
        end = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_img_batch, label_batch = batch["image"], batch["label"]
            input_img = input_img_batch.reshape((-1, 3, input_img_batch.size(-2), input_img_batch.size(-1))).to(self.device)
            label = label_batch.reshape((-1)).to(self.device)

            optimizer.zero_grad()
            set_requires_grad([self.model], True) 

            input_prob, input_fea = self.model(input_img, data=self.config.input_data)

            loss_cls = self.criterion(input_prob, label)

            loss = loss_cls

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_recorder['loss_cls'] += loss_cls.item()

            progbar.add(1, values=[('epoch', epoch)] + [(k, train_loss_recorder[k]/(batch_idx+1)) for k in train_loss_recorder.keys()] + [('lr', scheduler.get_last_lr()[0])])

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 300 == 0:
                loss_str = " | ".join([f"{k}: {train_loss_recorder[k]/(batch_idx+1):.4f}" for k in train_loss_recorder])
                logger.info(f"Step {global_step} - Losses: {loss_str}")

        return train_loss_recorder
    def forward(self, input_data: dict, inference=False) -> dict:  
        cls_out, features = self.extract_features(input_data)  

        out = {'logits': cls_out, 
               'features': features,
               'cls': torch.argmax(cls_out, dim=1)  }  

        if inference:  
            probs = F.softmax(cls_out, dim=1)  
            out['probs'] = probs  

        return out  

    def compute_losses(self, input_data: dict, pred_dict: dict) -> dict:  
        pass 

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