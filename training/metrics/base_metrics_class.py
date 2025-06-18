import numpy as np
from sklearn import metrics
from collections import defaultdict
import torch
import torch.nn as nn


def get_accracy(output, label):
    _, prediction = torch.max(output, 1)    # argmax
    correct = (prediction == label).sum().item()
    accuracy = correct / prediction.size(0)
    return accuracy


def get_prediction(output, label):
    prob = nn.functional.softmax(output, dim=1)[:, 1]
    prob = prob.view(prob.size(0), 1)
    label = label.view(label.size(0), 1)
    #print(prob.size(), label.size())
    datas = torch.cat((prob, label.float()), dim=1)
    return datas

def calculate_metrics_for_train(label, output, need_softmax=True):  
    num_classes = output.size(1)  
    if need_softmax: 
        prob = torch.softmax(output, dim=1)  
    else:
        prob = output 
    
    _, prediction = torch.max(output, 1)  
    # print(prediction)
    # print(label)
    accuracy = (prediction == label).float().mean().item()  

    y_true = label.cpu().numpy()  
    y_pred = prob.cpu().numpy()  

    y_true_onehot = np.eye(num_classes)[y_true]  

    aucs = []  
    aps = []  

    for class_i in range(num_classes):  
        y_true_cls = y_true_onehot[:, class_i]  
        y_pred_cls = y_pred[:, class_i]  

        # 判断该类别是否包含正负样本  
        if len(np.unique(y_true_cls)) < 2:  
            # 该类别只有单一类标签，跳过计算  
            continue  

        try:  
            auc = metrics.roc_auc_score(y_true_cls, y_pred_cls)  
            ap = metrics.average_precision_score(y_true_cls, y_pred_cls)  
        except Exception:  
            continue  

        aucs.append(auc)  
        aps.append(ap)  

    mean_auc = np.mean(aucs) if aucs else float('nan')  
    mean_ap = np.mean(aps) if aps else float('nan')  

    return mean_auc, accuracy, mean_ap  
from sklearn import metrics
import numpy as np
import torch
def calculate_metrics_for_test(label, output, semantic_label, need_softmax=True):  
    num_classes = output.size(1) 
    if need_softmax: 
        prob = torch.softmax(output, dim=1)  
    else:
        prob = output
    _, prediction = torch.max(output, 1)  
    accuracy = (prediction == label).float().mean().item()  

    y_true = label.cpu().numpy()  
    y_pred = prob.cpu().numpy()  
    y_pred_label = prediction.cpu().numpy()  
    semantic_labels_np = semantic_label.cpu().numpy()  

    # One-hot encode true labels  
    y_true_onehot = np.eye(num_classes)[y_true]  

    aucs = []  
    aps = []  

    for class_i in range(num_classes):  
        y_true_cls = y_true_onehot[:, class_i]  
        y_pred_cls = y_pred[:, class_i]  

        if len(np.unique(y_true_cls)) < 2:  
            continue  

        try:  
            auc = metrics.roc_auc_score(y_true_cls, y_pred_cls)  
            ap = metrics.average_precision_score(y_true_cls, y_pred_cls)  
        except Exception:  
            continue  

        aucs.append(auc)  
        aps.append(ap)  

    mean_auc = np.mean(aucs) if aucs else float('nan')  
    mean_ap = np.mean(aps) if aps else float('nan')  

    # 计算混淆矩阵（以预测标签为准）  
    conf_matrix = metrics.confusion_matrix(y_true, y_pred_label)  

    # 根据semantic_label分组计算分类准确率  
    unique_semantics = np.unique(semantic_labels_np)  
    semantic_acc = {}  
    for sem in unique_semantics:  
        idx = (semantic_labels_np == sem)  
        if np.sum(idx) == 0:  
            continue  
        group_acc = np.mean(y_pred_label[idx] == y_true[idx])  
        semantic_acc[int(sem)] = float(group_acc)  
    return mean_auc, accuracy, mean_ap, conf_matrix, semantic_acc 
# ------------ compute average metrics of batches---------------------
class Metrics_batch():
    def __init__(self):
        self.tprs = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.aucs = []
        self.eers = []
        self.aps = []

        self.correct = 0
        self.total = 0
        self.losses = []

    def update(self, label, output):
        acc = self._update_acc(label, output)
        if output.size(1) == 2:
            prob = torch.softmax(output, dim=1)[:, 1]
        else:
            prob = output
        #label = 1-label
        #prob = torch.softmax(output, dim=1)[:, 1]
        auc, eer = self._update_auc(label, prob)
        ap = self._update_ap(label, prob)

        return acc, auc, eer, ap

    def _update_auc(self, lab, prob):
        fpr, tpr, thresholds = metrics.roc_curve(lab.squeeze().cpu().numpy(),
                                                 prob.squeeze().cpu().numpy(),
                                                 pos_label=1)
        if np.isnan(fpr[0]) or np.isnan(tpr[0]):
            return -1, -1

        auc = metrics.auc(fpr, tpr)
        interp_tpr = np.interp(self.mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        self.tprs.append(interp_tpr)
        self.aucs.append(auc)

        # return auc

        # EER
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        self.eers.append(eer)

        return auc, eer

    def _update_acc(self, lab, output):
        _, prediction = torch.max(output, 1)    # argmax
        correct = (prediction == lab).sum().item()
        accuracy = correct / prediction.size(0)
        # self.accs.append(accuracy)
        self.correct = self.correct+correct
        self.total = self.total+lab.size(0)
        return accuracy

    def _update_ap(self, label, prob):
        y_true = label.cpu().detach().numpy()
        y_pred = prob.cpu().detach().numpy()
        ap = metrics.average_precision_score(y_true,y_pred)
        self.aps.append(ap)

        return np.mean(ap)

    def get_mean_metrics(self):
        mean_acc, std_acc = self.correct/self.total, 0
        mean_auc, std_auc = self._mean_auc()
        mean_err, std_err = np.mean(self.eers), np.std(self.eers)
        mean_ap, std_ap = np.mean(self.aps), np.std(self.aps)
        
        return {'acc':mean_acc, 'auc':mean_auc, 'eer':mean_err, 'ap':mean_ap}

    def _mean_auc(self):
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        return mean_auc, std_auc

    def clear(self):
        self.tprs.clear()
        self.aucs.clear()
        # self.accs.clear()
        self.correct=0
        self.total=0
        self.eers.clear()
        self.aps.clear()
        self.losses.clear()


# ------------ compute average metrics of all data ---------------------
class Metrics_all():
    def __init__(self):
        self.probs = []
        self.labels = []
        self.correct = 0
        self.total = 0

    def store(self, label, output):
        prob = torch.softmax(output, dim=1)[:, 1]
        _, prediction = torch.max(output, 1)    # argmax
        correct = (prediction == label).sum().item()
        self.correct += correct
        self.total += label.size(0)
        self.labels.append(label.squeeze().cpu().numpy())
        self.probs.append(prob.squeeze().cpu().numpy())

    def get_metrics(self):
        y_pred = np.concatenate(self.probs)
        y_true = np.concatenate(self.labels)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true,y_pred,pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true,y_pred)
        # acc
        acc = self.correct / self.total
        return {'acc':acc, 'auc':auc, 'eer':eer, 'ap':ap}

    def clear(self):
        self.probs.clear()
        self.labels.clear()
        self.correct = 0
        self.total = 0


# only used to record a series of scalar value
class Recorder:
    def __init__(self):
        self.sum = 0
        self.num = 0
    def update(self, item, num=1):
        if item is not None:
            self.sum += item * num
            self.num += num
    def average(self):
        if self.num == 0:
            return None
        return self.sum/self.num
    def clear(self):
        self.sum = 0
        self.num = 0
