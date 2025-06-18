import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics.registry import LOSSFUNC

@LOSSFUNC.register_module(module_name="Softmax")
class Softmax(nn.Module):
    def __init__(self, config):
        super(Softmax, self).__init__()
        self.temp = config.temp

    def forward(self, x, logits, labels=None):
        logits = logits / self.temp
        probs = F.softmax(logits, dim=1) 

        if labels is None:
            return probs, 0
        loss = F.cross_entropy(logits, labels) 
        return probs, loss
