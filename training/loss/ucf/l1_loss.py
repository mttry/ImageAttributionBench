import torch.nn as nn
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="l1loss")
class L1Loss(AbstractLossClass):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, inputs, targets):
        """
        Computes the l1 loss.
        """
        # Compute the l1 loss
        # print(inputs.shape)
        # print(targets)
        loss = self.loss_fn(inputs, targets)
        # print(loss)
        # exit(0)
        return loss