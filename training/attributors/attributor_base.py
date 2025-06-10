import abc  
import torch  
import torch.nn as nn  
from typing import Union  

class AbstractAttributor(nn.Module, metaclass=abc.ABCMeta):  
    """  
    All attribution models should subclass this class.  
    """  
    def __init__(self, config=None, load_param: Union[bool, str] = False):  
        """  
        config:   (dict)  
            configurations for the model  
        load_param:  (False | True | Path(str))  
            False - do not load parameters;  
            True - load from default path;  
            Path(str) - load from specified path  
        """  
        super().__init__()  

    @abc.abstractmethod  
    def extract_features(self, input_data: dict) -> torch.Tensor:  
        """  
        Given input data, extract features needed for attribution.  
        """  
        pass  

    @abc.abstractmethod  
    def forward(self, input_data: dict, inference=False) -> dict:  
        """  
        Forward pass of the model, returning attribution results and optional intermediate outputs.  
        """  
        pass  

    @abc.abstractmethod  
    def classifier(self, features: torch.Tensor) -> torch.Tensor:  
        """  
        Perform attribution based on features, returning attribution map or scores.  
        """  
        pass  

    @abc.abstractmethod  
    def build_model(self, config):  
        """  
        Build the backbone or main module of the attribution model.  
        """  
        pass  

    @abc.abstractmethod  
    def build_loss(self, config):  
        """  
        Build the loss function for attribution training.  
        """  
        pass  

    @abc.abstractmethod  
    def compute_losses(self, input_data: dict, pred_dict: dict) -> dict:  
        """  
        Compute the losses given input and prediction dictionaries.  
        """  
        pass  

    @abc.abstractmethod  
    def compute_metrics(self, input_data: dict, pred_dict: dict) -> dict:  
        """  
        Compute training or evaluation metrics.  
        """  
        pass  