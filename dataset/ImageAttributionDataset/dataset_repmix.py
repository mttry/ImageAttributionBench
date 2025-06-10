import torchvision.transforms as T  
from .dataset import ImageAttributionDataset
from ImageAttributionDataset import DATASET
from utils.repmix.augment_imagenetc import get_transforms
import numpy as np
import torch


@DATASET.register_module(module_name='repmix')
class RepmixDataset(ImageAttributionDataset):  
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000, transform=None,degraded=0,**kwargs):  
        super().__init__(root_dir, num_images_per_semantic_per_class, transform, degraded=degraded)  
        self.config = config = kwargs.get('config', {})  
        self.transform = get_transforms(  
            self.config['img_mean'], self.config['img_std'], self.config['img_rsize'],   
            self.config['img_csize'], self.config['pertubation'], False, False, 15  
        )  
        self.transform["test"] = self.transform["test_unseen"]
        self.N = len(self.samples)
        self.mixup_samples = config.get('mixup_samples', 2)  
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(self.mixup_samples))

    def __getitem__(self, idx):  
        """  
        返回一个图像和对应的标签，使用 mixup 方法。  

        Parameters:  
        index (int): 数据索引。  

        Returns:  
        tuple: (处理后的图像字典, 标签输出字典)  
        """  
        index = idx
        if self.mode in ["test", "val"]:
            ids = [index]
        else:
            ids = [index] + np.random.choice(self.N, self.mixup_samples-1).tolist()  
        x = []  
        y_gans = []  
        y_semantics = [] 

        for i in ids:  
            item = super().__getitem__(i)  
            image = item['image'] 
            y_gan = item['label']
            y_semantic = item['semantic_label']

            if self.transform:  
                transform = self.transform[self.mode]
                x_pre = transform[0](image)
                # x_post = transform[1](x_pre)
                x.append(transform[2](x_pre))

    
            y_gans.append(y_gan)  
            y_semantics.append(y_semantic)  

        x = torch.stack(x)  
        beta = self.dirichlet.sample()  

        y_gan = np.array(y_gans) 
        y_semantic = np.array(y_semantics)  
        y_out = {'x':x,'label': y_gan, 'semantic_label': y_semantic,
        'y_gan': y_gan, 'y_semantic':y_semantic, 'y_det': np.int64(y_gan != 22), 'beta': beta.clone()}  

        return y_out  
    
    @staticmethod  
    def collate_fn(batch):  
        collated = {}  
        for key in batch[0].keys():  
            vals = [b[key] for b in batch]  
            if key == 'x' or key == 'beta':  
                collated[key] = torch.cat(vals) if key == 'x' else torch.stack(vals)  
            else:  
                collated[key] = torch.stack([torch.as_tensor(v) for v in vals])  
        return collated  
