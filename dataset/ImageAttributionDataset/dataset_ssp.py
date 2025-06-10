import torchvision.transforms as T  
from .dataset import ImageAttributionDataset
from ImageAttributionDataset import DATASET


class ConfigToAttr:  
    def __init__(self, config):  
        for k, v in config.items():  
            setattr(self, k, v)  
        self.opt = config 

@DATASET.register_module(module_name='ssp')
class SSPDataset(ImageAttributionDataset):  
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000, transform=None,degraded=0, **kwargs):  
        super().__init__(root_dir, num_images_per_semantic_per_class, transform, degraded=degraded)  
        from utils.ssp.tdataloader import processing
        self.processing = processing
        if self.transform is None:  
            self.transform = T.Compose([  
                T.Resize(256),  
                T.CenterCrop(256),  
                # T.ToTensor(),  
                # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  
            ])  
        self.opt = kwargs.get('config', {})  
        self.opt = ConfigToAttr(self.opt)  


    def __getitem__(self, idx):  
        item = super().__getitem__(idx)  
        image = item['image']  
        if self.transform:  
            image = self.transform(image)  
        image = self.processing(image, self.opt)
        item['image'] = image  
        return item  