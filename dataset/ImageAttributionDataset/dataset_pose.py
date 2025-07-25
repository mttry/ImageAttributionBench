import torchvision.transforms as T  
from .dataset import ImageAttributionDataset
from ImageAttributionDataset import DATASET


@DATASET.register_module(module_name='pose')
class POSEDataset(ImageAttributionDataset):  
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000, transform=None,degraded=0, **kwargs):  
        super().__init__(root_dir, num_images_per_semantic_per_class, transform, degraded=degraded)  
        if self.transform is None:  
            self.transform = T.Compose([  
                T.Resize(256),  
                T.CenterCrop(256),  
                T.ToTensor(),  
                T.Normalize(mean=[0.5, 0.5, 0.5],  
                            std=[0.5, 0.5, 0.5]),  
            ])  

    def __getitem__(self, idx):  
        item = super().__getitem__(idx)  
        image = item['image']  
        if self.transform:  
            image = self.transform(image)  
        item['image'] = image  
        return item  