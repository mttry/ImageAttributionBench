import torchvision.transforms as T  
from .dataset import ImageAttributionDataset
from ImageAttributionDataset import DATASET
from scipy.fftpack import dct
import numpy as np

@DATASET.register_module(module_name='dct')
class DCTDataset(ImageAttributionDataset):  
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000, degraded=0, transform=None, **kwargs):  
        super().__init__(root_dir, num_images_per_semantic_per_class, transform, degraded=degraded)  
        print(f"loading DCT dataset....") 

        self.transform_pre = T.Compose([  
            T.Resize(256),  
            T.CenterCrop(256),  
        ])  
        self.transform_post = T.Compose([  
            T.ToTensor(),  
        ]) 

    def extract_dct(self, img):
        img = img.astype(np.float32)
        if img.ndim == 3:
            dct_channels = []
            for c in range(img.shape[2]):
                channel = img[..., c]
                channel_dct = dct(channel, type=2, norm='ortho', axis=0)
                channel_dct = dct(channel_dct, type=2, norm='ortho', axis=1)
                dct_channels.append(channel_dct)
            img = np.stack(dct_channels, axis=2)
        else:
            img = dct(img, type=2, norm='ortho', axis=0)
            img = dct(img, type=2, norm='ortho', axis=1)
        img = np.abs(img)
        # ...
        return img
    

    def __getitem__(self, idx):  
        item = super().__getitem__(idx)  
        image = item['image']  
        image = self.transform_pre(image)  
        image = np.array(image)    
        image = self.extract_dct(image)
        # print(image.shape)
        image = self.transform_post(image).float()  
        # print(image.shape)
        item['image'] = image  
        return item  