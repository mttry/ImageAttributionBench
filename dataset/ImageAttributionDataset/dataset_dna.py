import torchvision.transforms as transforms 
from .dataset import ImageAttributionDataset
from ImageAttributionDataset import DATASET
from training.utils.dna.transforms import MultiCropTransform, get_transforms
from PIL import Image, ImageFile
import random
import numpy as np
import torch
from utils.dataset_util import ConfigToAttr


@DATASET.register_module(module_name='dna')
class DNADataset(ImageAttributionDataset):  
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000, transform=None,degraded=0, **kwargs):  
        super().__init__(root_dir, num_images_per_semantic_per_class, transform, degraded=degraded)  
        config = self.config = kwargs.get("config", None)
        self.config = config = ConfigToAttr(config)
        self.pretrain_transforms = get_transforms(config.crop_size)
        self.class_num = self.pretrain_transforms.class_num
        self.multi_size = config.multi_size
        self.resize_size = config.resize_size
        self.second_resize_size = config.get("second_resize_size", None)

        self.train_stage = config.train_stage



        crop_transforms = []
        for s in self.multi_size:
            RandomCrop = transforms.RandomCrop(size=s)
            crop_transforms.append(RandomCrop)
        self.pre_transform = transforms.Compose([  
                transforms.Resize(256),          
                transforms.CenterCrop(256)])
        self.multicroptransform = MultiCropTransform(crop_transforms)
        self.norm_transform = transforms.Compose([  
            transforms.ToTensor(),   
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  
        ])  

    def load_sample(self, img):
        if img.size[0]!=img.size[1]:
            img = transforms.CenterCrop(size=self.config.crop_size)(img)

        if self.resize_size is not None:
            img = img.resize(self.resize_size)
        if self.second_resize_size is not None:
            img = img.resize(self.second_resize_size)
            
        crops = self.multicroptransform(img)
        img = self.norm_transform(img)
        crops = [self.norm_transform(crop) for crop in crops]

        return img, crops

    def __getitem__(self, idx):  
        item = super().__getitem__(idx)  
        image = self.pre_transform(item['image'])

        if self.train_stage == 1:
            # pretrain
            img = transforms.RandomCrop(size=self.config.crop_size)(image)
            select_id=random.randint(0,self.class_num-1)
            pretrain_transform=self.pretrain_transforms.select_tranform(select_id)
            transformed = pretrain_transform(image=np.asarray(img))
            img = Image.fromarray(transformed["image"])

            if self.resize_size is not None:
                img = img.resize(self.resize_size)

            crops = self.multicroptransform(img)
            img = self.norm_transform(img)
            crops = [self.norm_transform(crop) for crop in crops]
            lab = torch.tensor(select_id, dtype=torch.long)
            item['image'] = img 
            item['crops'] = crops 
            item['label'] = lab
            return item  
        
        else:
            img, crops = self.load_sample(image)
            item['image'] = img 
            item['crops'] = crops 
            return item

