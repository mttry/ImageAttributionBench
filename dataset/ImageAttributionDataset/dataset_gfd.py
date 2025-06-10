import torchvision.transforms as T  
from .dataset import ImageAttributionDataset
from ImageAttributionDataset import DATASET


@DATASET.register_module(module_name='gfd')
class GFDDataset(ImageAttributionDataset):  
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000, transform=None,degraded=0, **kwargs):  
        super().__init__(root_dir, num_images_per_semantic_per_class, transform, degraded=degraded)  
        if self.transform is None:  
            self.transform = T.Compose([  
                T.Resize(256),  
                T.CenterCrop(256),  
                T.ToTensor(),  
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  
            ])  


    def __getitem__(self, idx):  
        item = super().__getitem__(idx)  
        image = item['image']  
        item["image"] = self.transform(image)
        if self.mode == "train":
            label = item['label']

            if label == 22:
                item_fake = self.get_fake_one()
                item['real_img'] = self.transform(image)  
                item['fake_img'] = self.transform(item_fake['image'])
                item['real_label'] = 22
                item['fake_label'] = item_fake['label']  
                
            else:  
                item_real = self.get_real_one()
                item['real_img'] = self.transform(item_real['image'])
                item['fake_img'] = self.transform(image)
                item['real_label'] = 22
                item['fake_label'] = item['label']
                
        else:
            image = self.transform(image)  
            item['image'] = image  
        
        return item

    def get_fake_one(self):
        import random
        candidates = [i for i, sample in enumerate(self.samples) if sample[1] != 22]
        idx = random.choice(candidates)
        return super().__getitem__(idx)

    def get_real_one(self):
        import random
        candidates = [i for i, sample in enumerate(self.samples) if sample[1] == 22]
        idx = random.choice(candidates)
        return super().__getitem__(idx)