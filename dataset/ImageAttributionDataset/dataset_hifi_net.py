import torchvision.transforms as T  
from .dataset import ImageAttributionDataset
from ImageAttributionDataset import DATASET
from torchvision import transforms

@DATASET.register_module(module_name="hifi_net")  
class HiFiNetDataset(ImageAttributionDataset):  
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000, transform=None, degraded=0, **kwargs):  
        super().__init__(root_dir, num_images_per_semantic_per_class, transform, degraded)  
        if self.transform is None:  
            self.transform = transforms.Compose([  
                transforms.Resize(256),            
                transforms.CenterCrop(256),           
                transforms.ToTensor(),  
                transforms.Normalize(mean=(0.5, 0.5, 0.5),  
                                     std=(0.5, 0.5, 0.5))  
            ])  
            self.model_class_to_label = {  
            '4o': 0,  
            'CogView3_PLUS': 1,  
            'FLUX': 2,  
            'KANDINSKY': 3,  
            'PIXART': 4,  
            'PLAYGROUND_2_5': 5,  
            'SD1_5': 6,  
            'SD2_1': 7,  
            'SD3': 8,  
            'SD3_5': 9,  
            'SDXL': 10,  
            'dalle3': 11,  
            'gemini': 12,  
            'grok3': 13,  
            'hidream': 14,  
            'hunyuan': 15,  
            'ideogram': 16,  
            'infinity': 17,  
            'janus-pro': 18,  
            'kling': 19,  
            'mid-5.2': 20,  
            'mid-6.0': 21,  
            'real': 22  
        }  
        # level0: 0 generated, 1 real;
        # level1: 0 commercial, 1 open-source, 2 real;
        # level2: 0 commercial, 1 SD, 2 diffusers, 3 DiT, 4 AR, 5 real;
        # level3: the same as label
        self.label_mapping = [
            (0,0,0,0),
            (0,1,3,1),
            (0,1,2,2),
            (0,1,2,3),
            (0,1,2,4),
            (0,1,2,5),
            (0,1,1,6),
            (0,1,1,7),
            (0,1,1,8),
            (0,1,1,9),
            (0,1,1,10),
            (0,0,0,11),
            (0,0,0,12),
            (0,0,0,13),
            (0,1,3,14),
            (0,1,3,15),
            (0,0,0,16),
            (0,1,4,17),
            (0,1,4,18),
            (0,0,0,19),
            (0,0,0,20),
            (0,0,0,21),
            (1,2,5,22),

        ]
    def __getitem__(self, idx):  
        item = super().__getitem__(idx)  
        image = item["image"]  
        if self.transform:  
            image = self.transform(image)  
        item["image"] = image  
        item["hierachi_label"] = self.label_mapping[item["label"]]
        return item  