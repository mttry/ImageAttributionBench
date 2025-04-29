import torchvision.transforms as T  
from .dataset import ImageAttributionDataset
from ImageAttributionDataset import DATASET

@DATASET.register_module(module_name='resnet50')
class Resnet50Dataset(ImageAttributionDataset):  
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000, transform=None):  
        super().__init__(root_dir, num_images_per_semantic_per_class, transform)  
        # 如果外部传了transform，以外部为准，否则用ResNet50预处理默认transform  
        # print(f"loading resnet50 dataset....")
        if self.transform is None:  
            self.transform = T.Compose([  
                T.Resize(256),  
                T.CenterCrop(224),  
                T.ToTensor(),  
                T.Normalize(mean=[0.485, 0.456, 0.406],  
                            std=[0.229, 0.224, 0.225]),  
            ])  

    def __getitem__(self, idx):  
        item = super().__getitem__(idx)  
        image = item['image']  
        # 这里覆盖父类传入的 image 类型，确保是tensor且规范化  
        if self.transform:  
            image = self.transform(image)  
        item['image'] = image  
        return item  