import os  
from PIL import Image  
from torch.utils.data import Dataset  
import re
from io import BytesIO  
from PIL import ImageFilter  
from torchvision import transforms

semantic_label_map = {  
            "cat": 0,  
            "dog": 1,  
            "wild": 2,  
            "COCO": 3,  
            "FFHQ": 4,  
            "celebahq": 5,  
            "ImageNet-1k": 6,  
            "bedroom": 7,  
            "church": 8,  
            "classroom": 9,  
        }  
model_class_to_label = {  
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
semantic_to_relpath = {  
            "cat": "AnimalFace/cat",  
            "dog": "AnimalFace/dog",  
            "wild": "AnimalFace/wild",  
            "COCO": "COCO",  
            "FFHQ": "HumanFace/FFHQ",  
            "celebahq": "HumanFace/celebahq",  
            "ImageNet-1k": "ImageNet-1k",  
            "bedroom": "Scene/bedroom",  
            "church": "Scene/church",  
            "classroom": "Scene/classroom",  
        }  
subclass_to_superclass = {  
            "cat": "AnimalFace",  
            "dog": "AnimalFace",  
            "wild": "AnimalFace",  
            "COCO": "COCO",  
            "FFHQ": "HumanFace",  
            "celebahq": "HumanFace",  
            "ImageNet-1k": "ImageNet-1k",  
            "bedroom": "Scene",  
            "church": "Scene",  
            "classroom": "Scene",  
        }  
class ImageAttributionDataset(Dataset):  
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000, transform=None,degraded = 0):  
        self.root_dir = root_dir  
        self.transform = transform  
        self.num_images_per_semantic_per_class = num_images_per_semantic_per_class

        self.model_class_to_label = model_class_to_label
        
        self.semantic_label_map = semantic_label_map  
        
        self.semantic_to_relpath = semantic_to_relpath 
        
        self.subclass_to_superclass = subclass_to_superclass
        # degraded_test
        self.mode = None
        self.degraded = degraded
        assert degraded in range(7), "illegal degrade number"
        self.samples = []  
        self._make_dataset()  
        
    def _make_dataset(self):  
        pattern = re.compile(r'_p(\d+)_i(\d+)')  
        for model_class, model_label in self.model_class_to_label.items(): 
            if model_class in ["mid-5.2","mid-6.0"]:
                idx_to_load = (0,1,2,3)
            else:
                idx_to_load = (0,1) 
            model_path = os.path.join(self.root_dir, model_class)  
            if not os.path.isdir(model_path):  
                continue  
            for semantic, rel_path in self.semantic_to_relpath.items():  
                pic_count_per_semantic = 0
                full_path = os.path.join(model_path, rel_path)  
                if not os.path.isdir(full_path):  
                    continue  
                
                semantic_label = self.semantic_label_map.get(semantic, -1)  
                
                for fname in sorted(os.listdir(full_path)):  
                    if (model_label != 22 and not fname.lower().endswith(('.png'))) or (model_label == 22 and not fname.lower().endswith(('.png',".jpg",".jpeg"))):  
                        # print("do not load:", full_path, fname)
                        continue  
                    if pic_count_per_semantic >= self.num_images_per_semantic_per_class:
                        break
                    # load real
                    if model_label == 22:
                        img_path = os.path.join(full_path, fname)  
                        self.samples.append((img_path, model_label, semantic_label, semantic))
                        pic_count_per_semantic +=1 

                    # load generation model,load by filename 
                    else:
                        match = pattern.search(fname)  
                        if match:  
                            p_val = int(match.group(1))  
                            i_val = int(match.group(2))  
                            if p_val < 1000 and i_val in idx_to_load:  
                                img_path = os.path.join(full_path, fname)  
                                self.samples.append((img_path, model_label, semantic_label, semantic))
                                pic_count_per_semantic +=1   
                    
    def __len__(self):  
        return len(self.samples)  
    
    def set_train(self):
        self.mode = "train"

    def set_val(self):
        self.mode = "val"

    def set_test(self):
        self.mode = "test"
    
    def compress(self, img, quality=65):  
        """图像JPEG压缩，quality为压缩质量，1-100"""  
        img = img.convert("RGB")  
        buffer = BytesIO()  
        img.save(buffer, format="JPEG", quality=quality)  
        buffer.seek(0)  
        compressed_img = Image.open(buffer)  
        return compressed_img  

    def blur(self, img, sigma=2):  
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))  

    def downsample(self, image, scale_factor=0.5):  
        original_size = image.size 
        new_size = (max(1,int(original_size[0] * scale_factor)), max(1,int(original_size[1] * scale_factor)))  
        image = image.resize(new_size, Image.NEAREST)  
        image = image.resize(original_size, Image.NEAREST)  
        return image  

    def get_degraded_img(self, image):  
        if self.degraded == 1:  
            image = self.downsample(image, scale_factor=0.5)  
        elif self.degraded == 2:  
            image = self.downsample(image, scale_factor=0.25)  
        elif self.degraded == 3:  
            image = self.compress(image, quality=65)  
        elif self.degraded == 4:  
            image = self.compress(image, quality=30)  
        elif self.degraded == 5:  
            image = self.blur(image, sigma=3)  
        elif self.degraded == 6:  
            image = self.blur(image, sigma=5)  
        return image  

    def __getitem__(self, idx):  
        img_path, label, semantic_label, semantic_subclass = self.samples[idx]  
        image = Image.open(img_path).convert("RGB")  

        GROK_LABEL = 13  
        if label == GROK_LABEL:  
            width, height = image.size  
            crop_box = (0, 0, width - 100, height - 50)  
            image = image.crop(crop_box)  

        # print(f"mode: {self.mode}, degraded: {self.degraded}")  

        if self.mode and self.mode == "test":  
            image = self.get_degraded_img(image)  

        semantic_superclass = self.subclass_to_superclass.get(semantic_subclass, None)  

        return {  
            "image": image,  
            "label": label,  
            "semantic_label": semantic_label,  
            "semantic_subclass": semantic_subclass,  
            "semantic_superclass": semantic_superclass  
        }  

from collections import defaultdict  

def load_stats(dataset):  
    stats = defaultdict(lambda: defaultdict(int))  
    
    for _, label, _, semantic_subclass in dataset.samples:  
        model_name = None  
        for k, v in dataset.model_class_to_label.items():  
            if v == label:  
                model_name = k  
                break  
        if model_name is None:  
            model_name = str(label)  

        stats[model_name][semantic_subclass] += 1  

    for model_name, sub_dict in stats.items():  
        print(f"Model {model_name} loaded image count:")  
        for semantic_subclass, count in sub_dict.items():  
            print(f"  Semantic class {semantic_subclass}: {count} images")  
        print()  

def check_load_order_consistency(root_dir, num_runs=5):  
    orders = []  
    for i in range(num_runs):  
        dataset = ImageAttributionDataset(root_dir, 2000)  
        order = [sample[0] for sample in dataset.samples]  
        orders.append(order)  

    consistent = True  
    for i in range(1, num_runs):  
        if orders[i] != orders[0]:  
            print(f"The load order in run {i+1} differs from the first run!")  
            consistent = False  
    if consistent:  
        print(f"Load order is consistent across all {num_runs} runs.")  
# if __name__ == "__main__":  
#     check_load_order_consistency("/home/final_dataset")  
if __name__ == "__main__":  
    dataset = ImageAttributionDataset("/remote-home/share/gzy/attribution/final_dataset_thats_real", 2000,degraded=0)  
    load_stats(dataset)  
    # root_dir = "/home/final_dataset"  
    # save_dir = "./test_output"  
    # os.makedirs(save_dir, exist_ok=True)  

    # degraded_levels = list(range(7))  # 0 ~ 6  

    def save_sample(dataset, degraded_level):  
        data = dataset[0] 
        img = data["image"]  
        label = data.get("label", "unknown")  
        filename = f"sample_degraded{degraded_level}_label{label}.png"  
        img.save(os.path.join(save_dir, filename))  
        print(f"Saved {filename}")  

    # for level in degraded_levels:  
    #     print(f"Processing degraded level {level} ...")  
    #     dataset = ImageAttributionDataset(root_dir, 2000, degraded=level)  
    #     dataset.set_test()  
    #     save_sample(dataset, level)  
