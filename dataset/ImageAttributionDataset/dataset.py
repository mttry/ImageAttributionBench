import os  
from PIL import Image  
from torch.utils.data import Dataset  
import re

class ImageAttributionDataset(Dataset):  
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000, transform=None,):  
        self.root_dir = root_dir  
        self.transform = transform  
        self.num_images_per_semantic_per_class = num_images_per_semantic_per_class

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
        
        self.semantic_label_map = {  
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
        
        self.semantic_to_relpath = {  
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
        
        self.subclass_to_superclass = {  
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
    
    def __getitem__(self, idx):  
        img_path, label, semantic_label, semantic_subclass = self.samples[idx]  
        image = Image.open(img_path).convert("RGB")  

        GROK_LABEL = 13  # grok3对应的label编号  

        if label == GROK_LABEL:  
            width, height = image.size  
            # 裁剪掉右下角 100x50 像素（你可以根据实际调整尺寸）  
            crop_box = (0, 0, width - 100, height - 50)  
            image = image.crop(crop_box)  

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
    # 结果字典： {model_class_label: {semantic_subclass: count}}  
    stats = defaultdict(lambda: defaultdict(int))  
    
    for _, label, _, semantic_subclass in dataset.samples:  
        # label 是数字，反查model_class字符串方便展示  
        model_name = None  
        for k, v in dataset.model_class_to_label.items():  
            if v == label:  
                model_name = k  
                break  
        if model_name is None:  
            model_name = str(label)  

        stats[model_name][semantic_subclass] += 1  

    # 打印结果  
    for model_name, sub_dict in stats.items():  
        print(f"模型 {model_name} 加载图片数：")  
        for semantic_subclass, count in sub_dict.items():  
            print(f"  语义类别 {semantic_subclass}: {count} 张")  
        print()  

def check_load_order_consistency(root_dir, num_runs=5):  
    orders = []  
    for i in range(num_runs):  
        dataset = ImageAttributionDataset(root_dir, 2000)  
        # 提取每条样本的图片完整路径  
        order = [sample[0] for sample in dataset.samples]  
        orders.append(order)  

    consistent = True  
    for i in range(1, num_runs):  
        if orders[i] != orders[0]:  
            print(f"第{i+1}次加载顺序和第一次不同！")  
            consistent = False  
    if consistent:  
        print(f"{num_runs}次加载顺序完全一致。")  

# if __name__ == "__main__":  
#     check_load_order_consistency("/home/final_dataset")  
# 使用示例  
if __name__ == "__main__":  
    dataset = ImageAttributionDataset("/home/final_dataset", 2000)  
    load_stats(dataset)  