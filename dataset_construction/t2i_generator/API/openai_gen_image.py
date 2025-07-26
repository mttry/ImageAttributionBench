import argparse  
import os  
import pandas as pd  
import torch  
import csv  
from typing import Tuple  
import traceback  
from tqdm import tqdm  # 导入进度条  

# 设置HuggingFace镜像源  

# 导入模型模块  
from AIModel import AIClient, ModelType  

# 配置路径  
CAPTION_DIR = "/home/final_captions"
OUTPUT_BASE = "/home/final_dataset"
MAPPING_DIR = "/home/final__mapping"  

# 需要运行的模型列表  
MODELS_TO_RUN = [  
    # SDVersion.SD1_5,  
    # SDVersion.SD2_1,  
    # SDVersion.SD3,  
    # SDVersion.SDXL,  
    ModelType.DALLE3,  
    ModelType.GPT4O,  
]  

def parse_filename(filename: str) -> Tuple[str, str, str]:  
    """解析文件名到目录结构，返回（主类别，子类别，基础名称）"""  
    base_name = os.path.splitext(filename)[0]  
    
    # 处理特殊文件  
    if base_name == "imagenet-1k":  
        return "ImageNet-1k", "", base_name  
    if base_name == "COCO":  
        return "COCO", "", base_name  
    
    # 分解主类别和子类别  
    parts = base_name.split('_', 1)  # 只分割第一个下划线  
    
    if len(parts) == 1:  
        return parts[0], "", base_name  
    
    main_category = parts[0]  
    sub_parts = parts[1].split('_', 1)  # 处理二级分类  
    
    # 特殊处理HumanFace的二级分类  
    if main_category == "HumanFace":  
        if "FFHQ" in sub_parts[0]:  
            return main_category, "FFHQ", base_name  
        elif "celebahq" in sub_parts[0].lower():  
            return main_category, "celebahq", base_name  
    
    # Scene类别的特殊处理  
    if main_category == "Scene" and sub_parts[0].startswith("LSUN"):  
        sub_category = "_".join(sub_parts).replace("LSUN_", "LSUN/")[5:]  
        return main_category, sub_category, base_name  
    
    # AnimalFace的二级分类  
    if main_category == "AnimalFace":  
        valid_subs = ["cat", "dog", "wild"]  
        for sub in valid_subs:  
            if sub in sub_parts[0].lower():  
                return main_category, sub, base_name  
    
    return main_category, "_".join(sub_parts), base_name  

def generate_images(version: ModelType):  
    # 获取所有CSV文件  
    csv_files = [f for f in os.listdir(CAPTION_DIR) if f.endswith('.csv')]  
    
    # 遍历所有模型  
    model_name = version.version_name  
    print(f"\n{'='*40}\nProcessing model: {model_name}\n{'='*40}")  
    
    try:  
        # 初始化模型  
        model = AIClient()  
    except Exception as e:  
        print(f"Error initializing {model_name}: {str(e)}")  
        traceback.print_exc()  
        return  
        
    # 遍历所有CSV文件  
    for csv_file in csv_files:  
        csv_path = os.path.join(CAPTION_DIR, csv_file)  
        
        # 解析目录结构  
        main_cat, sub_cat, base_name = parse_filename(csv_file)  
        # print(f"\nProcessing {main_cat}/{sub_cat}...")  

        try:  
            # 读取CSV并获取前5个提示词  
            df = pd.read_csv(csv_path)  
            captions = df['Caption'].tolist()  
        except Exception as e:  
            print(f"Error reading {csv_file}: {str(e)}")  
            continue  

        if main_cat in ["ImageNet-1k", "COCO"]:  
            continue  
        elif main_cat in ["ImageNet-1k-new", "COCO-new"]:  
            num_images_per_prompt = 2  
            main_cat = main_cat[:-4]  
        
        if main_cat == "Scene" and sub_cat not in ["church", "bedroom", "classroom"]:  
            continue  
        
        print(main_cat, "----", sub_cat)  
        
        # 创建映射文件路径  
        mapping_subdir = os.path.join(MAPPING_DIR, model_name, main_cat, sub_cat)  
        os.makedirs(mapping_subdir, exist_ok=True)  
        mapping_file = os.path.join(mapping_subdir, f"{base_name}.csv")  

         # 计算已经生成的文件数  
        save_subdir = os.path.join(OUTPUT_BASE, model_name, main_cat, sub_cat)
        os.makedirs(save_subdir, exist_ok=True)  
        existing_files = [f for f in os.listdir(os.path.join(OUTPUT_BASE, model_name, main_cat, sub_cat)) if f.endswith('.png')]  
        start_index = len(existing_files) // 2  # 计算从哪个索引开始  
        
        # 初始化映射文件  
        file_exists = os.path.exists(mapping_file)  
        
        # 处理每个提示词  
        for idx, caption in tqdm(enumerate(captions), desc="Generating images", total=len(captions)):  
            if idx < start_index:  # 跳过已生成的项  
                continue  
            try:  
                # 生成两张图像  
                images = model.generate(  
                    prompt=caption,  
                    num_images=2,  
                    model=version  
                )  
            except Exception as e:  
                print(f"Generation failed for caption {idx}: {str(e)}")  
                traceback.print_exc()  
                continue  
            
            # 保存生成的图像  
            for img_idx, img in enumerate(images):  
                # 构建保存路径  
                save_subdir = os.path.join(  
                    OUTPUT_BASE,  
                    model_name,  
                    main_cat,  
                    sub_cat  
                )  
                os.makedirs(save_subdir, exist_ok=True)  
                
                # 生成唯一文件名  
                filename = f"{base_name}_p{idx}_i{img_idx}.png"  
                save_path = os.path.join(save_subdir, filename)  
                
                try:  
                    img.save(save_path)  
                    print(f"Saved: {save_path}")  
                    
                    # 写入映射文件  
                    with open(mapping_file, 'a', newline='', encoding='utf-8') as f:  
                        writer = csv.writer(f)  
                        if not file_exists:  
                            writer.writerow(['ImagePath', 'Caption'])  
                            file_exists = True  
                        writer.writerow([save_path, caption])  
                except Exception as e:  
                    print(f"Failed to save {save_path}: {str(e)}")  
                    traceback.print_exc()  

def parse_args():  
    parser = argparse.ArgumentParser(description='生成图像脚本')  
    parser.add_argument('-m', '--model',  
                        required=True,  
                        choices=[v.short_name for v in ModelType] + [v.name for v in ModelType],  
                        help=f'''支持的模型简写：  
                        4o        -> gpt-4o  
                        playground-> Playground v2.5  
                        dalle3         -> dal-le3  
                        ''')  
    return parser.parse_args()  

if __name__ == "__main__":  
    args = parse_args()  
    # 匹配模型（优先匹配简写）  
    selected = None  
    for v in ModelType:  
        if args.model.lower() == v.short_name or args.model.upper() == v.name:  
            selected = v  
            break  
    
    if not selected:  
        raise ValueError(f"无效的模型参数: {args.model}")  
    
    print(f"已选择模型: {selected.name} ({selected.model_name})")  
    # 后续生成逻辑...  
    generate_images(selected)  