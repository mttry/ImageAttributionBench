import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from prompt_template import v1, v3,build_scene_prompt

torch.manual_seed(1234)

# 初始化模型
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# 路径配置
caption_dir = '/home/final_captions'
csv_files = [f for f in os.listdir(caption_dir) if f.endswith('.csv') and f != 'COCO.csv']

def select_prompt(filename):
    lower_name = filename.lower()

    if 'coco' in lower_name:
        return None  # 跳过 COCO

    elif any(x in lower_name for x in ['humanface', 'ffhq', 'celebahq', 'animalface']):
        return v3  # 用脸部 prompt

    elif 'scene' in lower_name:
        # 提取场景类别标签，如 'LSUN_kitchen' -> 'kitchen'
        scene_label = filename.lower().replace('.csv', '').split('_')[-1]
        return build_scene_prompt(scene_label)

    else:
        return v1  # 默认 fallback

# 遍历每个 CSV 文件
for csv_file in csv_files:
    csv_path = os.path.join(caption_dir, csv_file)
    df = pd.read_csv(csv_path)

    # 选择对应 prompt 模板
    prompt = select_prompt(csv_file)

    # 只处理前 10 个样本
    # subset = df.head(10)
    subset = df
    captions = []

    print(f'Processing {csv_file}...')

    for img_path in tqdm(subset['ImgPath']):
        if not os.path.exists(img_path):
            captions.append("")
            continue
        try:
            query = tokenizer.from_list_format([
                {'image': img_path},
                {'text': prompt}
            ])
            response, _ = model.chat(tokenizer, query=query, history=None, attention_mask=None)
            captions.append(response.strip())
        except Exception as e:
            print(f"Error with {img_path}: {e}")
            captions.append("")

    # 将前 10 个 caption 写入对应行，其余为空
    # full_captions = captions + [""] * (len(df) - 10)
    full_captions = captions
    df['Caption'] = full_captions

    df.to_csv(csv_path, index=False)
    print(f">>> Saved updated CSV with 10 captions: {csv_path}")
