import os
import argparse
import pandas as pd
import csv
from typing import Tuple
import traceback
from enum import Enum
from tqdm import tqdm
from diffuser_models.SDModel import SDVersion

other_models = [
    "janus-pro", "hidream", "infinity", 
    "kling", "gemini", "ideogram", "grok3", "4o", "dalle3",
]

def parse_args():
    parser = argparse.ArgumentParser(description='Image generation script')
    parser.add_argument('-m', '--model',
                        required=True,
                        choices=[v.short_name for v in SDVersion] + [v.name for v in SDVersion] + other_models,
                        help='Model name')
    parser.add_argument('-c', '--device', default=0)
    parser.add_argument('-t', '--test', default=False, action="store_true")
    parser.add_argument('-r', '--reverse', default=False, action="store_true")
    parser.add_argument('--trancate', default=False, action="store_true")
    return parser.parse_args()

args = parse_args()
selected = None
for v in SDVersion:
    if args.model.lower() == v.short_name or args.model.upper() == v.name:
        selected = v
        break

if args.model.lower() == "infinity":
    from Infinity.load_model import InfinityModel
    selected = InfinityModel()

if args.model.lower() == "kling":
    from API.kling_model import KlingModel
    selected = KlingModel()

if args.model.lower() == "gemini":
    from API.Gemini_model import GeminiModel
    selected = GeminiModel()

if args.model.lower() == "ideogram":
    from API.ideogram_model import IdeoGramModel
    selected = IdeoGramModel()

if args.model.lower() == "grok3":
    from API.grok_model import GrokModel
    selected = GrokModel()

if args.model.lower() == "janus-pro":
    from Janus.janus_model import JanusProModel
    selected = JanusProModel()

if args.model.lower() == "hidream":
    import sys
    sys.path.append("HiDream-I1-nf4")
    from hdi1.load_model import HiDreamModel
    selected = HiDreamModel()

if args.model.lower() in ["dalle3", "4o"]:
    from API.AIModel import AIClient
    selected = AIClient(args.model.lower())

if not selected:
    raise ValueError(f"Invalid model argument: {args.model}")

print(f"Selected model: {selected.name} ({selected.model_name})")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from models.SDModel import SDVersion, StableDiffusionModel

if args.test:
    CAPTION_DIR = "/home/final_captions"
    OUTPUT_BASE = "/home/final_dataset_test"
    MAPPING_DIR = "/home/final_img_cap_mapping"
else:
    CAPTION_DIR = "/home/final_captions"
    OUTPUT_BASE = "/home/final_dataset"
    MAPPING_DIR = "/home/final__mapping"

def parse_filename(filename: str) -> Tuple[str, str, str]:
    base_name = os.path.splitext(filename)[0]
    if base_name in ["imagenet-1k", "imagenet-1k-new", "COCO", "COCO-new"]:
        return base_name.replace("-new", ""), "", base_name
    parts = base_name.split('_', 1)
    if len(parts) == 1:
        return parts[0], "", base_name
    main_category = parts[0]
    sub_parts = parts[1].split('_', 1)
    if main_category == "HumanFace":
        if "FFHQ" in sub_parts[0]:
            return main_category, "FFHQ", base_name
        elif "celebahq" in sub_parts[0].lower():
            return main_category, "celebahq", base_name
    if main_category == "Scene" and sub_parts[0].startswith("LSUN"):
        sub_category = "_".join(sub_parts).replace("LSUN_", "LSUN/")[5:]
        return main_category, sub_category, base_name
    if main_category == "AnimalFace":
        for sub in ["cat", "dog", "wild"]:
            if sub in sub_parts[0].lower():
                return main_category, sub, base_name
    return main_category, "_".join(sub_parts), base_name

def generate_images(model_version: SDVersion, device="cuda:0"):
    import torch
    csv_files = [f for f in os.listdir(CAPTION_DIR) if f.endswith('.csv')]
    if args.reverse:
        csv_files.reverse()
    if args.trancate:
        csv_files = csv_files[9:]

    print(f"\n{'='*40}\nProcessing model: {model_version.name}\n{'='*40}")

    try:
        if model_version.name in other_models:
            model = model_version
        else:
            model = StableDiffusionModel(
                version=model_version,
                device=device if torch.cuda.is_available() else "cpu"
            )
    except Exception as e:
        print(f"Error initializing {model_version.name}: {str(e)}")
        traceback.print_exc()
        return

    for csv_file in csv_files:
        csv_path = os.path.join(CAPTION_DIR, csv_file)
        main_cat, sub_cat, base_name = parse_filename(csv_file)

        try:
            df = pd.read_csv(csv_path)
            captions = df['Caption'].head(5).tolist() if args.test else df['Caption'].tolist()
        except Exception as e:
            print(f"Error reading {csv_file}: {str(e)}")
            continue

        num_images_per_prompt = 2
        if main_cat in ["ImageNet-1k", "COCO"]:
            continue
        elif main_cat in ["ImageNet-1k-new", "COCO-new"]:
            num_images_per_prompt = 2
            main_cat = main_cat[:-4]

        if main_cat == "Scene" and sub_cat not in ["church", "bedroom", "classroom"]:
            continue

        print(main_cat, "----", sub_cat)

        mapping_subdir = os.path.join(MAPPING_DIR, model_version.name, main_cat, sub_cat)
        os.makedirs(mapping_subdir, exist_ok=True)
        mapping_file = os.path.join(mapping_subdir, f"{base_name}.csv")

        save_subdir = os.path.join(OUTPUT_BASE, model_version.name, main_cat, sub_cat)
        os.makedirs(save_subdir, exist_ok=True)
        existing_files = [f for f in os.listdir(save_subdir) if f.endswith('.png')]
        start_index = len(existing_files) // num_images_per_prompt
        file_exists = os.path.exists(mapping_file)

        for idx, caption in enumerate(tqdm(captions, desc="Generating images")):
            missing_indices = []
            for img_idx in range(num_images_per_prompt):
                filename = f"{base_name}_p{idx}_i{img_idx}.png"
                save_path = os.path.join(save_subdir, filename)
                print(save_path)
                if not os.path.exists(save_path):
                    missing_indices.append(img_idx)

            if not missing_indices:
                print(f"All images for prompt index {idx} already exist, skipping generation.")
                continue

            try:
                images = model.generate(
                    prompt=caption,
                    num_images=len(missing_indices)
                )
            except Exception as e:
                print(f"Generation failed for caption {idx}: {str(e)}")
                traceback.print_exc()
                continue

            for img_idx, img in zip(missing_indices, images):
                filename = f"{base_name}_p{idx}_i{img_idx}.png"
                save_path = os.path.join(save_subdir, filename)

                if os.path.exists(save_path):
                    print(f"Image already exists, skipping save: {save_path}")
                    continue

                try:
                    img.save(save_path)
                    print(f"Saved: {save_path}")
                    with open(mapping_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['ImagePath', 'Caption'])
                            file_exists = True
                        writer.writerow([save_path, caption])
                except Exception as e:
                    print(f"Failed to save {save_path}: {str(e)}")
                    traceback.print_exc()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    import torch
    print(torch.cuda.current_device())
    generate_images(model_version=selected, device=f"cuda:{args.device}")

if __name__ == "__main__":
    import torch
    print(torch.__version__)
    print(torch.version.cuda)
    main()
