import os
import argparse
import pandas as pd
import csv
import torch
from typing import Tuple
import traceback
from tqdm import tqdm

other_models = [
    "janus-pro", "hidream", "infinity", 
    "kling", "gemini", "ideogram", "grok3", "4o", "dalle3",
]
openai_compatible_models = [
    "gpt-image-1",
    "gpt-image-1.5",
    "doubao-seedream-3.0-t2i",
    "doubao-seedream-5.0-lite",
    "gemini-2.5-flash-image",
    "gemini-3-pro-image",
]

def parse_args():
    parser = argparse.ArgumentParser(description='Image generation script')
    parser.add_argument('-m', '--model',
                        required=True,
                        help='Model name')
    parser.add_argument('-c', '--device', default=0)
    parser.add_argument('-t', '--test', default=False, action="store_true")
    parser.add_argument('-r', '--reverse', default=False, action="store_true")
    parser.add_argument('--trancate', default=False, action="store_true")
    return parser.parse_args()

args = parse_args()
selected = None
SDVersion = None
StableDiffusionModel = None
sd_import_error = None


def is_sd_version(model) -> bool:
    return SDVersion is not None and isinstance(model, SDVersion)


non_sd_models = set(other_models + openai_compatible_models)
if args.model not in non_sd_models:
    try:
        from diffuser_models.SDModel import SDVersion as ImportedSDVersion, StableDiffusionModel as ImportedStableDiffusionModel
        SDVersion = ImportedSDVersion
        StableDiffusionModel = ImportedStableDiffusionModel
    except Exception as exc:
        sd_import_error = exc

    if SDVersion is not None:
        for v in SDVersion:
            if args.model.lower() == v.short_name or args.model.upper() == v.name:
                selected = v
                break

    if selected is None:
        if sd_import_error is not None:
            raise RuntimeError(
                f"Failed to load Stable Diffusion models while resolving '{args.model}': {sd_import_error}"
            ) from sd_import_error
        raise ValueError(f"Invalid model argument: {args.model}")


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

if args.model in openai_compatible_models:
    from API.new_gen_image import OpenAICompatibleImageModel
    selected = OpenAICompatibleImageModel(args.model)

if not selected:
    raise ValueError(f"Invalid model argument: {args.model}")

selected_name = selected.name if is_sd_version(selected) else getattr(selected, 'name', args.model)
selected_model_name = selected.model_name if is_sd_version(selected) else getattr(selected, 'model_name', 'API/Custom')
print(f"Selected model: {selected_name} ({selected_model_name})")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

if args.test:
    CAPTION_DIR = "/home/i-moutingshu/mts/IA/ImageAttributionBench/dataset_construction/prompt_generator/downloaded_captions/final_captions_new"
    OUTPUT_BASE = "/mnt/ws-jfs/IA/ImageAttributionBench-test"
    MAPPING_DIR = "/mnt/ws-jfs/IA/ImageAttributionBench-test-mapping"
else:
    CAPTION_DIR = "/home/i-moutingshu/mts/IA/ImageAttributionBench/dataset_construction/prompt_generator/downloaded_captions/final_captions_new"
    OUTPUT_BASE = "/mnt/ws-jfs/IA/ImageAttributionBench"
    MAPPING_DIR = "/mnt/ws-jfs/IA/ImageAttributionBench-mapping"

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

def generate_images(model_version, device="cuda:0"):
    csv_files = [f for f in os.listdir(CAPTION_DIR) if f.endswith('.csv')]
    if args.reverse:
        csv_files.reverse()
    if args.trancate:
        csv_files = csv_files[9:]

    model_name_str = model_version.name if is_sd_version(model_version) else getattr(model_version, 'name', args.model)
    print(f"\n{'='*40}\nProcessing model: {model_name_str}\n{'='*40}")

    try:
        if is_sd_version(model_version):
            model = StableDiffusionModel(
                version=model_version,
                device=device if torch.cuda.is_available() else "cpu"
            )
        else:
            model = model_version
            
    except Exception as e:
        print(f"Error initializing {model_name_str}: {str(e)}")
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
        old_base_names = {"imagenet-1k", "COCO"}

        if base_name in old_base_names:
            continue
        if base_name == "imagenet-1k-new":
            # Use the canonical ImageNet-1k name for both folder and filename.
            main_cat = "ImageNet-1k"
            base_name = "ImageNet-1k"
        elif base_name == "COCO-new":
            main_cat = "COCO"

        if main_cat == "Scene" and sub_cat not in ["church", "bedroom", "classroom"]:
            continue

        print(main_cat, "----", sub_cat)

        mapping_subdir = os.path.join(MAPPING_DIR, model_name_str, main_cat, sub_cat)
        os.makedirs(mapping_subdir, exist_ok=True)
        mapping_file = os.path.join(mapping_subdir, f"{base_name}.csv")

        save_subdir = os.path.join(OUTPUT_BASE, model_name_str, main_cat, sub_cat)
        os.makedirs(save_subdir, exist_ok=True)
        
        file_exists = os.path.exists(mapping_file)

        for idx, caption in enumerate(tqdm(captions, desc="Generating images")):
            missing_indices = []
            for img_idx in range(num_images_per_prompt):
                filename = f"{base_name}_p{idx}_i{img_idx}.png"
                save_path = os.path.join(save_subdir, filename)
                
                if not os.path.exists(save_path):
                    missing_indices.append(img_idx)

            if not missing_indices:
                continue

            try:
                images = model.generate(
                    prompt=caption,
                    num_images=len(missing_indices)
                )
            except SystemExit as e:
                print(
                    f"Generation skipped for caption {idx} due to API/system exit: {str(e)}"
                )
                continue
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
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available, using CPU.")
        
    generate_images(model_version=selected, device=f"cuda:{args.device}")

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    main()
