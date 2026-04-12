import os
import requests
import argparse
import urllib3
import threading
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 抑制关闭 SSL 验证后产生的 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_TOKEN = "47825235-f0f9-4ef5-a739-a2b0e8f21b8a" # TODO: fill in your api token here.
PERSISTENT_ID = "doi:10.7910/DVN/O4S4IV"
VERSION = ":latest"
BASE_API = "https://dataverse.harvard.edu"

ALL_MODEL_CLASSES = [
    "4o", "CogView3_PLUS", "FLUX", "KANDINSKY", "PIXART", "PLAYGROUND_2_5",
    "SD1_5", "SD2_1", "SD3", "SD3_5", "SDXL", "dalle3", "gemini", "grok3",
    "hidream", "hunyuan", "ideogram", "infinity", "janus-pro", "kling",
    "mid-5.2", "mid-6.0", "real"
]

ALL_SEMANTIC_CLASSES = [
    "COCO", "FFHQ", "ImageNet-1k", "bedroom", "cat", "celebahq",
    "church", "classroom", "dog", "wild"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Download and extract selected model-semantic zip files from Dataverse.")
    parser.add_argument("--download_path", type=str, required=True, help="Path to save downloaded zip files.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to extract dataset zip files.")
    parser.add_argument("--model_classes", nargs="+", choices=ALL_MODEL_CLASSES, default=ALL_MODEL_CLASSES, help="Target model classes.")
    parser.add_argument("--semantic_classes", nargs="+", choices=ALL_SEMANTIC_CLASSES, default=ALL_SEMANTIC_CLASSES, help="Target semantic classes.")
    parser.add_argument("--delete_zip", action="store_true", help="If set, delete zip files after extraction.")
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable parallel downloading (default: True). Use --no-parallel to disable.")
    return parser.parse_args()

def download_file_parallel(url: str, save_path: str, num_threads: int = 8, verify_ssl: bool = False) -> None:
    """
    Accelerates download by splitting the file into chunks and downloading them concurrently.
    Merges the part files sequentially after all threads complete.
    """
    with requests.get(url, stream=True, verify=verify_ssl) as r:
        r.raise_for_status()
        actual_url = r.url
        total_size = int(r.headers.get('content-length', 0))
        accept_ranges = r.headers.get('accept-ranges', 'none').lower() == 'bytes'

    if total_size == 0 or not accept_ranges:
        print("Server does not support Range requests or size is unknown. Falling back to sequential download...")
        download_file_sequential(actual_url, save_path, verify_ssl)
        return

    chunk_size = total_size // num_threads
    part_files = [f"{save_path}.part{i}" for i in range(num_threads)]
    progress_lock = threading.Lock()
    
    def _download_chunk(start_byte: int, end_byte: int, part_path: str, progress_bar: tqdm) -> None:
        headers = {"Range": f"bytes={start_byte}-{end_byte}"}
        with requests.get(actual_url, headers=headers, stream=True, verify=verify_ssl) as response:
            response.raise_for_status()
            with open(part_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    size = f.write(chunk)
                    with progress_lock:
                        progress_bar.update(size)

    print(f"Starting parallel download with {num_threads} threads for {os.path.basename(save_path)}...")
    with tqdm(total=total_size, unit='iB', unit_scale=True, unit_divisor=1024, desc=os.path.basename(save_path)) as bar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_threads):
                start = i * chunk_size
                end = total_size - 1 if i == num_threads - 1 else (start + chunk_size - 1)
                futures.append(executor.submit(_download_chunk, start, end, part_files[i], bar))
            
            for future in as_completed(futures):
                future.result()

    print(f"\nMerging {num_threads} parts into {save_path}...")
    with open(save_path, "wb") as outfile:
        for part_file in part_files:
            with open(part_file, "rb") as infile:
                while True:
                    data = infile.read(1024 * 1024 * 10)
                    if not data:
                        break
                    outfile.write(data)
            os.remove(part_file)

def download_file_sequential(url: str, save_path: str, verify_ssl: bool = False) -> None:
    """
    Standard sequential download with tqdm progress bar.
    """
    with requests.get(url, stream=True, verify=verify_ssl) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(save_path, "wb") as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                size = f.write(chunk)
                bar.update(size)

import os
import subprocess
import shutil
from collections import defaultdict
import argparse

# 按照要求的文件夹结构进行父级分类映射
SEMANTIC_TO_SUPER = {
    "cat": "AnimalFace",
    "dog": "AnimalFace",
    "wild": "AnimalFace",
    "celebahq": "HumanFace",
    "FFHQ": "HumanFace",
    "bedroom": "Scene",
    "church": "Scene",
    "classroom": "Scene",
    "COCO": "COCO",
    "ImageNet-1k": "ImageNet-1k"
}

def process_dataset_groups(files: list, args: argparse.Namespace) -> None:
    """
    Uses pre-computed hash maps of all valid combinations and routes multiple 
    image formats (.png, .jpg, .jpeg) into the canonical directory structure.
    """
    dataset_groups = defaultdict(list)
    
    # 1. 预计算所有合法的组合字典
    valid_combinations = {}
    for m in ALL_MODEL_CLASSES:
        for s in ALL_SEMANTIC_CLASSES:
            valid_combinations[f"{m}_{s}".lower()] = (m, s)
            if "-" in s:
                valid_combinations[f"{m}_{s.replace('-', '_')}".lower()] = (m, s)

    # 2. 精确扫描文件列表
    for file_info in files:
        filename = file_info["dataFile"]["filename"]
        base_name, ext = filename.rsplit(".", 1)
        
        if not (ext.lower() == "zip" or ext.lower().startswith("z")):
            continue
            
        match = valid_combinations.get(base_name.lower())
        if not match:
            print(f"⚠️ Unrecognized file format, skipping: {filename}")
            continue
            
        model_name, semantic_name = match
        
        if model_name in args.model_classes and semantic_name in args.semantic_classes:
            dataset_groups[base_name].append(file_info)

    # 3. 下载与解压路由
    # 定义支持的图像后缀名元组
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')

    for base_name, parts in dataset_groups.items():
        print(f"\n[{base_name}] Starting processing...")
        
        downloaded_paths = []
        main_zip_path = None
        
        model_name, semantic_name = valid_combinations[base_name.lower()]
        parts.sort(key=lambda x: x["dataFile"]["filename"])
        
        for file_info in parts:
            filename = file_info["dataFile"]["filename"]
            save_path = os.path.join(args.download_path, filename)
            download_url = f"{BASE_API}/api/access/datafile/{file_info['dataFile']['id']}"
            
            if args.parallel:
                download_file_parallel(download_url, save_path, num_threads=8, verify_ssl=False)
            else:
                download_file_sequential(download_url, save_path, verify_ssl=False)
                
            downloaded_paths.append(save_path)
            if filename.lower().endswith(".zip"):
                main_zip_path = save_path

        if not main_zip_path:
            print(f"❌ Error: Main .zip file missing for {base_name}. Skipping...")
            continue

        super_category = SEMANTIC_TO_SUPER.get(semantic_name, semantic_name)
        if super_category == semantic_name:
            canonical_dir = os.path.join(args.dataset_path, model_name, semantic_name)
        else:
            canonical_dir = os.path.join(args.dataset_path, model_name, super_category, semantic_name)
        
        os.makedirs(canonical_dir, exist_ok=True)
        
        temp_sandbox_dir = os.path.join(args.dataset_path, f"_temp_{base_name}")
        os.makedirs(temp_sandbox_dir, exist_ok=True)

        print(f"📦 Extracting to sandbox and routing to: {canonical_dir}")
        subprocess.run(["7z", "x", "-y", f"-o{temp_sandbox_dir}", main_zip_path], check=True)
        
        # 4. 遍历沙盒，提取所有符合 VALID_EXTENSIONS 的图片
        routed_count = 0
        for root, _, extracted_files in os.walk(temp_sandbox_dir):
            for file in extracted_files:
                if file.lower().endswith(VALID_EXTENSIONS):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(canonical_dir, file)
                    os.replace(src_path, dst_path)
                    routed_count += 1
                    
        print(f"✅ Successfully routed {routed_count} images (.png/.jpg/.jpeg) to standard structure.")

        shutil.rmtree(temp_sandbox_dir)
        if args.delete_zip:
            for path in downloaded_paths:
                os.remove(path)
            print(f"🗑️ Cleaned up zip parts for {base_name}")

def main():
    args = parse_args()
    os.makedirs(args.download_path, exist_ok=True)
    os.makedirs(args.dataset_path, exist_ok=True)

    headers = {"X-Dataverse-key": API_TOKEN} if API_TOKEN else {}
    url = f"{BASE_API}/api/datasets/:persistentId/versions/{VERSION}/files?persistentId={PERSISTENT_ID}"
    print(f"Fetching file list from: {url}")
    
    response = requests.get(url, headers=headers, verify=False)
    
    if response.status_code != 200:
        print(f"Failed to fetch file list. Status code: {response.status_code}")
        print(response.text)
        exit(1)

    files = response.json().get("data", [])
    print(f"Found {len(files)} files.")

    process_dataset_groups(files, args)

if __name__ == "__main__":
    main()