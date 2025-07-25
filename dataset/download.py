import os
import requests
import argparse
import zipfile

API_TOKEN = "" # TODO: fill in your api token here.
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
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.download_path, exist_ok=True)
    os.makedirs(args.dataset_path, exist_ok=True)

    headers = {"X-Dataverse-key": API_TOKEN} if API_TOKEN else {}
    url = f"{BASE_API}/api/datasets/:persistentId/versions/{VERSION}/files?persistentId={PERSISTENT_ID}"
    print(f"Fetching file list from: {url}")
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch file list. Status code: {response.status_code}")
        print(response.text)
        exit(1)

    files = response.json().get("data", [])
    print(f"Found {len(files)} files.")

    for file_info in files:
        filename = file_info["dataFile"]["filename"]
        file_id = file_info["dataFile"]["id"]

        if not filename.endswith(".zip") or "_" not in filename:
            continue

        model_name = filename.rsplit("_", 1)[0]
        semantic_name = filename.rsplit("_", 1)[-1].replace(".zip", "")

        if model_name in args.model_classes and semantic_name in args.semantic_classes:
            print(f"Downloading: {filename} (Model: {model_name}, Semantic: {semantic_name})")

            download_url = f"{BASE_API}/api/access/datafile/{file_id}"
            save_path = os.path.join(args.download_path, filename)

            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            print(f"✅ Downloaded to: {save_path}")

            # Unzip
            with zipfile.ZipFile(save_path, "r") as zip_ref:
                zip_ref.extractall(args.dataset_path)
            print(f"📦 Extracted to: {args.dataset_path}")

            if args.delete_zip:
                os.remove(save_path)
                print(f"🗑️ Deleted zip file: {save_path}")

if __name__ == "__main__":
    main()
