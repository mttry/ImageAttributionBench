import os

for k in [
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY"
]:
    os.environ.pop(k, None)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

save_dir = "/mnt/oss/IA/ImageAttributionBench-hf"
cache_dir = "/mnt/oss/IA/cache"

os.makedirs(save_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

ds = load_dataset(
    "multiitsuki/ImageAttributionBench",
    cache_dir=cache_dir,
)

ds.save_to_disk(save_dir)
print(f"saved to: {save_dir}")
