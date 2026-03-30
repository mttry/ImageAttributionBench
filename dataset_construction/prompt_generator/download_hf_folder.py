import os
from huggingface_hub import snapshot_download

# 1. 填入你的 Hugging Face Token（因为之前提到可能是私有仓库）
# 如果这个仓库是公开的，把这里改成 HF_TOKEN = None 即可
import os
hf_token = os.environ.get("image_attribution_bench_access_token")
HF_TOKEN = hf_token

# 2. 定义仓库信息
REPO_ID = "multiitsuki/ImageAttributionBench-caption"
# 注意：因为你要下的是 dataset，而不是 model，所以必须指定 repo_type
REPO_TYPE = "dataset" 

# 3. 指定要下载的特定文件夹路径（支持通配符）
# 这里表示只下载 final_captions 文件夹下的所有内容
ALLOW_PATTERNS = "final_captions/*"

# 4. 指定下载到本地的哪个路径
# 这里设置为当前目录下的 downloaded_captions 文件夹
LOCAL_DIR = "./downloaded_captions"

print(f"开始从 {REPO_ID} 下载 {ALLOW_PATTERNS} ...")

try:
    # 5. 执行下载
    download_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=ALLOW_PATTERNS,
        local_dir=LOCAL_DIR,
        token=HF_TOKEN,
        resume_download=True  # 开启断点续传，如果在集群上网络中断可以接着下
    )
    print(f"\n✅ 下载成功！文件已保存至: {os.path.abspath(download_path)}")
    
except Exception as e:
    print(f"\n❌ 下载失败，错误信息: {e}")