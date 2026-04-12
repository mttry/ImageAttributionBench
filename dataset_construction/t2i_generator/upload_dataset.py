import os
import traceback
from pathlib import Path
from huggingface_hub import login, upload_file

# ================= 核心网络配置 =================
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# ================================================

# ================= 配置区域 =================
BASE_PATH = "/home/ImageAttributionBench-hf/data"
REPO_ID = "multiitsuki/ImageAttributionBench"
LOG_FILE = "file_upload_success.log" 

# 🎯 数据类目白名单
VALID_CATEGORIES = [
    "cat", "dog", "wild",               
    "COCO",                             
    "FFHQ", "celebahq",                 
    "ImageNet-1k",                      
    "bedroom", "church", "classroom"    
]
# ============================================

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)

uploaded_files = set()
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        uploaded_files = set(line.strip() for line in f if line.strip())

base_dir = Path(BASE_PATH)
if not base_dir.exists():
    raise FileNotFoundError(f"目录不存在: {BASE_PATH}")

print(f"🔍 正在扫描 {BASE_PATH} 下的 parquet 文件...")

for filepath in base_dir.rglob("*.parquet"):
    file_stem = filepath.stem 
    
    if file_stem not in VALID_CATEGORIES:
        continue
        
    try:
        rel_path = filepath.relative_to(base_dir.parent)
        path_in_repo = str(rel_path).replace("\\", "/")
    except ValueError:
        continue

    if path_in_repo in uploaded_files:
        print(f"⏭️  [跳过] '{path_in_repo}' 已在记录中。")
        continue
        
    print(f"\n⬆️  [正在上传单文件] '{path_in_repo}' ...")
    
    try:
        # 单文件排队上传
        upload_file(
            path_or_fileobj=str(filepath),           
            path_in_repo=path_in_repo, 
            repo_id=REPO_ID,
            repo_type="dataset"
        )
        
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{path_in_repo}\n")
        print(f"✅ [成功] '{path_in_repo}' 上传完毕！")
            
    except Exception as e:
        # 🚨 抓捕到异常！立刻打印底层完整堆栈并退出程序
        print(f"\n❌ [致命报错] 抓到错误了！文件 '{path_in_repo}' 上传崩溃！")
        print("\n⬇️⬇️⬇️⬇️⬇️ 详细报错堆栈开始 ⬇️⬇️⬇️⬇️⬇️\n")
        
        traceback.print_exc()
        
        print("\n⬆️⬆️⬆️⬆️⬆️ 详细报错堆栈结束 ⬆️⬆️⬆️⬆️⬆️\n")
        print("⚠️ 调试模式：为了防止报错信息被刷走，脚本已强制停止。")
        print("👉 请把上面 ⬇️ 和 ⬆️ 之间的完整英文报错发给我！")
        exit(1) # 强制停止脚本

print("\n🎉 所有单文件排队上传任务结束！")