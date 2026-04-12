from huggingface_hub import HfApi
import os

# 配置代理（保持与之前一致）
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

api = HfApi()
repo_id = "multiitsuki/ImageAttributionBench"

# 需要移动的文件夹列表
folders_to_move = ["FLUX2_KLEIN", "Z_IMAGE"]

print(f"🚀 开始在线调整 {repo_id} 的目录结构...")

for folder in folders_to_move:
    try:
        # 将根目录下的文件夹移动到 data/ 路径下
        api.move_folder(
            from_path=folder,
            to_path=f"data/{folder}",
            repo_id=repo_id,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN")
        )
        print(f"✅ 成功：{folder} -> data/{folder}")
    except Exception as e:
        if "Source path not found" in str(e):
            print(f"⏭️  跳过：文件夹 '{folder}' 在根目录下不存在（可能已经移动过了）")
        else:
            print(f"❌ 失败：移动 {folder} 时出错: {e}")

print("\n🎉 目录调整任务执行完毕！请刷新网页端确认。")