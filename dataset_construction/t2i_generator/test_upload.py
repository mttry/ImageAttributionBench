import os
import logging
import traceback
from huggingface_hub import upload_file

# ================= 1. 开启极致网络调试 =================
# 必须最先设置！
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# =======================================================

# ================= 2. 基础配置 =================
REPO_ID = "multiitsuki/ImageAttributionBench"
TOKEN = token = os.environ.get("HF_TOKEN")
LOCAL_TEST_FILE = "debug_hello_world.txt"
REPO_TEST_FILE = "debug_hello_world.txt"
# ===============================================

print(f"{'='*40}\n🚀 开始执行极简上传测试\n{'='*40}")

# 第一步：在本地创建一个极小的测试文件
print(f"📄 正在创建本地测试文件: {LOCAL_TEST_FILE} ...")
try:
    with open(LOCAL_TEST_FILE, "w", encoding="utf-8") as f:
        f.write("Hello Hugging Face! 如果你能看到这句话，说明最小链路已打通。\n")
        f.write("This is a minimal debug test file.\n")
    print("✅ 本地文件创建成功！")
except Exception as e:
    print(f"❌ 严重错误：连本地文件都创建失败了！检查磁盘空间或权限。报错: {e}")
    exit(1)

# 第二步：使用最基础的 upload_file 进行单文件上传
print(f"\n🌐 开始向 Hugging Face ({REPO_ID}) 发起上传请求...")
try:
    # upload_file 是最底层的 API，比 upload_folder 更直接
    upload_file(
        path_or_fileobj=LOCAL_TEST_FILE,
        path_in_repo=REPO_TEST_FILE,
        repo_id=REPO_ID,
        repo_type="dataset",
        token=TOKEN  # 直接显式传入 Token，避免环境变量读取失败
    )
    
    print(f"\n🎉 测试上传大成功！")
    print(f"👉 请前往网页端确认: https://huggingface.co/datasets/{REPO_ID}/tree/main")
    print("结论：你的网络、Token权限、仓库配置没有任何问题！问题出在之前的文件过大或并发逻辑上。")

except Exception as e:
    print("\n💥 测试上传失败！")
    print("⬇️⬇️⬇️ 真正的底层死因如下 (请仔细看最后几行) ⬇️⬇️⬇️")
    traceback.print_exc()
    print("⬆️⬆️⬆️ 真正的底层死因如上 ⬆️⬆️⬆️")
    
    print("\n🔍 常见排查指南：")
    print("- 如果是 404 Not Found: 你的仓库还没在网页端创建。")
    print("- 如果是 403 Forbidden: 你的 Token 没有 Write 权限，或者填错了。")
    print("- 如果是 ConnectionError / ReadTimeout: 服务器连 hf-mirror.com 都连不上，彻底断网了。")

finally:
    # 打扫战场（清理本地测试文件）
    if os.path.exists(LOCAL_TEST_FILE):
        os.remove(LOCAL_TEST_FILE)
        print(f"\n🧹 已清理本地测试文件 {LOCAL_TEST_FILE}。")