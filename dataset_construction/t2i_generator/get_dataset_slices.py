import os
from datasets import Dataset, Image, Features, Value, disable_caching

# 1. 极其关键：强制禁用缓存，确保底层 API 每次都必须重新从硬盘读取真实图片数据！
disable_caching()

# --- 路径配置 (请确保与你当前的服务器路径一致) ---
BASE_DIR = "/fs/projects/SGH_CR_RAI-AP_szh-hpc_users/workplace/mot2sgh/ImageAttributionBench"
PARQUET_OUT_BASE = "/fs/projects/SGH_CR_RAI-AP_szh-hpc_users/workplace/mot2sgh/ImageAttributionBench-hf"

# 2. 定义 Hugging Face 的数据特征规范
features = Features({
    "image": Image(),            # 声明这是一个图像列
    "model": Value("string"),
    "main_category": Value("string"),
    "sub_category": Value("string"),
    "filename": Value("string")  # 保留原文件名以便溯源
})

def create_image_generator(sub_cat_path, model, main_cat, sub_cat):
    """闭包生成器：懒加载读取某个子类别下的所有图片"""
    def generator():
        for filename in os.listdir(sub_cat_path):
            if not filename.endswith((".png", ".jpg", ".jpeg")):
                continue
                
            filepath = os.path.join(sub_cat_path, filename)
            
            # 3. 手动读取原始二进制数据，保留生成模型最原始的像素和伪影
            try:
                with open(filepath, "rb") as f:
                    img_bytes = f.read()
            except Exception as e:
                print(f"读取失败: {filepath}, 错误: {e}")
                continue
            
            yield {
                # ！！！终极核心修复！！！
                # 绝对不能传绝对路径给 'path'，否则 HF 会丢弃 bytes 只存指针。
                # 这里只传 filename，欺骗底层 API，强迫它吞下 img_bytes！
                "image": {"bytes": img_bytes, "path": filename}, 
                "model": model,
                "main_category": main_cat,
                "sub_category": sub_cat,
                "filename": filename
            }
    return generator

def process_and_pack():
    if not os.path.exists(BASE_DIR):
        print(f"Error: 找不到基础目录 {BASE_DIR}")
        return

    # 1. 遍历模型级目录
    models = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    
    for model in models:
        model_dir = os.path.join(BASE_DIR, model)
        main_cats = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        
        # 2. 遍历主类别目录
        for main_cat in main_cats:
            main_cat_dir = os.path.join(model_dir, main_cat)
            sub_cats = [d for d in os.listdir(main_cat_dir) if os.path.isdir(os.path.join(main_cat_dir, d))]
            
            # 3. 遍历子类别目录
            for sub_cat in sub_cats:
                sub_cat_path = os.path.join(main_cat_dir, sub_cat)
                
                # 目标 Parquet 存放路径
                out_dir = os.path.join(PARQUET_OUT_BASE, "data", model, main_cat)
                os.makedirs(out_dir, exist_ok=True)
                parquet_path = os.path.join(out_dir, f"{sub_cat}.parquet")
                
                # --- 智能断点续传与纠错机制 ---
                if os.path.exists(parquet_path):
                    file_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
                    # 如果文件小于 50MB，大概率是之前生成的只包含路径的“假文件”
                    if file_size_mb < 50:
                        print(f"发现异常小文件 {parquet_path} ({file_size_mb:.2f} MB)，正在删除重建...")
                        os.remove(parquet_path)
                    else:
                        print(f"跳过已存在且有效的大文件: {parquet_path} ({file_size_mb:.2f} MB)")
                        continue

                # 检查目录下是否有图片
                has_images = any(f.endswith((".png", ".jpg", ".jpeg")) for f in os.listdir(sub_cat_path))
                if not has_images:
                    continue

                print(f"\n正在打包: Model={model} | Cat={main_cat}/{sub_cat} ...")
                
                # 4. 创建生成器并转换数据集
                gen = create_image_generator(sub_cat_path, model, main_cat, sub_cat)
                
                # 从生成器读取并写入 Parquet
                ds = Dataset.from_generator(gen, features=features)
                ds.to_parquet(parquet_path)
                
                # 打包完成后立即验证文件大小
                new_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
                print(f"✅ 保存成功: {parquet_path} (包含图片: {len(ds)} 张, 真实体积: {new_size_mb:.2f} MB)")

if __name__ == "__main__":
    print(f"{'='*50}\n🚀 开始执行 ImageAttributionBench 数据集 Parquet 封印计划\n{'='*50}")
    process_and_pack()
    print(f"\n{'='*50}\n🎉 全部数据已成功切片并封装为 Parquet！\n{'='*50}")