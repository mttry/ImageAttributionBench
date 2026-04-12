import os

from datasets import Dataset, Features, Image, Value, disable_caching


disable_caching()

BASE_DIR = "/home/final_dataset"
ETHICS_OVERRIDE_BASE = "/home/final_dataset_ethics"
PARQUET_OUT_BASE = "/home/ImageAttributionBench-hf"
VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg")
MIN_VALID_PARQUET_MB = 50

features = Features(
    {
        "image": Image(),
        "model": Value("string"),
        "main_category": Value("string"),
        "sub_category": Value("string"),
        "filename": Value("string"),
    }
)


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(VALID_IMAGE_EXTS)


def resolve_source_path(base_filepath: str) -> str:
    """Use the ethics file when it exists at the same relative path."""
    rel_path = os.path.relpath(base_filepath, BASE_DIR)
    ethics_filepath = os.path.join(ETHICS_OVERRIDE_BASE, rel_path)
    if os.path.isfile(ethics_filepath):
        return ethics_filepath
    return base_filepath


def create_image_generator(target_path, model, main_cat, sub_cat):
    def generator():
        # 注意：使用 target_path 而不是固定的 sub_cat_path
        filenames = sorted(os.listdir(target_path))
        override_count = 0

        for filename in filenames:
            if not is_image_file(filename):
                continue

            base_filepath = os.path.join(target_path, filename)
            source_filepath = resolve_source_path(base_filepath)
            if source_filepath != base_filepath:
                override_count += 1

            try:
                with open(source_filepath, "rb") as f:
                    img_bytes = f.read()
            except Exception as e:
                print(f"读取失败: {source_filepath}, 错误: {e}")
                continue

            yield {
                "image": {"bytes": img_bytes, "path": filename},
                "model": model,
                "main_category": main_cat,
                "sub_category": sub_cat,
                "filename": filename,
            }

        if override_count:
            print(
                f"  使用 ethics 覆盖文件: {override_count} 张 "
                f"({model}/{main_cat}/{sub_cat})"
            )

    return generator


def process_and_pack():
    if not os.path.exists(BASE_DIR):
        print(f"Error: 找不到基础目录 {BASE_DIR}")
        return

    models = sorted(
        d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))
    )

    for model in models:
        model_dir = os.path.join(BASE_DIR, model)
        main_cats = sorted(
            d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))
        )

        for main_cat in main_cats:
            main_cat_dir = os.path.join(model_dir, main_cat)
            
            # 获取所有的子文件夹
            sub_cats_dirs = sorted(
                d
                for d in os.listdir(main_cat_dir)
                if os.path.isdir(os.path.join(main_cat_dir, d))
            )

            # 核心修改：动态构建目标列表
            targets = []
            if sub_cats_dirs:
                # 存在子文件夹的情况 (例如: AnimalFace -> cat, dog)
                for sc in sub_cats_dirs:
                    targets.append((sc, os.path.join(main_cat_dir, sc)))
            else:
                # 不存在子文件夹的情况 (例如: COCO, ImageNet-1k)
                # 直接将 main_cat 作为子类名，路径就是 main_cat_dir 本身
                targets.append((main_cat, main_cat_dir))

            for sub_cat_name, target_path in targets:
                out_dir = os.path.join(PARQUET_OUT_BASE, "data", model, main_cat)
                os.makedirs(out_dir, exist_ok=True)
                parquet_path = os.path.join(out_dir, f"{sub_cat_name}.parquet")

                # 这里的 check 逻辑会自动跳过已经成功生成的旧文件
                if os.path.exists(parquet_path):
                    file_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
                    if file_size_mb < MIN_VALID_PARQUET_MB:
                        print(
                            f"发现异常小文件 {parquet_path} "
                            f"({file_size_mb:.2f} MB)，正在删除重建..."
                        )
                        os.remove(parquet_path)
                    else:
                        print(
                            f"⏭️ 跳过已存在且有效的大文件: {parquet_path} "
                            f"({file_size_mb:.2f} MB)"
                        )
                        continue

                has_images = any(is_image_file(f) for f in os.listdir(target_path))
                if not has_images:
                    continue

                print(f"\n正在打包: Model={model} | Cat={main_cat}/{sub_cat_name} ...")

                gen = create_image_generator(target_path, model, main_cat, sub_cat_name)
                ds = Dataset.from_generator(gen, features=features)
                ds.to_parquet(parquet_path)

                new_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
                print(
                    f"✅ 保存成功: {parquet_path} "
                    f"(包含图片: {len(ds)} 张, 真实体积: {new_size_mb:.2f} MB)"
                )


if __name__ == "__main__":
    print(f"{'=' * 50}\n🚀 开始执行 ethics 覆盖版数据集 Parquet 打包\n{'=' * 50}")
    print(f"BASE_DIR={BASE_DIR}")
    print(f"ETHICS_OVERRIDE_BASE={ETHICS_OVERRIDE_BASE}")
    print(f"PARQUET_OUT_BASE={PARQUET_OUT_BASE}")
    process_and_pack()
    print(f"\n{'=' * 50}\n🎉 全部数据已成功切片并封装为 Parquet！\n{'=' * 50}")