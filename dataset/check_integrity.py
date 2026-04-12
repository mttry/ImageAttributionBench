import os
from typing import List, Tuple

def count_png_per_model_with_total(base_dir: str) -> None:
    """
    Traverses the base directory for all model folders, calculates the total 
    number of .png files per model, and prints the breakdown per subdirectory.
    """
    if not os.path.isdir(base_dir):
        raise NotADirectoryError(f"The provided path is not a valid directory: {base_dir}")

    models = sorted([
        m for m in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, m))
    ])

    for model in models:
        model_path = os.path.join(base_dir, model)
        
        dir_counts: List[Tuple[str, int]] = []
        total_model_pngs = 0
        
        for root, dirs, files in sorted(os.walk(model_path)):
            png_count = sum(1 for f in files if f.endswith(('.png','.jpg','.JPEG')))
            
            if not dirs or png_count > 0:
                rel_path = os.path.relpath(root, model_path)
                dir_counts.append((rel_path, png_count))
                total_model_pngs += png_count
        
        print(f"[{model}] - Total: {total_model_pngs} images")
        
        for rel_path, count in dir_counts:
            if rel_path == ".":
                if count > 0:
                    print(f"  ├── / : {count} pngs")
            else:
                print(f"  ├── {rel_path} : {count} pngs")
        
        print("-" * 50)

parent_directory = "/fs/projects/SGH_CR_RAI-AP_szh-hpc_users/workplace/mot2sgh/ImageAttributionBench"
count_png_per_model_with_total(parent_directory)