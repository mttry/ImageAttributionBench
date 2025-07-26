import os
import pandas as pd
import random

root_dir = '/home/final_dataset/real'
output_dir = '/home/final_captions'
os.makedirs(output_dir, exist_ok=True)

excluded_folders = {'COCO', 'ImageNet-1k'}
max_images = 1000
valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

for current_root, dirs, files in os.walk(root_dir):
    if any(part in excluded_folders for part in current_root.replace(root_dir, '').split(os.sep)):
        continue

    if not dirs:
        rel_path = os.path.relpath(current_root, root_dir)
        csv_name = rel_path.replace(os.sep, '_') + '.csv'
        output_csv = os.path.join(output_dir, csv_name)

        image_files = [os.path.join(current_root, f) for f in files
                       if os.path.splitext(f)[1].lower() in valid_exts]
        if not image_files:
            continue

        selected = random.sample(image_files, min(max_images, len(image_files)))
        df = pd.DataFrame({'ImgPath': selected})
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(selected)} paths to {output_csv}")
