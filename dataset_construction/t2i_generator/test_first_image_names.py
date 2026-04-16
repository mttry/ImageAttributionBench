#!/usr/bin/env python3
import argparse
import os
from typing import Tuple


DEFAULT_CAPTION_DIR = (
    "/home/i-moutingshu/mts/IA/ImageAttributionBench/dataset_construction/"
    "prompt_generator/downloaded_captions/final_captions_new"
)


def parse_filename(filename: str) -> Tuple[str, str, str]:
    base_name = os.path.splitext(filename)[0]
    if base_name in ["imagenet-1k", "imagenet-1k-new", "COCO", "COCO-new"]:
        return base_name.replace("-new", ""), "", base_name

    parts = base_name.split("_", 1)
    if len(parts) == 1:
        return parts[0], "", base_name

    main_category = parts[0]
    sub_parts = parts[1].split("_", 1)

    if main_category == "HumanFace":
        if "FFHQ" in sub_parts[0]:
            return main_category, "FFHQ", base_name
        if "celebahq" in sub_parts[0].lower():
            return main_category, "celebahq", base_name

    if main_category == "Scene" and sub_parts[0].startswith("LSUN"):
        sub_category = "_".join(sub_parts).replace("LSUN_", "LSUN/")[5:]
        return main_category, sub_category, base_name

    if main_category == "AnimalFace":
        for sub in ["cat", "dog", "wild"]:
            if sub in sub_parts[0].lower():
                return main_category, sub, base_name

    return main_category, "_".join(sub_parts), base_name


def should_skip(base_name: str, main_cat: str, sub_cat: str) -> bool:
    old_base_names = {"imagenet-1k", "COCO"}
    if base_name in old_base_names:
        return True
    if main_cat == "Scene" and sub_cat not in {"church", "bedroom", "classroom"}:
        return True
    return False


def normalize_main_and_base(base_name: str, main_cat: str) -> Tuple[str, str]:
    if base_name == "imagenet-1k-new":
        return "ImageNet-1k", "ImageNet-1k"
    if base_name == "COCO-new":
        return "COCO", "COCO-new"
    return main_cat, base_name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the first expected image filename per processed category without model calls."
    )
    parser.add_argument("--caption-dir", default=DEFAULT_CAPTION_DIR)
    parser.add_argument("--reverse", action="store_true", default=False)
    parser.add_argument("--trancate", action="store_true", default=False)
    args = parser.parse_args()

    csv_files = [f for f in os.listdir(args.caption_dir) if f.endswith(".csv")]
    csv_files.sort()
    if args.reverse:
        csv_files.reverse()
    if args.trancate:
        csv_files = csv_files[9:]

    printed = 0
    for csv_file in csv_files:
        main_cat, sub_cat, base_name = parse_filename(csv_file)
        if should_skip(base_name, main_cat, sub_cat):
            continue

        out_main_cat, out_base_name = normalize_main_and_base(base_name, main_cat)
        first_filename = f"{out_base_name}_p0_i0.png"
        print(f"{out_main_cat}\t{sub_cat}\t{first_filename}")
        printed += 1

    print(f"TOTAL={printed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
