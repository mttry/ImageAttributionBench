import numpy as np
import PIL.Image

ANY_ASPECT_RATIO = (0, 0)

HW_ASPECT_RATIOS = [
    (8, 32),  # 256
    (9, 28),  # 252
    (10, 25),  # 250
    (11, 23),  # 253
    (12, 21),  # 252
    (13, 19),  # 247
    (14, 18),  # 252
    (15, 17),  # 255
    (16, 16),  # 256
    (17, 15),  # 255
    (18, 14),  # 252
    (19, 13),  # 247
    (21, 12),  # 252
    (23, 11),  # 253
    (25, 10),  # 250
    (28, 9),  # 252
    (32, 8),  # 256
]


def get_ar_base(ars: list[tuple[int, int]] = HW_ASPECT_RATIOS):
    sqrt_products = [round(np.sqrt(h * w)) for h, w in ars]
    return round(np.mean(sqrt_products))


def ar2str(h: int, w: int) -> str:
    return f"{h}*{w}"


def str2ar(s: str) -> tuple[int, int]:
    return tuple(map(int, s.split("*")))

def center_crop_arr_with_buckets(pil_image, ars: list[tuple[int, int]] = HW_ASPECT_RATIOS, crop=True, buckets: list[int] = [256, 512, 768, 1024]):
    """
    Center crop the image to match the closest aspect ratio from the provided list.

    Args:
        pil_image: PIL Image to be cropped
        image_size: Target size for the smaller dimension
        ars: List of aspect ratios as (height, width) tuples

    Returns:
        PIL Image cropped to the closest aspect ratio
    """
    # ar_base = get_ar_base(ars)
    # Get current image dimensions
    width, height = pil_image.size
    
    buckets = sorted(buckets, reverse=True)
    image_size = buckets[-1]

    for bucket in buckets:
        if width * height >= bucket * bucket:
            image_size = bucket
            break

    return center_crop_arr_with_ar(pil_image, image_size, ars, crop)

def center_crop_arr_with_ar(pil_image, image_size: int, ars: list[tuple[int, int]] = HW_ASPECT_RATIOS, crop=True):
    """
    Center crop the image to match the closest aspect ratio from the provided list.

    Args:
        pil_image: PIL Image to be cropped
        image_sizes: Target size for the smaller dimension
        ars: List of aspect ratios as (height, width) tuples

    Returns:
        PIL Image cropped to the closest aspect ratio
    """

    ar_base = get_ar_base(ars)
    assert image_size % ar_base == 0, f"image_size must be divisible by {ar_base}"

    # Get current image dimensions
    width, height = pil_image.size
        
    current_ar = height / width

    # Find the closest aspect ratio
    closest_ar_idx = np.argmin([abs(current_ar - (h / w)) for h, w in ars])
    target_h, target_w = ars[closest_ar_idx]

    if crop:
        target_h, target_w = round(image_size / ar_base * target_h), round(image_size / ar_base * target_w)

        # First, resize the image while maintaining aspect ratio to ensure the smaller dimension is at least the target size
        scale = max(target_h / height, target_w / width)
        new_height = round(height * scale)
        new_width = round(width * scale)
        pil_image = pil_image.resize((new_width, new_height), resample=PIL.Image.LANCZOS)

        arr = np.array(pil_image)
        # Then perform center crop to the target dimensions
        crop_y = (new_height - target_h) // 2
        crop_x = (new_width - target_w) // 2

        return PIL.Image.fromarray(arr[crop_y : crop_y + target_h, crop_x : crop_x + target_w])
    else:
        scale = image_size // ar_base
        return pil_image.resize((round(target_w * scale), round(target_h * scale)), resample=PIL.Image.LANCZOS)
