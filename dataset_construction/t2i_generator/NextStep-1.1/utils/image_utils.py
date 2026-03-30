import io
import os
from typing import Literal, TypeAlias

import numpy as np
import PIL.Image
import PIL.ImageOps
import requests
import torch

"""
- pil: `PIL.Image.Image`, size (w, h), seamless conversion between `uint8`
- np: `np.ndarray`, shape (h, w, c), default `np.uint8`
- pt: `torch.Tensor`, shape (c, h, w), default `torch.uint8`
"""
ImageType: TypeAlias = PIL.Image.Image | np.ndarray | torch.Tensor
ImageTypeStr: TypeAlias = Literal["pil", "np", "pt"]
ImageFormat: TypeAlias = Literal["JPEG", "PNG"]
DataFormat: TypeAlias = Literal["255", "01", "11"]


IMG_SUPPORT_MODE = ["L", "LA", "RGB", "RGBA", "CMYK", "P", "1"]
IMAGE_EXT_LOWER = ["png", "jpeg", "jpg", "webp"]
IMAGE_EXT = IMAGE_EXT_LOWER + [_ext.upper() for _ext in IMAGE_EXT_LOWER]


def check_image_type(image: ImageType):
    if not (isinstance(image, PIL.Image.Image) or isinstance(image, np.ndarray) or isinstance(image, torch.Tensor)):
        raise TypeError(f"`image` should be PIL Image, ndarray or Tensor. Got `{type(image)}`.")


def to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
    # Automatically adjust the orientation of the image to match the direction it was taken.
    image = PIL.ImageOps.exif_transpose(image)

    if image.mode not in IMG_SUPPORT_MODE:
        raise ValueError(f"Only support mode in `{IMG_SUPPORT_MODE}`, got `{image.mode}`")

    if image.mode == "LA":
        image = image.convert("RGBA")

    # add white background for RGBA images, and convert to RGB
    if image.mode == "RGBA":
        background = PIL.Image.new("RGBA", image.size, "white")
        image = PIL.Image.alpha_composite(background, image).convert("RGB")

    # then convert to RGB
    image = image.convert("RGB")

    return image


def load_image(
    image: str | os.PathLike | PIL.Image.Image | bytes,
    *,
    output_type: ImageTypeStr = "pil",
) -> ImageType:
    """
    Loads `image` to a PIL Image, NumPy array or PyTorch tensor.

    Args:
        image (str | PIL.Image.Image): The path to image or PIL Image.
        mode (ImageMode, optional): The mode to convert to. Defaults to None (no conversion).
            The current version supports all possible conversions between "L", "RGB", "RGBA".
        output_type (ImageTypeStr, optional): The type of the output image. Defaults to "pil".
            The current version supports "pil", "np", "pt".

    Returns:
        ImageType: The loaded image in the given type.
    """
    timeout = 10
    # Load the `image` into a PIL Image.
    if isinstance(image, str) or isinstance(image, os.PathLike):
        if image.startswith("http://") or image.startswith("https://"):
            try:
                image = PIL.Image.open(requests.get(image, stream=True, timeout=timeout).raw)
            except requests.exceptions.Timeout:
                raise ValueError(f"HTTP request timed out after {timeout} seconds")
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://`, `https://` or `s3+[profile]://`, and `{image}` is not a valid path."
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    elif isinstance(image, bytes):
        image = PIL.Image.open(io.BytesIO(image))
    else:
        raise ValueError(f"`image` must be a path or PIL Image, got `{type(image)}`")

    image = to_rgb(image)

    if output_type == "pil":
        image = image
    elif output_type == "np":
        image = to_np(image)
    elif output_type == "pt":
        image = to_pt(image)
    else:
        raise ValueError(f"`output_type` must be one of `{ImageTypeStr}`, got `{output_type}`")

    return image


def to_pil(image: ImageType, image_mode: DataFormat | None = None) -> PIL.Image.Image:
    """
    Convert a NumPy array or a PyTorch tensor to a PIL image.
    """
    check_image_type(image)

    if isinstance(image, PIL.Image.Image):
        return image

    elif isinstance(image, np.ndarray):
        image = normalize_np(image, image_mode)

    elif isinstance(image, torch.Tensor):
        image = normalize_pt(image, image_mode)

        image = image.cpu().permute(1, 2, 0).numpy()
        assert image.dtype == np.uint8, f"Supposed to convert `torch.uint8` to `np.uint8`, but got `{image.dtype}`"

    mode_map = {1: "L", 3: "RGB"}
    mode = mode_map[image.shape[-1]]

    if image.shape[-1] == 1:
        image = image[:, :, 0]

    return PIL.Image.fromarray(image, mode=mode)


def to_np(image: ImageType, image_mode: DataFormat | None = None) -> np.ndarray:
    """
    Convert a PIL image or a PyTorch tensor to a NumPy array.
    """
    check_image_type(image)

    if isinstance(image, PIL.Image.Image):
        image = np.array(image, np.uint8, copy=True)

    if isinstance(image, np.ndarray):
        image = normalize_np(image, image_mode)

    elif isinstance(image, torch.Tensor):
        image = normalize_pt(image, image_mode)

        image = image.cpu().permute(1, 2, 0).numpy()
        assert image.dtype == np.uint8, f"Supposed to convert `torch.uint8` to `np.uint8`, but got `{image.dtype}`"

    return image


def to_pt(image: ImageType, image_mode: DataFormat | None = None) -> torch.Tensor:
    """
    Convert a PIL image or a NumPy array to a PyTorch tensor.
    """
    check_image_type(image)

    if isinstance(image, torch.Tensor):
        image = normalize_pt(image, image_mode)
        return image

    # convert PIL Image to NumPy array
    if isinstance(image, PIL.Image.Image):
        image = np.array(image, np.uint8, copy=True)

    image = normalize_np(image, image_mode)

    image = torch.from_numpy(image.transpose((2, 0, 1))).contiguous()
    assert image.dtype == torch.uint8, f"Supposed to convert `np.uint8` to `torch.uint8`, but got `{image.dtype}`"
    return image


def normalize_np(image: np.ndarray, image_mode: DataFormat | None = None) -> np.ndarray:
    """
    Normalize a NumPy array to the standard format of shape (h, w, c) and uint8.
    """
    if image.ndim not in {2, 3}:
        raise ValueError(f"`image` should be 2 or 3 dimensions. Got {image.ndim} dimensions.")

    elif image.ndim == 2:
        # if 2D image, add channel dimension (HWC)
        image = np.expand_dims(image, 2)

    if image.shape[-1] not in {1, 3}:
        raise ValueError(f"`image` should have 1 (`L`) or 3 (`RGB`) channels. Got {image.shape[-1]} channels.")

    image = to_dataformat(image, image_mode=image_mode, mode="255")

    return image


def normalize_pt(image: torch.Tensor, image_mode: DataFormat | None = None) -> torch.Tensor:
    """
    Normalize a PyTorch tensor to the standard format of shape (c, h, w) and uint8.
    """
    if image.ndimension() not in {2, 3}:
        raise ValueError(f"`image` should be 2 or 3 dimensions. Got {image.ndimension()} dimensions.")

    elif image.ndimension() == 2:
        # if 2D image, add channel dimension (CHW)
        image = image.unsqueeze(0)

    # check number of channels
    if image.shape[-3] not in {1, 3}:
        raise ValueError(f"`image` should have 1 (`L`) or 3 (`RGB`) channels. Got {image.shape[-3]} channels.")

    image = to_dataformat(image, image_mode=image_mode, mode="255")

    return image


def to_dataformat(
    image: ImageType,
    *,
    image_mode: DataFormat | None = None,
    mode: DataFormat = "255",
) -> np.ndarray | torch.Tensor:
    check_image_type(image)

    # convert PIL Image to NumPy array
    if isinstance(image, PIL.Image.Image):
        image = np.array(image, np.uint8, copy=True)
        image_mode = "255"

    # guess image mode
    if image.dtype == np.uint8 or image.dtype == torch.uint8:
        guess_image_mode = "255"
    elif image.dtype == np.float32 or image.dtype == np.float16 or image.dtype == torch.float32 or image.dtype == torch.float16:
        if image.min() < 0.0:
            guess_image_mode = "11"
        else:
            guess_image_mode = "01"
    else:
        raise ValueError(f"Unsupported dtype `{image.dtype}`")

    if image_mode is None:
        image_mode = guess_image_mode
    else:
        if guess_image_mode != image_mode:
            print(f"Guess image mode is `{guess_image_mode}`, but image mode is `{image_mode}`")

    if isinstance(image, np.ndarray):
        if image_mode == "255" and mode != "255":
            np.clip((image.astype(np.float32) / 255), 0, 1, out=image)
            if mode == "11":
                np.clip((image * 2 - 1), -1, 1, out=image)

        elif image_mode == "01" and mode != "01":
            if mode == "255":
                np.clip(image, 0, 1, out=image)
                image = (image * 255).round().astype(np.uint8)
            elif mode == "11":
                np.clip((image * 2 - 1), -1, 1, out=image)

        elif image_mode == "11" and mode != "11":
            np.clip((image / 2 + 0.5), 0, 1, out=image)
            if mode == "255":
                image = (image * 255).round().astype(np.uint8)

    elif isinstance(image, torch.Tensor):
        if image_mode == "255" and mode != "255":
            image = image.to(dtype=torch.float32).div(255).clamp(0, 1)
            if mode == "11":
                image = (image * 2 - 1).clamp(-1, 1)

        elif image_mode == "01" and mode != "01":
            if mode == "255":
                image = image.clamp(0, 1)
                image = (image * 255).round().to(dtype=torch.uint8)
            elif mode == "11":
                image = (image * 2 - 1).clamp(-1, 1)

        elif image_mode == "11" and mode != "11":
            image = (image / 2 + 0.5).clamp(0, 1)
            if mode == "255":
                image = image.mul(255).round().to(dtype=torch.uint8)

    return image


def resize_image(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.BICUBIC)
    return pil_image


def center_crop_arr(pil_image, image_size, crop=True):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    if crop:
        pil_image = resize_image(pil_image, image_size)
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return PIL.Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])
    else:
        # 将图像填充为正方形
        width, height = pil_image.size
        if width != height:
            # 创建一个正方形画布，尺寸为较大的边长
            max_dim = max(width, height)
            padded_img = PIL.Image.new(pil_image.mode, (max_dim, max_dim), (0, 0, 0))
            # 将原图居中粘贴到正方形画布上
            padded_img.paste(pil_image, ((max_dim - width) // 2, (max_dim - height) // 2))
            pil_image = padded_img
        pil_image = resize_image(pil_image, image_size)
        return pil_image
