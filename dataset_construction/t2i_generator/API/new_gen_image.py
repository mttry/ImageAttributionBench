#!/usr/bin/env python3
import argparse
import base64
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


DEFAULT_BASE_URL = "https://models-proxy.stepfun-inc.com/v1"
DEFAULT_PROMPT = (
    "A cinematic still of a futuristic glass greenhouse filled with bioluminescent "
    "plants at blue hour, high detail, natural lighting."
)
IMAGE_URL_RE = re.compile(r"https?://[^\s\"')>]+", re.IGNORECASE)
MODEL_MIN_PIXELS = {
    "doubao-seedream-5.0-lite": 3686400,
}
FORCE_SINGLE_IMAGE_MODELS = {
    "doubao-seedream-3.0-t2i",
    "doubao-seedream-5.0-lite",
    "gemini-2.5-flash-image",
    "gemini-3-pro-image",
}


class ApiRequestError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 2 images with gpt-image-1.5 using the OpenAI-compatible images API."
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt for image generation.",
    )
    parser.add_argument(
        "--out-dir",
        default="/Users/jyxc-dz-0101303/Desktop/image-attribution-workspace/test",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        # default="gpt-image-1.5",
        default="gpt-image-1",
        help="Image model name.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--size",
        default="1024x1024",
        help="Requested image size.",
    )
    parser.add_argument(
        "--api-format",
        choices=("auto", "images", "chat", "completions"),
        default="auto",
        help="Request format to use. 'auto' prefers /v1/completions for Gemini image models.",
    )
    return parser.parse_args()


def get_api_key() -> str:
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Export API_KEY or OPENAI_API_KEY first.")
    return api_key


def post_json(url: str, payload: dict, api_key: str) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ApiRequestError(exc.code, body) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"API request failed: {exc}") from exc


def fetch_binary(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=180) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        raise SystemExit(f"Image download failed: {exc}") from exc


def sanitize_model_name(model_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name.strip())
    return cleaned.strip("-") or "image-model"


def model_prefers_completions(model_name: str) -> bool:
    lowered = model_name.lower()
    return lowered.startswith("gemini-") and "image" in lowered


def model_requires_single_image_requests(model_name: str) -> bool:
    return model_name.lower() in FORCE_SINGLE_IMAGE_MODELS


def normalize_size_for_model(model_name: str, size: str) -> str:
    min_pixels = MODEL_MIN_PIXELS.get(model_name.lower())
    if not min_pixels:
        return size

    match = re.fullmatch(r"(\d+)x(\d+)", size.strip().lower())
    if not match:
        return size

    width = int(match.group(1))
    height = int(match.group(2))
    if width * height >= min_pixels:
        return size

    side = math.isqrt(min_pixels)
    if side * side < min_pixels:
        side += 1
    return f"{side}x{side}"


def extract_items_from_content(content) -> list[dict]:
    items: list[dict] = []

    if isinstance(content, str):
        for match in IMAGE_URL_RE.findall(content):
            items.append({"url": match})
        return items

    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                for match in IMAGE_URL_RE.findall(part):
                    items.append({"url": match})
                continue
            if not isinstance(part, dict):
                continue

            image_url = part.get("image_url")
            if isinstance(image_url, dict) and image_url.get("url"):
                items.append({"url": image_url["url"]})
            elif isinstance(image_url, str):
                items.append({"url": image_url})

            for key in ("url", "b64_json", "image_base64", "base64"):
                value = part.get(key)
                if isinstance(value, str):
                    normalized_key = "b64_json" if key != "url" else "url"
                    items.append({normalized_key: value})

            inline_data = part.get("inline_data")
            if isinstance(inline_data, dict) and isinstance(inline_data.get("data"), str):
                items.append({"b64_json": inline_data["data"]})

            for text_key in ("text", "content"):
                text_value = part.get(text_key)
                if isinstance(text_value, str):
                    for match in IMAGE_URL_RE.findall(text_value):
                        items.append({"url": match})

    return items


def extract_image_items(response: dict) -> list[dict]:
    if isinstance(response.get("data"), list):
        direct_items = []
        for item in response["data"]:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("b64_json"), str):
                direct_items.append({"b64_json": item["b64_json"]})
            elif isinstance(item.get("url"), str):
                direct_items.append({"url": item["url"]})
        if direct_items:
            return direct_items

    items: list[dict] = []
    for choice in response.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {})
        if isinstance(message, dict):
            items.extend(extract_items_from_content(message.get("content")))
        if isinstance(choice.get("text"), str):
            items.extend(extract_items_from_content(choice["text"]))
        if isinstance(choice.get("content"), (str, list)):
            items.extend(extract_items_from_content(choice["content"]))

    for candidate in response.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content", {})
        if isinstance(content, dict):
            items.extend(extract_items_from_content(content.get("parts")))

    unique_items: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if "b64_json" in item:
            key = ("b64_json", item["b64_json"])
        elif "url" in item:
            key = ("url", item["url"])
        else:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(item)

    return unique_items


def build_images_payload(args: argparse.Namespace) -> dict:
    return {
        "model": args.model,
        "prompt": args.prompt,
        "n": args.n,
        "size": args.size,
    }


def build_chat_payload(model_name: str, prompt: str, size: str) -> dict:
    instruction = (
        "Generate exactly one image for the following prompt. "
        f"Target image size: {size}. "
        "Return the image output directly if this chat endpoint supports image generation.\n\n"
        f"Prompt: {prompt}"
    )
    return {
        "model": model_name,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": instruction,
                    }
                ],
            }
        ],
    }


def build_completions_payload(model_name: str, prompt: str, size: str) -> dict:
    instruction = (
        "Generate exactly one image for the following prompt. "
        f"Target image size: {size}. "
        "Return the image output directly if this completions endpoint supports image generation.\n\n"
        f"Prompt: {prompt}"
    )
    return {
        "model": model_name,
        "prompt": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": instruction,
                    }
                ],
            }
        ],
        "stream": False,
        "max_tokens": 3000,
    }


def generate_via_images(args: argparse.Namespace, base_url: str, api_key: str) -> tuple[dict, list[dict]]:
    response = post_json(
        f"{base_url}/images/generations",
        build_images_payload(args),
        api_key,
    )
    image_items = extract_image_items(response)
    return response, image_items


def generate_via_chat(args: argparse.Namespace, base_url: str, api_key: str) -> tuple[dict, list[dict]]:
    raw_responses: list[dict] = []
    image_items: list[dict] = []

    for _ in range(args.n):
        response = post_json(
            f"{base_url}/chat/completions",
            build_chat_payload(args.model, args.prompt, args.size),
            api_key,
        )
        raw_responses.append(response)
        extracted = extract_image_items(response)
        if not extracted:
            raise SystemExit(
                "Chat response did not contain any image payloads or image URLs. "
                "Check the raw response JSON for the provider-specific format."
            )
        image_items.append(extracted[0])

    return {"object": "list", "data": raw_responses}, image_items


def generate_via_completions(
    args: argparse.Namespace, base_url: str, api_key: str
) -> tuple[dict, list[dict]]:
    raw_responses: list[dict] = []
    image_items: list[dict] = []

    for _ in range(args.n):
        response = post_json(
            f"{base_url}/completions",
            build_completions_payload(args.model, args.prompt, args.size),
            api_key,
        )
        raw_responses.append(response)
        extracted = extract_image_items(response)
        if not extracted:
            raise SystemExit(
                "Completions response did not contain any image payloads or image URLs. "
                "Check the raw response JSON for the provider-specific format."
            )
        image_items.append(extracted[0])

    return {"object": "list", "data": raw_responses}, image_items


def request_images(
    model_name: str,
    prompt: str,
    n: int = 1,
    size: str = "1024x1024",
    api_format: str = "auto",
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[dict, list[dict], str]:
    if n > 1 and model_requires_single_image_requests(model_name):
        merged_items: list[dict] = []
        raw_responses: list[dict] = []
        resolved_format = api_format

        for _ in range(n):
            response_one, items_one, used_format = request_images(
                model_name=model_name,
                prompt=prompt,
                n=1,
                size=size,
                api_format=resolved_format,
                base_url=base_url,
                api_key=api_key,
            )
            if not items_one:
                break
            merged_items.append(items_one[0])
            raw_responses.append(response_one)
            resolved_format = used_format

        if not merged_items:
            raise SystemExit(
                f"API returned no images for model {model_name} when requesting {n} images."
            )

        return (
            {
                "object": "list",
                "data": merged_items,
                "meta": {
                    "request_format": resolved_format,
                    "single_image_mode": True,
                    "raw_response_count": len(raw_responses),
                },
            },
            merged_items,
            resolved_format,
        )

    resolved_size = normalize_size_for_model(model_name, size)
    args = argparse.Namespace(
        model=model_name,
        prompt=prompt,
        n=n,
        size=resolved_size,
        api_format=api_format,
    )
    resolved_api_key = api_key or get_api_key()
    resolved_base_url = (base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")

    request_format = args.api_format
    if request_format == "auto":
        request_format = "completions" if model_prefers_completions(args.model) else "images"

    try:
        if request_format == "chat":
            response, image_items = generate_via_chat(args, resolved_base_url, resolved_api_key)
        elif request_format == "completions":
            response, image_items = generate_via_completions(args, resolved_base_url, resolved_api_key)
        else:
            response, image_items = generate_via_images(args, resolved_base_url, resolved_api_key)
    except ApiRequestError as exc:
        if args.api_format != "auto":
            raise

        if request_format == "completions":
            fallback_formats = ["chat", "images"]
        elif request_format == "chat":
            fallback_formats = ["completions", "images"]
        else:
            fallback_formats = ["completions", "chat"]

        previous_errors = [f"Primary ({request_format}) -> HTTP {exc.status_code}\n{exc.body}"]
        success_format = None

        for fallback_format in fallback_formats:
            try:
                if fallback_format == "chat":
                    response, image_items = generate_via_chat(args, resolved_base_url, resolved_api_key)
                elif fallback_format == "completions":
                    response, image_items = generate_via_completions(
                        args, resolved_base_url, resolved_api_key
                    )
                else:
                    response, image_items = generate_via_images(args, resolved_base_url, resolved_api_key)
                success_format = fallback_format
                request_format = fallback_format
                break
            except ApiRequestError as fallback_exc:
                previous_errors.append(
                    f"Fallback ({fallback_format}) -> HTTP {fallback_exc.status_code}\n{fallback_exc.body}"
                )

        if success_format is None:
            raise SystemExit("API request failed for all formats.\n" + "\n\n".join(previous_errors))

    if n > 1 and len(image_items) < n:
        filled_items = list(image_items)
        topup_format = request_format
        max_topup_attempts = max(4, n * 2)
        attempts = 0

        while len(filled_items) < n and attempts < max_topup_attempts:
            attempts += 1
            try:
                _, extra_items, used_format = request_images(
                    model_name=model_name,
                    prompt=prompt,
                    n=1,
                    size=size,
                    api_format=topup_format,
                    base_url=base_url,
                    api_key=resolved_api_key,
                )
            except ApiRequestError:
                break

            if not extra_items:
                break

            filled_items.append(extra_items[0])
            topup_format = used_format

        image_items = filled_items
        request_format = topup_format

    return response, image_items, request_format


def decode_image_item(item: dict) -> bytes:
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    if item.get("url"):
        return fetch_binary(item["url"])
    raise SystemExit("Image response item has neither b64_json nor url.")


def convert_to_jpeg(image_bytes: bytes) -> bytes:
    if Image is not None:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()

    sips_bin = shutil.which("sips")
    if not sips_bin:
        raise SystemExit(
            "JPEG conversion requires Pillow or the macOS 'sips' command, but neither is available."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "source_image"
        dst_path = Path(tmpdir) / "converted.jpg"
        src_path.write_bytes(image_bytes)

        try:
            subprocess.run(
                [sips_bin, "-s", "format", "jpeg", str(src_path), "--out", str(dst_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"JPEG conversion failed via sips: {exc.stderr.strip()}") from exc

        return dst_path.read_bytes()


def image_bytes_to_pil(image_bytes: bytes):
    if Image is None:
        raise SystemExit("Pillow is required to load generated images into memory.")
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


class OpenAICompatibleImageModel:
    def __init__(
        self,
        model_name: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        size: str = "1024x1024",
        api_format: str = "auto",
    ) -> None:
        self.name = model_name
        self.model_name = model_name
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        self.api_key = api_key
        self.size = size
        self.api_format = api_format

    def generate(self, prompt: str, num_images: int = 2, **kwargs):
        response, image_items, _ = request_images(
            model_name=self.model_name,
            prompt=prompt,
            n=num_images,
            size=kwargs.get("size", self.size),
            api_format=kwargs.get("api_format", self.api_format),
            base_url=kwargs.get("base_url", self.base_url),
            api_key=kwargs.get("api_key", self.api_key),
        )
        if not image_items:
            raise SystemExit(f"API returned no images for model {self.model_name}.")

        images = []
        for item in extract_image_items({"data": image_items}):
            images.append(image_bytes_to_pil(decode_image_item(item)))
        return images


def save_images(response: dict, out_dir: Path, model_name: str) -> list[Path]:
    saved_paths: list[Path] = []
    filename_prefix = sanitize_model_name(model_name)
    for idx, item in enumerate(extract_image_items(response), start=1):
        image_bytes = decode_image_item(item)

        output_path = out_dir / f"{filename_prefix}-{idx}.jpg"
        output_path.write_bytes(convert_to_jpeg(image_bytes))
        saved_paths.append(output_path)
    return saved_paths


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        response, image_items, request_format = request_images(
            model_name=args.model,
            prompt=args.prompt,
            n=args.n,
            size=args.size,
            api_format=args.api_format,
            base_url=args.base_url,
        )
    except ApiRequestError as exc:
        raise SystemExit(f"API request failed: HTTP {exc.status_code}\n{exc.body}") from exc

    response_path = out_dir / f"{sanitize_model_name(args.model)}-{request_format}-response.json"
    response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")

    saved_paths = save_images({"data": image_items}, out_dir, args.model)
    if not saved_paths:
        raise SystemExit("API returned no images.")

    print(f"Saved {len(saved_paths)} image(s):")
    for path in saved_paths:
        print(path)
    print(f"Raw response saved to: {response_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
