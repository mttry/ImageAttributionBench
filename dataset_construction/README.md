# Dataset Construction

This folder contains the data construction pipeline for the Image Attribution Benchmark. It is organized into the following submodules:

## Structure

```
dataset_construction/
├── README.md
├── prompt_generator/
└── t2i_generator/
```

### `prompt_generator/`

This module is responsible for generating high-quality textual prompts for image generation.

- `select_candidate.py`: Samples candidate images from real datasets to be used for caption generation.
- `prompt_template.py`: Provides customized prompt templates tailored to different semantic categories.
- `Qwen_gen.py`: Leverages the vision-language model **Qwen-VL-Chat** to generate high-quality image captions.

### `t2i_generator/`

This module performs text-to-image generation using both open-source models and proprietary APIs.

- `gen_image_one.py`: An integrated script that supports most generation backends and serves as the main entry point for text-to-image generation.
- `diffuser_models/`: Contains latent diffusion models compatible with the 🤗 HuggingFace `diffusers` library.
- `API/`: Stores proprietary model interfaces accessible via API (e.g., MidJourney, Kling, Ideogram, etc.).
- `HiDream-I1-nf4`: A low-VRAM HiDream model repository. Clone it from [here](https://github.com/hykilpikonna/HiDream-I1-nf4.git) and integrate it with the adjustment code we provide.
- `HunyuanDiT`: Clone from the [official HunyuanDiT repository](https://github.com/Tencent-Hunyuan/HunyuanDiT.git) and apply our provided modifications.
- `Infinity`: Clone from the [official Infinity repository](https://github.com/FoundationVision/Infinity) and integrate it with the adjustment code we provide.
- `Janus`: Clone from the [official Janus-Pro repository](https://github.com/deepseek-ai/Janus) and apply the necessary adjustments using our code.
