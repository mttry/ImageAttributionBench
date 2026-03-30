---
license: apache-2.0
pipeline_tag: text-to-image
library_name: transformers
---

## NextStep-1.1

[Homepage](https://stepfun.ai/research/en/nextstep-1)&nbsp;
| [GitHub](https://github.com/stepfun-ai/NextStep-1)&nbsp;
| [Paper](https://arxiv.org/abs/2508.10711)&nbsp;

We introduce **NextStep-1.1**, a new model represents a significant leap forward in the NextStep series. This version effectively resolves the visualization failures seen in **NextStep-1** and substantially elevates image quality through extended training and a Flow-based Reinforcement Learning (RL) post-training paradigm.

<div align='center'>
<img src="assets/comparision.png" class="interpolation-image" alt="arch." width="100%" />
</div>

## What's New in 1.1?

NextStep-1.1 is not just a fine-tune; it is a re-engineered version focused on stability and high-fidelity output. Key improvements include:

- RL Enhanced Visual Fidelity: Significant improvement in image texture and a substantial reduction in visual artifacts via RL, ensuring much cleaner and more professional outputs.

- Technical Stability: Solves numerical instability inherent in the RL of autoregressive flow-based models.

## Environment Setup

To avoid potential errors when loading and running your models, we recommend using the following settings:

```shell
conda create -n nextstep python=3.11 -y
conda activate nextstep

pip install uv # optional

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/stepfun-ai/NextStep-1.1 && cd NextStep-1.1
uv pip install -r requirements.txt

hf download stepfun-ai/NextStep-1.1 "vae/checkpoint.pt" --local-dir ./
```

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModel
from models.gen_pipeline import NextStepPipeline

HF_HUB = "stepfun-ai/NextStep-1.1"

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(HF_HUB, local_files_only=True, trust_remote_code=True)
model = AutoModel.from_pretrained(HF_HUB, local_files_only=True, trust_remote_code=True)
pipeline = NextStepPipeline(tokenizer=tokenizer, model=model).to(device="cuda", dtype=torch.bfloat16)

# set prompts
positive_prompt = ""
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
example_prompt = "A REALISTIC PHOTOGRAPH OF A WALL WITH \"TOWARD AUTOREGRESSIVE IMAGE GENERATION WITH CONTINUOUS TOKENS AT SCALE\" PROMINENTLY DISPLAYED"

# generate image from text
IMG_SIZE = 512
image = pipeline.generate_image(
    example_prompt,
    hw=(IMG_SIZE, IMG_SIZE),
    num_images_per_caption=1,
    positive_prompt=positive_prompt,
    negative_prompt=negative_prompt,
    cfg=7.5,
    cfg_img=1.0,
    cfg_schedule="constant",
    use_norm=False,
    num_sampling_steps=28,
    timesteps_shift=1.0,
    seed=3407,
)[0]
image.save("./assets/output.jpg")
```

## Citation

If you find NextStep useful for your research and applications, please consider starring this repository and citing:

```bibtex
@article{nextstepteam2025nextstep1,
  title={NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale},
  author={NextStep Team and Chunrui Han and Guopeng Li and Jingwei Wu and Quan Sun and Yan Cai and Yuang Peng and Zheng Ge and Deyu Zhou and Haomiao Tang and Hongyu Zhou and Kenkun Liu and Ailin Huang and Bin Wang and Changxin Miao and Deshan Sun and En Yu and Fukun Yin and Gang Yu and Hao Nie and Haoran Lv and Hanpeng Hu and Jia Wang and Jian Zhou and Jianjian Sun and Kaijun Tan and Kang An and Kangheng Lin and Liang Zhao and Mei Chen and Peng Xing and Rui Wang and Shiyu Liu and Shutao Xia and Tianhao You and Wei Ji and Xianfang Zeng and Xin Han and Xuelin Zhang and Yana Wei and Yanming Xu and Yimin Jiang and Yingming Wang and Yu Zhou and Yucheng Han and Ziyang Meng and Binxing Jiao and Daxin Jiang and Xiangyu Zhang and Yibo Zhu},
  journal={arXiv preprint arXiv:2508.10711},
  year={2025}
}
```
