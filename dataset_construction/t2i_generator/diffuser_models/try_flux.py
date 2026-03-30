import torch
from diffusers import Flux2Pipeline

repo_id = "diffusers/FLUX.2-dev-bnb-4bit" 
device = "cuda:0"
torch_dtype = torch.bfloat16

# 移除 text_encoder=None，让 diffusers 在本地加载它
pipe = Flux2Pipeline.from_pretrained(
    repo_id, 
    torch_dtype=torch_dtype
).to(device)

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell..."

image = pipe(
    prompt=prompt, # 直接传文字 prompt，不需要 prompt_embeds 了
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50, 
    guidance_scale=4,
).images[0]

image.save("flux2_output_local.png")