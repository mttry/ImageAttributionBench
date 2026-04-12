import torch
from diffusers import Flux2KleinPipeline

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell..."

images = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(0),
    num_images_per_prompt=2
).images
images[0].save("flux-klein-0.png")
images[1].save("flux-klein-1.png")
