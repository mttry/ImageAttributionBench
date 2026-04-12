from diffusers import CogView4Pipeline
import torch

# 【关键修改 1】根据你的节点选择精度。
# 如果你确定指定了 05/06 节点（A800），可以用 bfloat16。
# 如果你可能被分配到 01-03 节点（V100），请务必改为 torch.float16！
current_dtype = torch.bfloat16 

# 【关键修改 2】在末尾加上 .to("cuda")，直接把整个模型送进显存
pipe = CogView4Pipeline.from_pretrained(
    "THUDM/CogView4-6B", 
    torch_dtype=current_dtype
).to("cuda")

# 已经关闭 CPU 卸载，模型常驻显存，速度最快
# pipe.enable_model_cpu_offload()

# 【提示】如果在 A800 (80GB) 上，下面这两行也可以注释掉以追求极限速度。
# 但如果在 V100 (32GB) 上生成 1024x1024 报错 OOM，就把这两行开着。
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

prompt = "A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth and flawless, casting a mirror-like reflection. The car features a low, aerodynamic body, angular headlights that gaze forward like predatory eyes, and a set of black, high-gloss racing rims that contrast starkly with the red. A subtle hint of chrome embellishes the grille and exhaust, while the tinted windows suggest a luxurious and private interior. The scene conveys a sense of speed and elegance, the car appearing as if it's about to burst into a sprint along a coastal road, with the ocean's azure waves crashing in the background."

image = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    num_images_per_prompt=2,
    num_inference_steps=50,
    width=1024,
    height=1024,
).images[0]

image.save("cogview4.png")