from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image-2512"

# Load the pipeline
import torch

if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
    
    # 获取设备属性
    current_device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(current_device)
    
    # 计算显存（单位：GB）
    total_memory = properties.total_memory / 1024**3
    reserved_memory = torch.cuda.memory_reserved(current_device) / 1024**3
    allocated_memory = torch.cuda.memory_allocated(current_device) / 1024**3
    free_memory = total_memory - allocated_memory # 粗略估算
    
    print(f"设备名称: {properties.name}")
    print(f"总显存: {total_memory:.2f} GB")
    print(f"已分配显存 (Allocated): {allocated_memory:.2f} GB")
    print(f"缓存显存 (Reserved): {reserved_memory:.2f} GB")
else:
    torch_dtype = torch.float32
    device = "cpu"
    print("当前使用 CPU，无法打印 CUDA 显存。")

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)

# Generate image
prompt = '''A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming with vivid, fresh, youthful charm.'''

negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("example_qwen.png")
