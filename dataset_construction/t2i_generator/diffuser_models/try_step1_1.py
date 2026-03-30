import torch
from transformers import AutoTokenizer, AutoModel
import sys 
sys.path.append("/home/mot2sgh/mts/ImageAttributionBench/dataset_construction/t2i_generator/NextStep-1.1")
from models.gen_pipeline import NextStepPipeline

HF_HUB = "stepfun-ai/NextStep-1.1"

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(HF_HUB, local_files_only=False, trust_remote_code=True)
model = AutoModel.from_pretrained(HF_HUB, local_files_only=False, trust_remote_code=True)
# print(model.config)
model.config.vae_name_or_path = "/home/mot2sgh/mts/ImageAttributionBench/dataset_construction/t2i_generator/NextStep-1.1/vae"
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
    num_images_per_caption=2,
    positive_prompt=positive_prompt,
    negative_prompt=negative_prompt,
    cfg=7.5,
    cfg_img=1.0,
    cfg_schedule="constant",
    use_norm=False,
    num_sampling_steps=28,
    timesteps_shift=1.0,
    seed=3407,
)
image[0].save("./output_0.jpg")
image[1].save("./output_1.jpg")
