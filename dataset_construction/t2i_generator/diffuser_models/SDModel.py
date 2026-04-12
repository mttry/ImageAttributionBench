import sys
from enum import Enum
import torch
from PIL import Image
from typing import List, Optional, Union

from transformers import AutoTokenizer, AutoModel

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    DPMSolverSinglestepScheduler,
    DiffusionPipeline,
    PixArtAlphaPipeline,
    AutoPipelineForText2Image,
    FluxPipeline,
    CogView3PlusPipeline,
    ZImagePipeline,
    Flux2KleinPipeline,
    CogView4Pipeline, # 新增引入 CogView4
)
from diffusers.pipelines.glm_image import GlmImagePipeline # 新增引入 GLM-Image

class SDVersion(Enum):
    SD1_5 = ("1.5", "runwayml/stable-diffusion-v1-5")
    SD2_1 = ("2.1", "stabilityai/stable-diffusion-2-1")
    SD3 = ("3", "stabilityai/stable-diffusion-3-medium-diffusers") 
    SDXL = ("xl", "stabilityai/stable-diffusion-xl-base-1.0")
    PLAYGROUND_2_5 = ("playground", "playgroundai/playground-v2.5-1024px-aesthetic")
    PIXART = ("pixart", "PixArt-alpha/PixArt-XL-2-1024-MS")
    KANDINSKY = ("kandinsky","kandinsky-community/kandinsky-3")
    FLUX = ("flux", "black-forest-labs/FLUX.1-schnell")
    SD3_5 = ("3.5", "stabilityai/stable-diffusion-3.5-medium")
    CogView3_PLUS = ("cogview3_p","THUDM/CogView3-Plus-3B")
    Z_IMAGE = ("z_image", "Tongyi-MAI/Z-Image-Turbo")
    FLUX2_KLEIN = ("flux2_klein", "black-forest-labs/FLUX.2-klein-4B")
    COGVIEW4 = ("cogview4", "THUDM/CogView4-6B")             # 新增 CogView4
    GLM_IMAGE = ("glm_image", "zai-org/GLM-Image")           # 新增 GLM-Image
    NEXTSTEP = ("nextstep", "stepfun-ai/NextStep-1.1")       # 新增 NextStep-1.1

    def __init__(self, version_name: str, model_name: str):
        self.short_name = self.version_name = version_name
        self.model_name = model_name


class StableDiffusionModel:
    def __init__(
        self,
        version: Union[SDVersion, str] = SDVersion.SDXL,
        device: Optional[str] = None,
        **model_kwargs
    ):
        self.version = self._parse_version(version)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = self._load_pipeline(**model_kwargs)
        self.generator = None

    def _parse_version(self, version: Union[SDVersion, str]) -> SDVersion:
        if isinstance(version, SDVersion):
            return version
        for v in SDVersion:
            if version.lower() in [v.version_name.lower(), v.name.lower()]:
                return v
        raise ValueError(f"Unsupported version: {version}")

    def _load_pipeline(self,**kwargs):
        version_config = {
            SDVersion.SD1_5: {
                "torch_dtype": torch.float16,
                "variant": "fp16",
                "safety_checker": None  
            },
            SDVersion.SD2_1: {
                "torch_dtype": torch.float16,
                "variant": "fp16"
            },
            SDVersion.SD3: {
                "torch_dtype": torch.float16, 
                "variant": "fp16",
            },
            SDVersion.SDXL: {
                "torch_dtype": torch.float16,
                "variant": "fp16"
            },
            SDVersion.PLAYGROUND_2_5: {
                "torch_dtype": torch.float16,
                "variant": "fp16"
            },
            SDVersion.PIXART: {
                "torch_dtype": torch.float16,
                "use_safetensors":True,
            },
            SDVersion.KANDINSKY: {
                "torch_dtype": torch.float16,
                "variant": "fp16",
            },
            SDVersion.FLUX: {
                "torch_dtype": torch.bfloat16,
            },
            SDVersion.SD3_5: {
                "torch_dtype": torch.bfloat16,
            },
            SDVersion.CogView3_PLUS:{
                "torch_dtype": torch.bfloat16,
            },
            SDVersion.Z_IMAGE: {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": False,
            },
            SDVersion.FLUX2_KLEIN: { 
                "torch_dtype": torch.bfloat16,
            },
            SDVersion.COGVIEW4: {      # 新增 CogView4 加载配置
                "torch_dtype": torch.bfloat16,
            },
            SDVersion.GLM_IMAGE: {     # 新增 GLM_IMAGE 加载配置
                "torch_dtype": torch.bfloat16,
            },
            SDVersion.NEXTSTEP: {      # NextStep 参数在分支内单独处理
                "torch_dtype": torch.bfloat16, 
            }
        }

        config = {**version_config[self.version],**kwargs}

        if self.version == SDVersion.SD3:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                self.version.model_name,
                **config
            )
        elif self.version == SDVersion.SDXL:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.version.model_name,
                **config
            )
        elif self.version == SDVersion.PLAYGROUND_2_5:
            pipe = DiffusionPipeline.from_pretrained(
                self.version.model_name,
                **config
            )
        elif self.version == SDVersion.PIXART:
            pipe = PixArtAlphaPipeline.from_pretrained(
                self.version.model_name,
                **config
            )
        elif self.version == SDVersion.KANDINSKY:
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.version.model_name,
                **config
            )
            pipe.enable_model_cpu_offload()  
            pipe.enable_attention_slicing()    
            self.generator = torch.Generator(device="cpu").manual_seed(0)
            return pipe
        elif self.version == SDVersion.FLUX:
            pipe = FluxPipeline.from_pretrained(
                self.version.model_name,
                **config
            )
            pipe.to(torch.float16)
            pipe.enable_model_cpu_offload()  
            pipe.enable_vae_slicing()            
            pipe.enable_vae_tiling()             
            pipe.enable_attention_slicing()     
            self.generator = torch.Generator(device="cpu").manual_seed(0)
            return pipe
        elif self.version == SDVersion.SD3_5:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                self.version.model_name,
                **config
            )
            pipe.to(torch.float16)
        elif self.version == SDVersion.CogView3_PLUS:
            pipe = CogView3PlusPipeline.from_pretrained(
                self.version.model_name,
                **config
            )
            pipe.enable_model_cpu_offload()  
            return pipe
        elif self.version == SDVersion.Z_IMAGE: 
            pipe = ZImagePipeline.from_pretrained(
                self.version.model_name,
                **config
            )
            return pipe.to(self.device)
        elif self.version == SDVersion.FLUX2_KLEIN: 
            pipe = Flux2KleinPipeline.from_pretrained(
                self.version.model_name,
                **config
            )
            pipe.enable_model_cpu_offload() 
            self.generator = torch.Generator(device=self.device).manual_seed(0)
            return pipe 
        
        # === 新增模型加载逻辑 ===
        elif self.version == SDVersion.COGVIEW4:
            pipe = CogView4Pipeline.from_pretrained(
                self.version.model_name,
                **config
            ).to(self.device)
            # 开启切片和分块以防止高分辨率 OOM
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            return pipe
            
        elif self.version == SDVersion.GLM_IMAGE:
            pipe = GlmImagePipeline.from_pretrained(
                self.version.model_name,
                **config
            )
            return pipe.to(self.device)
            
        elif self.version == SDVersion.NEXTSTEP:
            import os # 如果文件开头已经导入了 os，这里可以省略
            
            # 1. 获取当前文件 (SDModel.py) 的绝对目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 2. 获取上一级目录 (t2i_generator)
            parent_dir = os.path.dirname(current_dir)
            # 3. 拼接得到 NextStep-1.1 的绝对路径
            nextstep_dir = os.path.join(parent_dir, "NextStep-1.1")
            
            # 动态注入 NextStep 依赖路径
            if nextstep_dir not in sys.path:
                sys.path.append(nextstep_dir)
            
            from models.gen_pipeline import NextStepPipeline
            
            tokenizer = AutoTokenizer.from_pretrained(self.version.model_name, local_files_only=False, trust_remote_code=True)
            model = AutoModel.from_pretrained(self.version.model_name, local_files_only=False, trust_remote_code=True)
            
            # 绑定本地 VAE 路径 (NextStep-1.1/vae)
            model.config.vae_name_or_path = os.path.join(nextstep_dir, "vae")
            pipe = NextStepPipeline(tokenizer=tokenizer, model=model).to(device=self.device, dtype=config["torch_dtype"])
            return pipe
            
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                self.version.model_name,
                **config
            )

        return pipe.to(self.device)

    def generate(
            self,
            prompt: Union[str, List[str]],
            negative_prompt: Optional[str] = None,
            num_images: int = 1,
            **kwargs
        ) -> List[Image.Image]:

            neg_prompt = "cartoon, anime, drawing, painting, unrealistic, deformed features, blurry, low resolution, oversaturated, watermark, text"
            version_defaults = {
                SDVersion.SD1_5: {
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "width": 512,
                    "height": 512,
                    "negative_prompt": neg_prompt
                },
                SDVersion.SD2_1: {
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "width": 768,
                    "height": 768,
                    "negative_prompt": neg_prompt
                },
                SDVersion.SD3: {
                    "num_inference_steps": 28,
                    "guidance_scale": 7.0,
                    "width": 1024,
                    "height": 1024,
                    "negative_prompt":neg_prompt
                },
                SDVersion.SDXL: {
                    "num_inference_steps": 30,
                    "guidance_scale": 5.0,
                    "width": 1024,
                    "height": 1024,
                    "negative_prompt": neg_prompt
                },
                SDVersion.PLAYGROUND_2_5: {
                    "num_inference_steps": 50,
                    "guidance_scale": 3.0,
                    "width": 1024,
                    "height": 1024,
                    "negative_prompt": neg_prompt
                },
                SDVersion.PIXART: {
                    "num_inference_steps": 20,  
                    "guidance_scale": 6.0,
                    "width": 1024,
                    "height": 1024,
                    "negative_prompt": neg_prompt
                },
                SDVersion.KANDINSKY: {
                    "num_inference_steps": 25,
                    "guidance_scale": 4.0,
                    "width": 768,
                    "height": 768,
                    "negative_prompt": neg_prompt,
                    "generator": self.generator  
                },
                SDVersion.FLUX: {
                    "num_inference_steps": 4,
                    "guidance_scale": 3.5,
                    "width": 512,
                    "height": 512,
                    "max_sequence_length": 256,
                    "negative_prompt":neg_prompt,
                    "generator": self.generator  
                },
                SDVersion.SD3_5: {
                    "num_inference_steps": 40,
                    "guidance_scale": 4.5,
                    "width": 1024,
                    "height": 1024,
                    "negative_prompt": neg_prompt
                },
                SDVersion.CogView3_PLUS: {
                    "num_inference_steps": 50,
                    "guidance_scale": 7.0,
                    "width": 1024,
                    "height": 1024,
                    "negative_prompt": neg_prompt
                },
                SDVersion.Z_IMAGE: {
                    "num_inference_steps": 9,
                    "guidance_scale": 0.0,
                    "width": 1024,
                    "height": 1024,
                    "negative_prompt": None 
                },
                SDVersion.FLUX2_KLEIN: {
                    "num_inference_steps": 4,
                    "guidance_scale": 1.0,
                    "width": 1024,
                    "height": 1024,
                    "negative_prompt": None,
                    "generator": self.generator 
                },
                # === 新增模型的默认参数 ===
                SDVersion.COGVIEW4: {
                    "num_inference_steps": 50,
                    "guidance_scale": 3.5,
                    "width": 1024,
                    "height": 1024,
                    "negative_prompt": neg_prompt
                },
                SDVersion.GLM_IMAGE: {
                    "num_inference_steps": 50,
                    "guidance_scale": 1.5,
                    "width": 1024,
                    "height": 1024,
                    "negative_prompt": neg_prompt
                },
                SDVersion.NEXTSTEP: {
                    "num_inference_steps": 28,
                    "guidance_scale": 7.5,
                    "width": 512,   # 保持与你独立脚本中一致
                    "height": 512,
                    "negative_prompt": neg_prompt
                },
            }

            params = {
                **version_defaults[self.version], 
                **kwargs,
                "negative_prompt": kwargs.get('negative_prompt') or version_defaults[self.version]["negative_prompt"]
            }

            # 将 prompt 和 negative_prompt 扩展为列表，方便后续循环索引
            if isinstance(prompt, str):
                prompt = [prompt] * num_images
                if params["negative_prompt"] is not None:
                    params["negative_prompt"] = [params["negative_prompt"]] * num_images

            # === 针对支持原生批量生成的模型 ===
            batch_supported_versions = [
                SDVersion.CogView3_PLUS, 
                SDVersion.COGVIEW4, 
                SDVersion.GLM_IMAGE, 
                SDVersion.NEXTSTEP
            ]
            
            if self.version in batch_supported_versions:
                if self.version == SDVersion.CogView3_PLUS:
                    results = self.pipeline(
                        prompt=prompt[0],
                        negative_prompt=params["negative_prompt"][0] if params["negative_prompt"] else None,
                        num_inference_steps=params["num_inference_steps"],
                        guidance_scale=params["guidance_scale"],
                        width=params["width"],
                        height=params["height"],
                        num_images_per_prompt=num_images,
                    )
                    return results.images[:num_images]
                    
                elif self.version == SDVersion.COGVIEW4:
                    results = self.pipeline(
                        prompt=prompt[0],
                        guidance_scale=params["guidance_scale"],
                        num_images_per_prompt=num_images,
                        num_inference_steps=params["num_inference_steps"],
                        width=params["width"],
                        height=params["height"],
                    )
                    return results.images[:num_images]
                    
                elif self.version == SDVersion.GLM_IMAGE:
                    results = self.pipeline(
                        prompt=prompt[0],
                        height=params["height"],
                        width=params["width"],
                        num_inference_steps=params["num_inference_steps"],
                        guidance_scale=params["guidance_scale"],
                        generator=kwargs.get("generator") or torch.Generator(device=self.device).manual_seed(42),
                        num_images_per_prompt=num_images,
                    )
                    return results.images[:num_images]
                    
                elif self.version == SDVersion.NEXTSTEP:
                    # NextStep 的特定 API 调用方式
                    results = self.pipeline.generate_image(
                        prompt[0], # 对应 example_prompt
                        hw=(params["height"], params["width"]),
                        num_images_per_caption=num_images,
                        positive_prompt="",
                        negative_prompt=params["negative_prompt"][0] if params["negative_prompt"] else "",
                        cfg=params["guidance_scale"],
                        cfg_img=1.0,
                        cfg_schedule="constant",
                        use_norm=False,
                        num_sampling_steps=params["num_inference_steps"],
                        timesteps_shift=1.0,
                        seed=kwargs.get("seed", 3407),
                    )
                    return results[:num_images] # Generate_image 直接返回 List[PIL.Image]

            # === 以下为不支持 num_images_per_prompt，采用 for 循环串行生成的模型 ===
            images = []
            for i in range(num_images):
                current_prompt = [prompt[i]]
                current_negative_prompt = [params["negative_prompt"][i]] if params["negative_prompt"] else None

                if self.version == SDVersion.SD3:
                    result = self.pipeline(
                        prompt=current_prompt,
                        negative_prompt=current_negative_prompt,
                        num_inference_steps=params["num_inference_steps"],
                        guidance_scale=params["guidance_scale"],
                        width=params["width"],
                        height=params["height"]
                    )
                
                elif self.version == SDVersion.PIXART:
                    result = self.pipeline(
                        prompt=current_prompt,
                    )
                    
                elif self.version == SDVersion.KANDINSKY:
                    result = self.pipeline(
                        prompt=current_prompt,
                        generator=params["generator"],
                        width=params["width"],
                        height=params["height"]
                    )

                elif self.version == SDVersion.FLUX:
                    result = self.pipeline(
                        prompt=current_prompt,
                        num_inference_steps=params["num_inference_steps"],
                        guidance_scale=params["guidance_scale"],
                        width=params["width"],
                        height=params["height"],
                        max_sequence_length=params["max_sequence_length"],
                        generator=params["generator"],
                    )

                elif self.version == SDVersion.Z_IMAGE:
                    result = self.pipeline(
                        prompt=current_prompt,
                        height=params["height"],
                        width=params["width"],
                        num_inference_steps=params["num_inference_steps"],
                        guidance_scale=params["guidance_scale"],
                        generator=kwargs.get("generator")
                    )

                elif self.version == SDVersion.FLUX2_KLEIN:
                    result = self.pipeline(
                        prompt=current_prompt,
                        height=params["height"],
                        width=params["width"],
                        num_inference_steps=params["num_inference_steps"],
                        guidance_scale=params["guidance_scale"],
                        generator=params["generator"] or kwargs.get("generator")
                    )

                else:
                    # 默认后备：SD1.5, SD2.1, SDXL, Playground 等经典模型
                    result = self.pipeline(
                        prompt=current_prompt,
                        negative_prompt=current_negative_prompt,
                        num_inference_steps=params["num_inference_steps"],
                        guidance_scale=params["guidance_scale"],
                        width=params["width"],
                        height=params["height"]
                    )
                
                images.append(result.images[0])

            return images    
    def __call__(self, *args,**kwargs):
            return self.generate(*args,**kwargs)

if __name__ == "__main__":
    img_dir = "/home/final_dataset_generator/t2i_generator/imgs"
    import os
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.makedirs(img_dir, exist_ok=True) 

    # 你可以在这里取消注释以测试新加入的模型
    
    # print("Loading NextStep-1.1...")
    # nextstep = StableDiffusionModel("nextstep", device="cuda:0")
    # images = nextstep("A realistic photograph of a cute cat", num_images=2)
    # for i, img in enumerate(images):
    #     img.save(f"{img_dir}/nextstep_test_{i}.png")