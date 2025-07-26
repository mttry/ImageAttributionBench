from enum import Enum
class SDVersion(Enum):
    SD1_5 = ("1.5", "runwayml/stable-diffusion-v1-5")
    SD2_1 = ("2.1", "stabilityai/stable-diffusion-2-1")
    SD3 = ("3", "stabilityai/stable-diffusion-3-medium-diffusers") 
    SDXL = ("xl", "stabilityai/stable-diffusion-xl-base-1.0")
    PLAYGROUND_2_5 = ("playground", "playgroundai/playground-v2.5-1024px-aesthetic")
    PIXART = ("pixart", "PixArt-alpha/PixArt-XL-2-1024-MS")
    KANDINSKY = ("kandinsky","kandinsky-community/kandinsky-3")
    # FLUX = ("flux", "black-forest-labs/FLUX.1-dev")
    FLUX = ("flux", "black-forest-labs/FLUX.1-schnell")
    SD3_5 = ("3.5", "stabilityai/stable-diffusion-3.5-medium")
    CogView3_PLUS = ("cogview3_p","THUDM/CogView3-Plus-3B")
    def __init__(self, version_name: str, model_name: str):
        self.short_name = self.version_name = version_name
        self.model_name = model_name

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
)
from typing import List, Optional, Union
import torch
from PIL import Image



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
                # "use_safetensors":True,
                # "device_map":auto  
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
                # "variant": "fp16",
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
            # pipe.enable_vae_slicing()            
            # pipe.enable_vae_tiling()             
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
            # pipe.to(torch.float16)
            pipe.enable_model_cpu_offload()  
            # pipe.enable_vae_slicing()           
            # pipe.enable_vae_tiling()            
            # pipe.enable_attention_slicing()      
            # self.generator = torch.Generator(device="cpu").manual_seed(0)
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
        }

        params = {
            **version_defaults[self.version], 
            **kwargs,
            "negative_prompt": kwargs.get('negative_prompt') or version_defaults[self.version]["negative_prompt"]
        }

        if isinstance(prompt, str):
            prompt = [prompt] * num_images
            if params["negative_prompt"] is not None:
                params["negative_prompt"] = [params["negative_prompt"]] * num_images

        if self.version == SDVersion.SD3:
            images = []
            for i in range(num_images):
                result = self.pipeline(
                    prompt=[prompt[i]],
                    negative_prompt=[params["negative_prompt"][i]] if params["negative_prompt"] else None,
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    width=params["width"],
                    height=params["height"]
                )
                images.append(result.images[0])
            return images
        elif self.version == SDVersion.PIXART:
            results = self.pipeline(
                prompt=prompt,
                # negative_prompt=params["negative_prompt"],
                # num_inference_steps=params["num_inference_steps"],
                # guidance_scale=params["guidance_scale"],
                # width=params["width"],
                # height=params["height"]
            )
            return results.images[:num_images]

        elif self.version == SDVersion.KANDINSKY:
            # images = []
            # for i in range(num_images):
            #     result = self.pipeline(
            #         prompt=[prompt[i]],
            #         generator=params["generator"],
            #         # negative_prompt=params["negative_prompt"],
            #         # num_inference_steps=params["num_inference_steps"],
            #         # guidance_scale=params["guidance_scale"],
            #         # width=params["width"],
            #         # height=params["height"]
            #     )
            #     images.append(result.images[0])
            # return images

            results = self.pipeline(
                    prompt=prompt,
                    generator=params["generator"],
                    # negative_prompt=params["negative_prompt"],
                    # num_inference_steps=params["num_inference_steps"],
                    # guidance_scale=params["guidance_scale"],
                    width=params["width"],
                    height=params["height"]
                )
            return results.images[:num_images]
        elif self.version == SDVersion.FLUX:
            results = self.pipeline(
                prompt=prompt,
                # negative_prompt=params["negative_prompt"],
                num_inference_steps=params["num_inference_steps"],
                guidance_scale=params["guidance_scale"],
                width=params["width"],
                height=params["height"],
                max_sequence_length=params["max_sequence_length"],
                generator=params["generator"],
            )
            return results.images[:num_images]
        elif self.version == SDVersion.CogView3_PLUS:
            results = self.pipeline(
                prompt=prompt[0],
                negative_prompt=params["negative_prompt"][0],
                num_inference_steps=params["num_inference_steps"],
                guidance_scale=params["guidance_scale"],
                width=params["width"],
                height=params["height"],
                num_images_per_prompt=num_images,
                # max_sequence_length=params["max_sequence_length"],
                # generator=params["generator"],
            )
            return results.images[:num_images]
        else:
            results = self.pipeline(
                prompt=prompt,
                negative_prompt=params["negative_prompt"],
                num_inference_steps=params["num_inference_steps"],
                guidance_scale=params["guidance_scale"],
                width=params["width"],
                height=params["height"]
            )
            return results.images[:num_images]

    def __call__(self, *args,**kwargs):
        return self.generate(*args,**kwargs)

if __name__ == "__main__":
    img_dir = "/home/final_dataset_generator/t2i_generator/imgs"
    import importlib
    import os
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # SD1.5
    # sd15 = StableDiffusionModel("1.5")
    # images = sd15("A watercolor painting of a cat", num_images=10)
    # for i, img in enumerate(images):
    #     img.save(f"{img_dir}/sd15_cat_{i}.png")

    # # # SD3
    # sd3 = StableDiffusionModel(SDVersion.SD3)
    # sd3_images = sd3("A cyberpunk city at night", num_images=5)
    # for i, img in enumerate(sd3_images):
    #     img.save(f"{img_dir}/sd3_cyber_{i}.png")

    # SD2
    # sd2 = StableDiffusionModel("2.1")
    # sd2_images = sd2("A renaissance portrait of a dog", num_images=5)
    # for i, img in enumerate(sd2_images):
    #     img.save(f"{img_dir}/sd21_dog_{i}.png")

    # playground2.5
    # pg = StableDiffusionModel("playground", device="cuda:0")
    # images = pg("A futuristic cityscape", num_images=5)
    # for i, img in enumerate(images):
    #     img.save(f"{img_dir}/pg25_cityscape_{i}.png")

    # pixart
    # pixart = StableDiffusionModel("pixart", device="cuda:0")
    # images = pixart("A futuristic cityscape", num_images=5)
    # for i, img in enumerate(images):
    #     img.save(f"{img_dir}/pixart_cityscape_{i}.png")

    # kandinsky
    # kandinsky = StableDiffusionModel("kandinsky", device="cuda:0")
    # images = kandinsky("A futuristic cityscape", num_images=5)
    # for i, img in enumerate(images):
    #     img.save(f"{img_dir}/kandinsky_cityscape_{i}.png")

    # flux
    # flux = StableDiffusionModel("flux", device="cuda:0")
    # images = flux("A futuristic cityscape", num_images=2)
    # for i, img in enumerate(images):
    #     img.save(f"{img_dir}/flux_cityscape_{i}.png")

    # sd_3_5
    # sd_3_5 = StableDiffusionModel("3.5", device="cuda:0")
    # images = sd_3_5("A futuristic cityscape", num_images=2)
    # for i, img in enumerate(images):
    #     img.save(f"{img_dir}/sd35_cityscape_{i}.png")

    # cogview3_p
    cogview3 = StableDiffusionModel("cogview3_p", device="cuda:4")
    images = cogview3("A futuristic cityscape", num_images=2)
    for i, img in enumerate(images):
        img.save(f"{img_dir}/cogview3_p_cityscape_{i}.png")