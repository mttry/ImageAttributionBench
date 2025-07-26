import http.client
import json
import re
import time
import requests
from PIL import Image
import io
import os
import hashlib
from datetime import datetime
import pytz
from enum import Enum
import sys
sys.path.append("/home/final_dataset_generator/t2i_generator/API") # TODO: modify here
from key import api_key,openai_keys
from typing import List

import http.client  
import json  
import requests  
import re  
import io  
from PIL import Image  
from concurrent.futures import ThreadPoolExecutor  
from typing import List  
import socket 

api_config = {
        "dalle": openai_keys,
        "openai": openai_keys
    }

class ModelType(Enum):
    DALLE3 = ("dalle3","dall-e-3")
    # DALLE2 = "dall-e-2"
    # MIDJOURNEY_V5 = "midjourney-v5"
    # MIDJOURNEY_V6 = "midjourney-v6"
    GPT4O = ("4o","gpt-4o")
    def __init__(self, version_name: str, model_name: str):
        self.short_name = self.version_name = version_name
        self.model_name = model_name

class AIClient:
    def __init__(self,  model_type=ModelType.DALLE3, api_keys=api_config ,base_urls=None, ):
        self.api_keys = api_keys or {}
        self.base_urls = base_urls or {}
        self.timezone = pytz.timezone("Asia/Shanghai")
        if model_type == "4o":
            self.model_type = ModelType.GPT4O
        elif model_type == "dalle3":
            self.model_type = ModelType.DALLE3
        self.name, self.model_name = self.model_type.version_name, self.model_type.model_name

    def generate(
        self,
        prompt: str,
        n = 2,
        **kwargs
    ) -> dict:
        model = self.model_type
        model = ModelType(model) if isinstance(model, str) else model
        
        if model == ModelType.DALLE3:
            return self._generate_dalle(prompt, model, n, **kwargs)
        elif model == ModelType.GPT4O:
            return self._generate_gpt4o(prompt, n ,**kwargs)
        else:
            raise ValueError(f"un support model type: {model}")

    def _generate_dalle(self, prompt, model, n, **kwargs):  
        params = {  
            "size": kwargs.get("size", "1024x1024"),  
            "style": kwargs.get("style", "natural"),  
            "quality": kwargs.get("quality", "standard"),  
            "n": n  
        }  

        conn = http.client.HTTPSConnection("yunwu.zeabur.app", timeout=60)  
        payload = json.dumps({  
            "model": model.model_name,  
            "prompt": prompt,  
            **params  
        })  
        headers = {  
            'Authorization': f'Bearer {self.api_keys.get("dalle")}',  
            'Content-Type': 'application/json'  
        }  

        try:  
            conn.request("POST", "/v1/images/generations", payload, headers)  
            response = conn.getresponse()  
        except socket.timeout:  
            raise Exception("request timeout")  
        except Exception as e:  
            raise Exception(f"error: {e}")  

        if response.status != 200:  
            error = json.loads(response.read().decode()).get("error", {})  
            raise Exception(f"DALL-E API Error [{response.status}]: {error.get('message', 'Unknown error')}")  

        data = json.loads(response.read().decode())  
        return self._process_dalle_result(data, **kwargs)  

    def _generate_gpt4o(self, prompt, n: int = 1, **kwargs) -> list:  
        images = []  

        def generate_image(prompt: str):  
            if not prompt.strip():  
                print(f"[Error] Invalid prompt: '{prompt}' (Prompt is empty or whitespace)")  
                return None  

            try:  
                conn = http.client.HTTPSConnection("yunwu.zeabur.app", timeout=60)
                payload = json.dumps({  
                    "stream": False,  
                    "model": "gpt-4o-all",  
                    "messages": [  
                        {  
                            "role": "user",  
                            "content": [  
                                {  
                                    "type": "text",  
                                    "text": f"Generate a realistic and natural style image with a resolution of 1024x1024 pixels based on this prompt: {prompt}"  
                                }  
                            ]  
                        }  
                    ]  
                })  

                headers = {  
                    'Accept': 'application/json',  
                    'Authorization': f'Bearer {self.api_keys.get("openai")}',  
                    'Content-Type': 'application/json'  
                }  

                conn.request("POST", "/v1/chat/completions", payload, headers)  
                res = conn.getresponse()  
                data = res.read().decode("utf-8")  

                response_json = json.loads(data)  
                content = response_json["choices"][0]["message"]["content"]  

                if "No image URL found" in content:  
                    print(content)  
                    raise ValueError("No image URL found in GPT-4o response.")  

                match = re.search(r'(https://[^\s]+?\.webp)', content)  
                if not match:  
                    print(content)  
                    raise ValueError("No valid image URL found in GPT-4o response.")  

                image_url = match.group(1)  

                response = requests.get(image_url, timeout=30)  
                if response.status_code == 200:  
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")  
                    return image  
                else:  
                    raise RuntimeError(f"Image download failed: HTTP {response.status_code}")  

            except socket.timeout:  
                print(f"[Timeout] prompt: {prompt}")  
                return None  
            except requests.exceptions.Timeout:  
                print(f"[Timeout] URL: {image_url}")  
                return None  
            except Exception as e:  
                print(f"[Error] Generation for prompt '{prompt}' failed: {e}")  
                return None  

        with ThreadPoolExecutor(max_workers=n) as executor:  
            future_to_prompt = [executor.submit(generate_image, prompt) for _ in range(n)]  
            for future in future_to_prompt:  
                try:  
                    result = future.result(timeout=120) 
                except Exception as e:  
                    print(f"[Error] Future result timeout or exception: {e}")  
                    result = None  
                if result:  
                    images.append(result)  

        return images   
    def _process_dalle_result(self, data, **kwargs):
        """Process DALL-E response and return Image objects."""
        images = []
        for item in data["data"]:
            image_url = item["url"]
            img_data = requests.get(image_url).content
            image = Image.open(io.BytesIO(img_data))
            images.append(image)
        
        return images


    def _process_gpt4o_result(self, data, **kwargs):
        content = data["choices"][0]["message"]["content"]
        return {
            "text": content,
            "images": self._handle_multimodal_output(content, **kwargs),
            "usage": data.get("usage")
        }
if __name__ == "__main__":
    save_dir = "imgs"
    os.makedirs(save_dir,exist_ok=True)


    client = AIClient("dalle3")

    try:
        images = client.generate(
            prompt="Cyberpunk cat with neon glasses",
            # model=ModelType.DALLE3,
            n = 2,
            # size="1792x1024",
            # style="vivid"
        )
        
        for idx, img in enumerate(images):

            save_path = f"{save_dir}/dalle3_result_{idx+1}.png"
            img.save(save_path)
            print(f"DALL-E3 save to {save_path}")

    except Exception as e:
        print(str(e))


    # GPT-4o
    # client = AIClient(api_key)
    # images = client.generate(
    #     prompt="A fantasy city floating in the sky",
    #     model=ModelType.GPT4O,
    #     n = 2
    # )

    # 
    # for i, img in enumerate(images):
    #     # img.show() 
    #     img.save( f"{save_dir}/gpt4o_generated_{i}.png")
