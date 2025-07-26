from concurrent.futures import ThreadPoolExecutor
import http.client
import io
import json
import sys
sys.path.append("/home/final_dataset_generator/t2i_generator/API")  # TODO: modify here
from key import ideo
api_key = ideo

import http.client
from io import BytesIO
import time
from PIL import Image  
import json
import os
import requests

import http.client  
import json  
import time  
from io import BytesIO  
import requests  
from PIL import Image  

def retry(exceptions, tries=3, delay=2, backoff=2):  
    def decorator(func):  
        def wrapper(*args, **kwargs):  
            _tries = tries  
            _delay = delay  
            while _tries != 0:  
                try:  
                    return func(*args, **kwargs)  
                except exceptions as e:  
                    if _tries > 0:  
                        print(f"Call to {func.__name__} failed due to: {e}. Retrying {_tries-1} more times after {_delay}s...")
                        _tries -= 1  
                    else:  
                        print(f"Call to {func.__name__} failed due to: {e}. Retrying indefinitely after {_delay}s...")
                    time.sleep(_delay)  
                    _delay *= backoff  
            raise e  
        return wrapper  
    return decorator  

class IdeoGramModel:  
    def __init__(self, api_key=api_key):  
        self.name = "ideogram"  
        self.model_name = "ideogram_V_1_TURBO"  
        self.api_key = api_key  
        self.conn = http.client.HTTPSConnection("yunwu.zeabur.app")  

    def generate(self, prompt, num_images=2, max_poll=20, poll_interval=5):  
        images = []  
        results = []  

        def download_image_with_retry(url, retries=3):  
            for i in range(retries):  
                try:  
                    response = requests.get(url, timeout=30)  
                    response.raise_for_status()  
                    return Image.open(io.BytesIO(response.content)).convert("RGB")  
                except (requests.exceptions.RequestException, ChunkedEncodingError, IncompleteRead) as e:  
                    print(f"Download failed, retrying ({i+1}): {e}")  
                    time.sleep(2 ** i)  
            return None  

        def generate_image(prompt: str):  
            if not prompt.strip():  
                print(f"[Error] Invalid prompt: '{prompt}' (Prompt is empty or whitespace)")  
                return None  

            try:  
                conn = http.client.HTTPSConnection("yunwu.zeabur.app")  
                payload = json.dumps({  
                    "image_request": {  
                        "aspect_ratio": "ASPECT_1_1",  
                        "magic_prompt_option": "OFF",  
                        "model": "V_1_TURBO",  
                        "prompt": f"a realistic picture of {prompt}",  
                        "negative_prompt": "cartoon, anime, drawing, painting, unrealistic, deformed features, blurry, low resolution, oversaturated, watermark, text, illustration, flat design, vector art, 2D, simple, minimalistic, graphic design, stylized",  
                    }  
                })  
                headers = {  
                    'Accept': 'application/json',  
                    'Authorization': f'Bearer {api_key}',  
                    'Content-Type': 'application/json'  
                }  
                conn.request("POST", "/ideogram/generate", payload, headers)  
                res = conn.getresponse()  
                data = res.read().decode("utf-8")  
                response_json = json.loads(data)  

                print(f"API response: {response_json}")  
                image_url = response_json["data"][0]["url"]  
                image = download_image_with_retry(image_url)  
                if image:  
                    return image  
                else:  
                    raise RuntimeError("Image download failed after retries.")  

            except Exception as e:  
                print(f"[Error] Generation for prompt '{prompt}' failed: {e}")  
                return None  

        # Use thread pool to generate images  
        with ThreadPoolExecutor(max_workers=num_images) as executor:  
            future_to_prompt = [executor.submit(generate_image, prompt) for _ in range(num_images)]  
            for future in future_to_prompt:  
                result = future.result()  
                if result:  # If generation succeeded, add to result list  
                    images.append(result)  

        return images 

if __name__ == "__main__":
    model = IdeoGramModel()  
    images = model.generate("godfree versus godwin in the game Elden Ring", num_images=2)  
    dir = "imgs"
    os.makedirs(dir, exist_ok=True)
    for i, img in enumerate(images):  
        img.save(os.path.join(dir, f"idogram_{i}.png"))  
