import http.client
from io import BytesIO
import json
import os
import sys

# import concurrent
import concurrent.futures 
import requests
sys.path.append("/home/final_dataset_generator/t2i_generator/API") # TODO: modify here
from kling_model import retry
from key import api_key,gemini
from PIL import Image  

api_key = gemini
class GeminiModel:  
    def __init__(self, api_key=api_key):  
        self.api_key = api_key  
        # self.conn = http.client.HTTPSConnection("yunwu.ai")  
        self.log_file = "gemini_risk_control_failures.log"  
        self.name = "gemini"
        self.model_name = "gemini-2.0-flash-exp-image-generation"

    @retry((http.client.HTTPException, requests.RequestException, json.JSONDecodeError, RuntimeError), tries=-1 , delay=2, backoff=1.1)  
    def submit_task(self, prompt):  
        conn = http.client.HTTPSConnection("yunwu.ai")   
        prompt_prefix = "MUST generate a realistic and natural style image **exactly** with a resolution of 1024x1024 pixels based on this prompt: "  
        payload = json.dumps({  
            "max_tokens": 4096,  
            "model": "gemini-2.0-flash-exp-image-generation",  
            "messages": [  
                {  
                    "role": "user",  
                    "content": [  
                        {  
                            "type": "text",  
                            "text": prompt_prefix + prompt  
                        }  
                    ]  
                }  
            ]  
        })  
        headers = {  
            'Accept': 'application/json',  
            'Authorization': f'Bearer {self.api_key}',  
            'Content-Type': 'application/json'  
        }  
        conn.request("POST", "/v1/chat/completions", payload, headers)  
        res = conn.getresponse()  
        data = res.read()  
        json_data = json.loads(data.decode("utf-8"))  
        print("submit_task response:", json_data)  

        error = json_data.get("error")  
        if error :  
            print(error)
            raise RuntimeError(f"error: {error}")  

        return json_data  

    def parse_image_urls_from_content(self, content_str):  
        import re  
        urls = re.findall(r"\((https?://[^\)]+)\)", content_str)  
        return urls  

    @retry((requests.RequestException, IOError), tries=3)  
    def load_image_from_url(self, url):  
        response = requests.get(url)  
        response.raise_for_status()  
        image_data = BytesIO(response.content)  
        img = Image.open(image_data)  
        return img  

    def generate(self, prompt, num_images=2):  
        imgs = []  

        def submit_and_load(_):  
            data = self.submit_task(prompt)  
            if "error" in data:  
                print(f"❌ task submit failure:{data.get('error', 'unknown error')} — prompt: {prompt}")  
                return []  

            choices = data.get("choices", [])  
            if not choices:  
                print(f"❌ no response result:{data}")  
                return []  

            content = choices[0].get("message", {}).get("content", "")  
            image_urls = self.parse_image_urls_from_content(content)  
            if not image_urls:  
                print(f"❌ not url:{content}")  
                return []  

            loaded_imgs = []  
            for url in image_urls:  
                try:  
                    img = self.load_image_from_url(url)  
                    loaded_imgs.append(img)  
                except Exception as e:  
                    print(f"❌ image load failure {url} error: {e}")  
            return loaded_imgs  

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_images) as executor:  
            futures = [executor.submit(submit_and_load, i) for i in range(num_images)]  
            for future in concurrent.futures.as_completed(futures):  
                images = future.result()  
                imgs.extend(images)  

        return imgs  
    

if __name__ == "__main__":
    gemini = GeminiModel(api_key)  
    prompt = "A close-up of a cat's face with striking blue eyes and a pink nose. The cat has thick fur and is sitting on a cream-colored blanket."
    images = gemini.generate(prompt)  
    dir = "imgs"
    for i, img in enumerate(images):  
        img.save(f"{dir}/gemini_output_{i}.png")  