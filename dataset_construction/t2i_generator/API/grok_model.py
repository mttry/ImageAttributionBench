import os

from openai import OpenAI
import sys
sys.path.append("/home/final_dataset_generator/t2i_generator/API") # TODO: modify here
from key import grok,api_key


from openai import OpenAI

client = OpenAI(
    base_url="https://yunwu.ai/v1",
    api_key=grok,
    timeout=120
)



import re  
import requests  
from io import BytesIO  
from PIL import Image  

class GrokModel:  
    def __init__(self, client=client):  
        self.client = client  
        self.model_name = self.model = "grok-3-image"  
        self.name = "grok3"

    def generate(self, prompt,num_images=2):  
        messages = [  
            {"role": "system", "content": "You are a helpful assistant."},  
            {"role": "user", "content": f"Please generate a 1024x1024 image with a 1:1 aspect ratio of {prompt}"}  
        ]  
        response = self.client.chat.completions.create(  
            model=self.model,  
            messages=messages  
        )  

        content = response.choices[0].message.content  

        pattern = r'!\[.*?\]\((https?://[^\s\)]+)\)'  
        urls = re.findall(pattern, content)  

        images = []  
        for url in urls:  
            try:  
                resp = requests.get(url)  
                resp.raise_for_status()  
                img = Image.open(BytesIO(resp.content)).convert("RGB")  
                images.append(img)  
            except Exception as e:  
                print(f"image download failure: {url}, error: {e}")  

        return images  


grok = GrokModel()  
prompt = "A white and grey cat with green eyes, a pink nose, and whiskers. The cat has a relaxed expression and its fur is soft and fluffy."  
imgs = grok.generate(prompt)  
img_dir = "imgs"
os.makedirs(img_dir,exist_ok=True)
for idx,img in enumerate(imgs):
    img.save(os.path.join(img_dir, f"img{idx}.png"))