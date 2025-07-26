import http.client
from io import BytesIO
import time
from PIL import Image  
import json
import os

import requests
import sys
sys.path.append("/home/final_dataset_generator/t2i_generator/API")  # TODO: modify here
from key import kling
api_key = kling

import http.client  
import json  
import time  
from io import BytesIO  
import requests  
from PIL import Image  

def retry(exceptions, tries=3, delay=2, backoff=2):  
    """Simple retry decorator. Set tries=-1 for infinite retries."""  
    def decorator(func):  
        def wrapper(*args, **kwargs):  
            _tries = tries  
            _delay = delay  
            while _tries != 0:  # exit when _tries = 0  
                try:  
                    return func(*args, **kwargs)  
                except exceptions as e:  
                    if _tries > 0:  
                        print(f"Call to {func.__name__} failed due to: {e}, retrying {_tries-1} more times after {_delay}s")  
                        _tries -= 1  
                    else:  
                        print(f"Call to {func.__name__} failed due to: {e}, retrying indefinitely after {_delay}s")  
                    time.sleep(_delay)  
                    _delay *= backoff  
            # Raise exception after exhausting retries  
            raise e  
        return wrapper  
    return decorator  

class KlingModel:  
    def __init__(self, api_key=api_key):  
        self.name = "kling"  
        self.model_name = "kling-image"  
        self.api_key = api_key  
        self.conn = http.client.HTTPSConnection("yunwu.zeabur.app")  

    @retry((http.client.HTTPException, requests.RequestException, json.JSONDecodeError), tries=3)  
    def query_status(self, task_id):  
        payload = json.dumps({})  
        headers = {  
            'Accept': 'application/json',  
            'Authorization': f'Bearer {self.api_key}',  
            'Content-Type': 'application/json'  
        }  
        self.conn.request("GET", f"/kling/v1/images/generations/{task_id}", payload, headers)  
        res = self.conn.getresponse()  
        data = res.read()  
        json_data = json.loads(data.decode("utf-8"))  
        print("query_status response:", json_data)  
        return json_data  
    
    @retry((requests.RequestException, IOError), tries=3)  
    def load_image_from_url(self, url):  
        response = requests.get(url)  
        response.raise_for_status()  # ensure request success  
        image_data = BytesIO(response.content)  
        img = Image.open(image_data)  
        return img  
    
    @retry((http.client.HTTPException, json.JSONDecodeError), tries=3)  
    def submit_task(self, prompt, num_images):  
        payload = json.dumps({  
            "model": "kling-image",  
            "prompt": prompt,  
            "n": num_images,  
            "negative_prompt": "cartoon, anime, drawing, painting, unrealistic, deformed features, blurry, low resolution, oversaturated, watermark, text, illustration, flat design, vector art, 2D, simple, minimalistic, graphic design, stylized",
            "aspect_ratio": "1:1",  
        })  
        headers = {  
            'Accept': 'application/json',  
            'Authorization': f'Bearer {self.api_key}',  
            'Content-Type': 'application/json'  
        }  
        self.conn.request("POST", "/kling/v1/images/generations", payload, headers)  
        res = self.conn.getresponse()  
        data = res.read()  
        json_data = json.loads(data.decode("utf-8"))  
        print("submit_task response:", json_data)  
        return json_data  

    def generate(self, prompt, num_images=2, max_poll=20, poll_interval=5):  
        log_file = "kling_risk_control_failures.log"  # log file name  
        data = self.submit_task(prompt, num_images)  
        if "code" in data and data["code"] != 0:  
            print(f"❌ Task submission failed: {data.get('msg', 'Unknown error')} — prompt: {prompt}")  
            return []  

        task_id = data.get("data", {}).get("task_id", "")  
        if not task_id:  
            print(f"❌ Invalid task_id, submission failed, raw response: {data}")  
            return []  

        imgs = []  
        for _ in range(max_poll):  
            result = self.query_status(task_id)  
            status = result.get("data", {}).get("task_status", "").lower()  
            msg = result.get("data", {}).get("task_status_msg", "")  
            print(f"Current task status: {status}")  

            if status == "succeed":  
                images_info = result.get("data", {}).get("task_result", {}).get("images", [])  
                for img_info in images_info:  
                    image_url = img_info.get("url")  
                    if image_url:  
                        try:  
                            img = self.load_image_from_url(image_url)  
                            imgs.append(img)  
                        except Exception as e:  
                            print(f"❌ Failed to load image {image_url}, error: {e}")  
                break  

            elif status == "failed":  
                if "risk" in msg.lower() or "risk control" in msg.lower() or "failure to pass the risk control system" in msg.lower():  
                    print(f"❌ Task {task_id} skipped due to risk control failure. Message: {msg}, prompt: {prompt}")  
                    # write to log file  
                    with open(log_file, "a", encoding="utf-8") as f:  
                        f.write(f"task_id: {task_id}\nprompt: {prompt}\nmessage: {msg}\n\n")  
                    return []  
                else:  
                    print(f"❌ Task {task_id} failed. Message: {msg}. Terminating task.")  
                    break  

            elif status in ("failure", "error"):  
                print(f"❌ Task {task_id} failed or error occurred. Message: {msg}")  
                break  

            else:  
                time.sleep(poll_interval)  
        else:  
            print(f"❌ Task timeout. Exceeded maximum polling times: {max_poll}")  

        return imgs  

if __name__ == "__main__":  
    model = KlingModel()  
    images = model.generate("A white and grey cat with green eyes, a pink nose, and whiskers. The cat has a relaxed expression and its fur is soft and fluffy.", num_images=2)  
    dir = "imgs"
    os.makedirs(dir, exist_ok=True)
    for i, img in enumerate(images):  
        img.save(os.path.join(dir, f"kling_{i}.png"))  
