import os  
import csv  
import time  
import json  
import http.client  
import threading  
import queue  
import pandas as pd  
import requests  
from typing import Tuple  
from PIL import Image  
from io import BytesIO  
from concurrent.futures import ThreadPoolExecutor  
from key import mj  
import argparse  
import threading  

TEST_MODE = False

CAPTION_DIR = "/home/final_captions"  
OUTPUT_BASE_BASE = "/home/final_dataset"
MAPPING_DIR = "/home/final__mapping"
api_key = API_KEY = mj   
MAX_WORKERS = 1

submit_queue = queue.Queue()  
result_queue = queue.Queue()  

def parse_filename(filename: str) -> Tuple[str, str, str]:  
    base_name = os.path.splitext(filename)[0]  
    if base_name == "imagenet-1k":  
        return "ImageNet-1k", "", base_name  
    if base_name == "COCO":  
        return "COCO", "", base_name  

    parts = base_name.split('_', 1)  
    if len(parts) == 1:  
        return parts[0], "", base_name  

    main_category = parts[0]  
    sub_parts = parts[1].split('_', 1)  

    if main_category == "HumanFace":  
        if "FFHQ" in sub_parts[0]:  
            return main_category, "FFHQ", base_name  
        elif "celebahq" in sub_parts[0].lower():  
            return main_category, "celebahq", base_name  

    if main_category == "Scene" and sub_parts[0].startswith("LSUN"):  
        sub_category = "_".join(sub_parts).replace("LSUN_", "LSUN/")[5:]  
        return main_category, sub_category, base_name  

    if main_category == "AnimalFace":  
        valid_subs = ["cat", "dog", "wild"]  
        for sub in valid_subs:  
            if sub in sub_parts[0].lower():  
                return main_category, sub, base_name  

    return main_category, "_".join(sub_parts), base_name  

def submit_task(prompt, model_version, prompt_args=""): 
    try:  
        payload = json.dumps({  
            "base64Array": [],  
            "notifyHook": "",  
            "prompt": prompt + prompt_args,  
            "state": "",  
            "botType": "MID_JOURNEY"  
        })  

        headers = {  
            'Authorization': f'Bearer {api_key}',  
            'Content-Type': 'application/json'  
        }  

        conn = http.client.HTTPSConnection("yunwu.ai")  
        conn.request("POST", "/mj/submit/imagine", payload, headers)  
        res = conn.getresponse()  
        raw_data = res.read().decode("utf-8")  
        data = json.loads(raw_data)  

        if "code" in data and data["code"] != 1:  
            print(f"Task submission failed: {data}")  
            return None, data  

        task_id = data.get("result", "")  
        if not task_id or not isinstance(task_id, str):  
            print(f"Invalid task_id, submission failed: {data}")  
            return None, data  

        return task_id, data  

    except Exception as e:  
        print(f"Task submission error: {str(e)}")  
        return None, None  

def query_status(task_id: str):  
    conn = http.client.HTTPSConnection("yunwu.ai")  
    headers = {'Authorization': f'Bearer {API_KEY}'}  
    conn.request("GET", f"/mj/task/{task_id}/fetch", headers=headers)  
    res = conn.getresponse()  
    return json.loads(res.read().decode("utf-8"))  

from tqdm import tqdm  

submit_pbar = None  
poll_pbar = None  

def submit_worker_test(model_version):
    global submit_pbar 
    while True:  
        item = submit_queue.get()  
        if item is None:  
            submit_queue.task_done()  
            break  
        prompt, base_filename, mapping_file, caption = item  
        if os.path.exists(base_filename):  
            print(f"Image already exists, skip submission: {base_filename}")  
            submit_queue.task_done()  
            if submit_pbar:  
                submit_pbar.update(1)
        else:
            time.sleep(0.1)
            print(f"task submit: {base_filename}")
            print(f"prompt: {caption}")
            task_id = 114514
            result_queue.put((task_id, base_filename, mapping_file, caption)) 
            submit_queue.task_done()   
            if submit_pbar:  
                submit_pbar.update(1)  

def poll_worker_test():  
    global poll_pbar  
    while True:  
        item = result_queue.get()  
        if item is None:  
            result_queue.task_done()  
            break  
        task_id, save_prefix, mapping_file, caption = item  
        print(f"task downloading to: {save_prefix}")
        time.sleep(1)
        result_queue.task_done()  
        if poll_pbar:  
            poll_pbar.update(1) 

def submit_worker(model_version):  
    global submit_pbar  
    while True:  
        item = submit_queue.get()  
        if item is None:  
            submit_queue.task_done()  
            break  

        prompt, base_filename, mapping_file, caption = item  
        prompt_args = f" --v {model_version} --ar 1:1 --q 2"  

        try:  
            if os.path.exists(base_filename):  
                print(f"Image already exists, skip submission: {base_filename}")  
                if submit_pbar:  
                    submit_pbar.update(1)  
                continue  

            existing_task_id = None  
            if os.path.exists(mapping_file):  
                with open(mapping_file, "r", encoding="utf-8") as f:  
                    reader = csv.reader(f)  
                    header = next(reader, None)  
                    for row in reader:  
                        if len(row) >= 3 and row[1] == caption and row[2].strip():  
                            existing_task_id = row[2].strip()  
                            break  

            if existing_task_id:  
                print(f"Task already exists, reuse TaskID: {existing_task_id}")  
                result_queue.put((existing_task_id, base_filename, mapping_file, caption))  
                if submit_pbar:  
                    submit_pbar.update(1)  
                continue  

            task_id, submit_response = submit_task(prompt, model_version, prompt_args)  
            if task_id:  
                file_exists = os.path.exists(mapping_file)  
                with open(mapping_file, "a", newline='', encoding='utf-8') as f:  
                    writer = csv.writer(f)  
                    if not file_exists:  
                        writer.writerow(["ImagePath", "Caption", "TaskID"])  
                    writer.writerow(["", caption, task_id])  
                result_queue.put((task_id, base_filename, mapping_file, caption))  
                print("Task submitted successfully. TaskID:", task_id)  

        except Exception as e:  
            print("Submit error:", e)  
        finally:  
            submit_queue.task_done()  
            if submit_pbar:  
                submit_pbar.update(1)  

def poll_worker():  
    global poll_pbar  
    while True:  
        item = result_queue.get()  
        if item is None:  
            result_queue.task_done()  
            break  

        task_id, save_prefix, mapping_file, caption = item  
        try:  
            while True:  
                result = query_status(task_id)  
                status = result.get("status")  

                if status == "SUCCESS":  
                    image_url = result.get("imageUrl")  
                    if not image_url:  
                        print(f"Task {task_id} has no image URL")  
                        break  

                    response = requests.get(image_url)  
                    if response.status_code == 200:  
                        image = Image.open(BytesIO(response.content))  
                        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)  
                        image.save(save_prefix)  

                        temp_rows = []  
                        with open(mapping_file, "r", encoding='utf-8') as f:  
                            reader = csv.reader(f)  
                            temp_rows = list(reader)  

                        for i, row in enumerate(temp_rows):  
                            if len(row) > 2 and row[2] == task_id:  
                                row[0] = save_prefix  
                                break  

                        with open(mapping_file, "w", newline='', encoding='utf-8') as f:  
                            writer = csv.writer(f)  
                            writer.writerows(temp_rows)  

                    else:  
                        print(f"Image download failed, status: {response.status_code}")  
                    break  

                elif status in ("FAILURE", "ERROR"):  
                    print(f"Task {task_id} failed or errored")  
                    break  
                else:  
                    time.sleep(5)  
        except Exception as e:  
            print(f"Download error: {e}")  
        finally:  
            result_queue.task_done()  
            if poll_pbar:  
                poll_pbar.update(1)  

def generate_all(model_version):  
    total_tasks = 0  
    batch_size = 500  
    all_prompts = [] 

    for csv_file in os.listdir(CAPTION_DIR):  
        if not csv_file.endswith(".csv"):  
            continue  

        csv_path = os.path.join(CAPTION_DIR, csv_file)  
        main_cat, sub_cat, base_name = parse_filename(csv_file)  
        try:  
            df = pd.read_csv(csv_path)  
            captions = df['Caption'].tolist()  
        except Exception as e:  
            print(f"Unable to read {csv_file}: {e}")  
            continue  

        if main_cat in ["ImageNet-1k", "COCO"]:  
            continue  
        elif main_cat in ["ImageNet-1k-new", "COCO-new"]:  
            num_images_per_prompt = 2  
            main_cat = main_cat[:-4]  
        
        if main_cat == "Scene" and sub_cat not in ["church", "bedroom", "classroom"]:  
            continue  

        print(main_cat, "----", sub_cat)  
        
        mapping_dir = os.path.join(MAPPING_DIR,f"mid-{model_version[-3:]}", main_cat, sub_cat)  
        os.makedirs(mapping_dir, exist_ok=True)  
        mapping_file = os.path.join(mapping_dir, f"{base_name}.csv")  

        for idx, caption in enumerate(captions):  
            for img_idx in range(1):  
                save_dir = os.path.join(OUTPUT_BASE_BASE, f"mid-{model_version[-3:]}", main_cat, sub_cat)  
                os.makedirs(save_dir, exist_ok=True)  
                save_filename = f"{base_name}_p{idx}_i{img_idx}.png"  
                save_path = os.path.join(save_dir, save_filename)  
                all_prompts.append((caption, save_path, mapping_file, caption))  
                total_tasks += 1  

    submit_threads = []  
    poll_threads = []  
    global submit_pbar, poll_pbar  

    MAX_SUBMIT_WORKERS = 1  
    MAX_POLL_WORKERS = 4    

    if TEST_MODE:
        t_submit = threading.Thread(target=submit_worker_test, args=(model_version,), daemon=True)  
    else:
        t_submit = threading.Thread(target=submit_worker, args=(model_version,), daemon=True)  
    t_submit.start()  
    submit_threads.append(t_submit)  

    for _ in range(MAX_POLL_WORKERS):  
        if TEST_MODE:
            t = threading.Thread(target=poll_worker_test, daemon=True)  
        else:
            t = threading.Thread(target=poll_worker, daemon=True)  
        t.start()  
        poll_threads.append(t)  

    submit_batch_size = 200  
    task_interval = 0.1       

    for i in range(0, len(all_prompts), submit_batch_size):  
        batch_prompts = all_prompts[i:i + submit_batch_size]  

        submit_pbar = tqdm(total=len(batch_prompts), desc=f"Submitting batch {i // submit_batch_size + 1}")  
        poll_pbar = tqdm(total=len(batch_prompts), desc=f"Downloading batch {i // submit_batch_size + 1}")  

        for prompt in batch_prompts:  
            submit_queue.put(prompt)  
            time.sleep(task_interval)  

        submit_queue.join()  
        result_queue.join()  

        submit_pbar.close()  
        poll_pbar.close()  

    for _ in range(MAX_SUBMIT_WORKERS):  
        submit_queue.put(None)  
    for _ in range(MAX_POLL_WORKERS):  
        result_queue.put(None)  

    for t in submit_threads:  
        t.join()  
    for t in poll_threads:  
        t.join()  

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="MidJourney Image Generator")  
    parser.add_argument("--version", '-m', type=str, required=True, choices=["6.0", "5.2"],  
                        help="Model version to use (6.0 or 5.2)")  

    args = parser.parse_args()  
    model_version = args.version  

    start = time.time()  
    generate_all(model_version)  
    print(f"All tasks completed! Total time: {time.time() - start:.2f} seconds")  
