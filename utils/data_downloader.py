import os
import requests
from tqdm import tqdm

def download_dataset(url,save_path):
    if os.path.exists(save_path):
        print("[INFO] Dataset already exists.")
        return
    print("[INFO] Downloading dataset...")
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    response=requests.get(url,stream=True)
    total=int(response.headers.get('content-length',0))
    with open(save_path,'wb') as file,tqdm(
        desc="Downloading",
        total=total,
        unit='iB',
        unit_scale=True
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size=file.write(data)
            bar.update(size)

    print("[INFO] Download complete.")