import sys
import os
from tqdm import tqdm
import requests

"""
Credit to:
https://github.com/lantunes/CrystaLLM
Modified to automatically download crystallm_v1_small.tar.gz to pre-trained directory
"""

STORAGE_URL = "https://zenodo.org/records/10642388/files"
BLOCK_SIZE = 1024
DEFAULT_DIR = "pre-trained"
DEFAULT_MODEL = "crystallm_v1_large.tar.gz"

def get_out_path():
    """Create pre-trained directory if it doesn't exist and return output path."""
    if not os.path.exists(DEFAULT_DIR):
        print(f"Creating directory: {DEFAULT_DIR}")
        os.makedirs(DEFAULT_DIR)
    return os.path.join(DEFAULT_DIR, DEFAULT_MODEL)

def download_model():
    """Download CrystaLLM small model to pre-trained directory."""
    url = f"{STORAGE_URL}/{DEFAULT_MODEL}"
    out_path = get_out_path()
    
    print(f"Downloading {DEFAULT_MODEL} to {out_path} ...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        
        with open(out_path, "wb") as f:
            for data in response.iter_content(BLOCK_SIZE):
                progress_bar.update(len(data))
                f.write(data)
        
        progress_bar.close()
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("Error downloading!")
            sys.exit(1)
            
        print("Download completed successfully!")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

if __name__ == "__main__":
    download_model()
