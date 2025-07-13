
#download_rawdata.py
import os
import requests

# URL of the Tox21 dataset CSV
url = "https://tripod.nih.gov/tox21/challenge/data/tox21.csv"

# Local path to save the file
save_dir = "data/raw"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "drug_data.csv")

def download_file(url, path):
    print(f"Downloading from {url} ...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
        print(f"File saved to {path}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

if __name__ == "__main__":
    download_file(url, save_path)
