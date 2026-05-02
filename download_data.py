"""
download_data.py
Downloads the UCI Diabetes 130-US Hospitals dataset automatically.
Run: python download_data.py
"""

import urllib.request
import zipfile
import os

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
ZIP_PATH = "data/dataset_diabetes.zip"
DATA_DIR = "data"

def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading UCI Diabetes 130-US Hospitals dataset...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    print(f"  Downloaded → {ZIP_PATH}")

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    print(f"  Extracted to {DATA_DIR}/")

    # Rename to standard name expected by preprocess.py
    extracted = os.path.join(DATA_DIR, "dataset_diabetes", "diabetic_data.csv")
    target = os.path.join(DATA_DIR, "diabetic_data.csv")
    if os.path.exists(extracted):
        os.rename(extracted, target)
        print(f"  Renamed → {target}")

    print("Done. You can now run: python src/preprocess.py")

if __name__ == "__main__":
    download()
