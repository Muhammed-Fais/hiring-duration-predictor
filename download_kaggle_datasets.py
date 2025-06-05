import kagglehub
import os
from pathlib import Path

# Define the target directory for datasets
# This script is in 'requisition_ttf_forecasting/', so 'data/' is a subdirectory
base_download_path = Path("data/kaggle_hr_dataset")

# Ensure the base download directory exists
base_download_path.mkdir(parents=True, exist_ok=True)

datasets_to_download = [
    "rabieelkharoua/predicting-hiring-decisions-in-recruitment-data",
    "ravindrasinghrana/job-description-dataset"
]

print(f"Starting download process. Datasets will be stored in: {base_download_path.resolve()}\n")

for dataset_slug in datasets_to_download:
    print(f"Downloading {dataset_slug}...")
    try:
        # kagglehub downloads to <path>/<owner_slug>/<dataset_slug>
        # We want to ensure the individual dataset folders are directly under base_download_path
        owner_slug, specific_dataset_slug = dataset_slug.split('/')
        
        # Construct the path where kagglehub will place the files
        # For example: data/kaggle_hr_dataset/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data
        # We want to download *into* a folder named after the dataset directly under 'base_download_path'
        # So, if base_download_path is 'data/kaggle_hr_dataset',
        # and dataset is 'user/name', it should go to 'data/kaggle_hr_dataset/name'
        # dataset_target_path = base_download_path / specific_dataset_slug
        
        # The kagglehub library automatically creates owner_slug/dataset_slug subdirectories
        # within the path provided. So, providing base_download_path is correct.
        path = kagglehub.dataset_download(
            dataset_slug,
            path=str(base_download_path) # Download into the 'data/kaggle_hr_dataset' folder
        )
        print(f"Successfully downloaded {dataset_slug}. Files are in: {path}")
        
        # The actual download path will be <base_download_path>/<owner_slug>/<dataset_slug>
        # Let's print the actual final path clearly
        final_path = base_download_path / owner_slug / specific_dataset_slug
        print(f"Dataset stored at: {final_path.resolve()}\n")
        
    except Exception as e:
        print(f"Could not download {dataset_slug}. Error: {e}\n")

print("Dataset download attempts finished.")
print(f"Please check the directory: {base_download_path.resolve()} for the downloaded files.")
print("If you haven't already, ensure you have 'kagglehub' installed (pip install kagglehub) and are authenticated with Kaggle.") 