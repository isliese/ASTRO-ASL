import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("ayuraj/american-sign-language-dataset")

print("Path to dataset files:", path)

# Create asl directory if it doesn't exist
os.makedirs("asl", exist_ok=True)

# Extract the downloaded dataset
for file in os.listdir(path):
    if file.endswith(".zip"):
        shutil.unpack_archive(os.path.join(path, file), "asl")

# Delete numeric folders (0-9)
numeric_folders = [str(i) for i in range(10)]
for folder in numeric_folders:
    folder_path = os.path.join("asl", folder)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

print("Dataset extracted to ./asl with numeric folders removed")
