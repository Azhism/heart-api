#!/usr/bin/env python3
from huggingface_hub import HfApi
import os
import shutil

# Initialize API
api = HfApi()
hf_token = os.getenv("HF_TOKEN")  # Load from environment variable

# Files to upload
files_to_upload = [
    (r"D:\Classess\ITA\Rag_Project\Heart Attack Project\flask_app.py", "flask_app.py"),
    (r"D:\Classess\ITA\Rag_Project\Heart Attack Project\Dockerfile", "Dockerfile"),
    (r"D:\Classess\ITA\Rag_Project\Heart Attack Project\bm25_data.pkl", "bm25_data.pkl"),
]

# Upload individual files
print("Uploading critical files to HF Space...")
for local_path, remote_name in files_to_upload:
    if os.path.exists(local_path):
        try:
            print(f"\n📤 Uploading {remote_name}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_name,
                repo_id="azhab/heart-attack-assessment",
                repo_type="space",
                token=hf_token,
                commit_message=f"Upload {remote_name}"
            )
            print(f"✅ {remote_name} uploaded!")
        except Exception as e:
            print(f"❌ {remote_name} failed: {e}")
    else:
        print(f"⚠️  {remote_name} not found at {local_path}")

# Upload dist folder (React build)
dist_path = r"D:\Classess\ITA\Rag_Project\Heart Attack Project\heart-health-ai\dist"

print(f"\n📤 Uploading React dist folder...")
try:
    api.upload_folder(
        repo_id="azhab/heart-attack-assessment",
        folder_path=dist_path,
        repo_type="space",
        path_in_repo="dist",
        token=hf_token,
        commit_message="Upload React build dist files - fix 404 errors"
    )
    print("✅ dist folder uploaded!")
except Exception as e:
    print(f"❌ dist folder upload failed: {e}")
