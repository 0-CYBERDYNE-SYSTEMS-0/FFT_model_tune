#!/usr/bin/env python3
"""
Download Trinity Nano Preview model for agricultural fine-tuning
"""

import os
from huggingface_hub import snapshot_download

def download_trinity_model():
    """Download the Trinity Nano Preview model"""
    model_id = "arcee-ai/Trinity-Nano-Preview"
    
    print(f"Downloading {model_id}...")
    print("This may take a few minutes depending on your internet connection...")
    
    try:
        # Download the model
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir="models/trinity-nano-preview",
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Model downloaded successfully to: {local_dir}")
        print("Model is ready for fine-tuning!")
        
        # List the downloaded files
        for root, dirs, files in os.walk("models/trinity-nano-preview"):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"  📄 {file} ({file_size:.1f} MB)")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False
    
    return True

if __name__ == "__main__":
    download_trinity_model()
