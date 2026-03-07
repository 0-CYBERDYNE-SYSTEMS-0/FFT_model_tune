#!/usr/bin/env python3
"""
Download Trinity Nano Preview model for agricultural fine-tuning
"""

import os
import time
from huggingface_hub import snapshot_download
from utils.terminal_ui import spinner, timer, JSONFormatter, print_json_output, track_download_progress

def download_trinity_model():
    """Download the Trinity Nano Preview model with progress tracking"""
    model_id = "arcee-ai/Trinity-Nano-Preview"
    
    print("🌱 Trinity Agricultural AI - Model Download")
    print("=" * 50)
    print(f"📦 Downloading {model_id}...")
    print("This may take a few minutes depending on your internet connection...")
    
    try:
        with timer() as t:
            with spinner("Downloading Trinity model"):
                # Download the model
                local_dir = snapshot_download(
                    repo_id=model_id,
                    local_dir="models/trinity-nano-preview",
                    local_dir_use_symlinks=False
                )
        
        # Create success JSON output
        success_data = {
            "timestamp": JSONFormatter.format_model_response("")["timestamp"],
            "operation": "model_download",
            "model_id": model_id,
            "local_path": local_dir,
            "status": "success",
            "download_duration": t.format_duration(),
            "files_downloaded": []
        }
        
        print(f"✅ Model downloaded successfully to: {local_dir}")
        print(f"⏱️ Download completed in {t.format_duration()}")
        print("Model is ready for fine-tuning!")
        
        # List the downloaded files and track them
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk("models/trinity-nano-preview"):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                total_size += file_size
                file_count += 1
                print(f"  📄 {file} ({file_size:.1f} MB)")
                
                success_data["files_downloaded"].append({
                    "filename": file,
                    "size_mb": round(file_size, 1),
                    "path": file_path
                })
        
        # Add summary statistics
        success_data.update({
            "total_files": file_count,
            "total_size_mb": round(total_size, 1),
            "average_file_size": round(total_size / file_count, 1) if file_count > 0 else 0
        })
        
        # Print JSON output
        print_json_output(success_data, "Download Summary")
        
    except Exception as e:
        # Format error as JSON
        error_data = JSONFormatter.format_error(e, "model_download")
        print(f"❌ Error downloading model: {e}")
        print_json_output(error_data, "Download Error")
        return False
    
    return True

if __name__ == "__main__":
    download_trinity_model()
