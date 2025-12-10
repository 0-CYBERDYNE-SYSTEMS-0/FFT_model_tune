#!/usr/bin/env python3
"""
Trinity Model Download Progress Monitor
Monitors download progress and provides status updates
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime
import subprocess

def get_download_stats():
    """Get current download statistics"""
    model_dir = Path("models/trinity-nano-preview-complete")
    
    if not model_dir.exists():
        return None, None, None
    
    # Count incomplete files
    incomplete_files = list(model_dir.glob(".cache/huggingface/download/*.incomplete"))
    incomplete_count = len(incomplete_files)
    
    # Calculate total downloaded size
    cache_dir = model_dir / ".cache" / "huggingface" / "download"
    total_size = 0
    for file_path in cache_dir.rglob("*.incomplete"):
        try:
            total_size += file_path.stat().st_size
        except:
            pass
    
    # Check for completed files
    completed_files = [
        f for f in model_dir.iterdir() 
        if f.suffix == ".safetensors" and f.name != ".safetensors.lock"
    ]
    completed_count = len(completed_files)
    
    # Expected model files from index
    index_file = model_dir / "model.safetensors.index.json"
    expected_files = 0
    if index_file.exists():
        try:
            with open(index_file) as f:
                index_data = json.load(f)
                expected_files = len(index_data.get("weight_map", {}))
        except:
            pass
    
    return incomplete_count, total_size, expected_files, completed_files

def get_download_speed_stats():
    """Get download speed and time estimates"""
    model_dir = Path("models/trinity-nano-preview-complete")
    cache_dir = model_dir / ".cache" / "huggingface" / "download"
    
    if not cache_dir.exists():
        return None, None
    
    # Get current file sizes and timestamps
    incomplete_files = list(cache_dir.glob("*.incomplete"))
    
    total_size_mb = 0
    recent_files = []
    
    for file_path in incomplete_files:
        try:
            size = file_path.stat().st_size
            size_mb = size / (1024 * 1024)
            total_size_mb += size_mb
            
            # Get file modification time
            mtime = file_path.stat().st_mtime
            recent_files.append((file_path, size_mb, mtime))
        except:
            continue
    
    return total_size_mb, recent_files

def format_size(size_bytes):
    """Format size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def create_progress_report():
    """Create detailed progress report"""
    incomplete_count, total_size, expected_files, completed_files = get_download_stats()
    current_size_mb, recent_files = get_download_speed_stats()
    
    print("📊 Trinity Model Download Monitor")
    print("=" * 50)
    print(f"🕐 Status Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if model_dir := Path("models/trinity-nano-preview-complete"):
        print(f"📁 Model Directory: {model_dir}")
    
    if incomplete_count is None:
        print("❌ Model directory not found!")
        return
    
    print(f"📦 Download Status:")
    print(f"   • Expected model files: {expected_files}")
    print(f"   • Completed files: {len(completed_files) if completed_files else 0}")
    print(f"   • Incomplete downloads: {incomplete_count}")
    
    print(f"📈 Size Progress:")
    if total_size:
        total_size_formatted = format_size(total_size)
        print(f"   • Downloaded so far: {total_size_formatted}")
    
    print(f"📂 Downloaded Files:")
    if completed_files:
        for file_path in completed_files:
            try:
                size = file_path.stat().st_size
                size_mb = size / (1024 * 1024)
                print(f"   ✅ {file_path.name} ({size_mb:.1f} MB)")
            except:
                print(f"   ✅ {file_path.name} (size unknown)")
    else:
        print("   • No completed model files yet")
    
    print(f"🔄 Currently Downloading:")
    if incomplete_count > 0:
        cache_dir = Path("models/trinity-nano-preview-complete/.cache/huggingface/download")
        incomplete_files = list(cache_dir.glob("*.incomplete"))
        
        total_incomplete_mb = 0
        for file_path in incomplete_files:
            try:
                size = file_path.stat().st_size
                size_mb = size / (1024 * 1024)
                total_incomplete_mb += size_mb
                
                # Check file modification time
                mtime = file_path.stat().st_mtime
                time_diff = time.time() - mtime
                
                if time_diff < 300:  # File modified within last 5 minutes
                    status = "🟢 Active"
                else:
                    status = "⚠️ Stalled"
                
                print(f"   {status} {file_path.name} ({size_mb:.1f} MB)")
                
            except Exception as e:
                print(f"   ❓ {file_path.name} (error reading: {e})")
        
        print(f"   📊 Total in progress: {total_incomplete_mb:.1f} MB")
    else:
        print("   • No files currently downloading")
    
    # Estimate completion time
    if incomplete_count > 0 and recent_files:
        print(f"⏱️ Progress Indicators:")
        
        # Check if any files are actively growing
        active_downloads = 0
        for file_path, size_mb, mtime in recent_files:
            time_diff = time.time() - mtime
            if time_diff < 300:  # Modified in last 5 minutes
                active_downloads += 1
        
        if active_downloads > 0:
            print(f"   • {active_downloads} files actively downloading")
            print(f"   • Estimated time remaining: 30-90 minutes")
        else:
            print(f"   • Downloads may be stalled")
            print(f"   • Consider restarting download")
    
    print(f"🎯 Next Steps:")
    if incomplete_count == 0:
        print(f"   ✅ Download appears complete!")
        print(f"   🚀 Ready to test model: python3 test_and_analyze.py")
    else:
        print(f"   ⏳ Monitor for 30-60 more minutes")
        print(f"   🔄 Or restart: huggingface-cli download arcee-ai/Trinity-Nano-Preview")

def check_huggingface_process():
    """Check if HuggingFace CLI is still running"""
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True
        )
        
        huggingface_processes = [
            line for line in result.stdout.split('\n') 
            if 'huggingface' in line.lower() or 'hf' in line.lower()
        ]
        
        if huggingface_processes:
            print(f"🔄 Active HuggingFace processes:")
            for process in huggingface_processes:
                print(f"   {process.strip()}")
        else:
            print(f"😴 No active HuggingFace CLI processes")
            
    except Exception as e:
        print(f"⚠️ Could not check processes: {e}")

def create_monitoring_loop():
    """Create a continuous monitoring loop"""
    print("🔄 Starting continuous monitoring (Ctrl+C to stop)")
    print("-" * 50)
    
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\n📍 Monitor Update #{iteration}")
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
            
            create_progress_report()
            check_huggingface_process()
            
            print(f"\n" + "=" * 50)
            print(f"⏱️ Next update in 30 seconds...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Monitoring stopped by user")
        print(f"💡 Run this script again to resume monitoring")

def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        create_monitoring_loop()
    else:
        create_progress_report()
        
        print(f"\n💡 Monitor Options:")
        print(f"   • Single check: python3 monitor_download.py")
        print(f"   • Continuous monitoring: python3 monitor_download.py --continuous")
        print(f"   • Manual check: ls -la models/trinity-nano-preview-complete/")

if __name__ == "__main__":
    main()
