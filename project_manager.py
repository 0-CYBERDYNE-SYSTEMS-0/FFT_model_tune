#!/usr/bin/env python3
"""
Project Management Dashboard for Trinity 6B Agricultural Fine-tuning
Tracks progress, manages tasks, and provides status updates
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path
import subprocess

class ProjectManager:
    def __init__(self):
        self.project_root = Path("/Users/scrimwiggins/trin_train")
        self.tasks = {
            "download": {"status": "in_progress", "priority": "high", "progress": 0},
            "model_testing": {"status": "pending", "priority": "medium", "progress": 0},
            "fine_tuning": {"status": "pending", "priority": "medium", "progress": 0},
            "validation": {"status": "pending", "priority": "medium", "progress": 0},
            "documentation": {"status": "pending", "priority": "low", "progress": 0}
        }
        self.milestones = []
        
    def check_download_status(self):
        """Check current download progress"""
        log_file = self.project_root / "download.log"
        
        if not log_file.exists():
            return "No download log found"
            
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                recent_lines = [line for line in lines[-20:] if "Downloading" in line]
                
            if recent_lines:
                return "\n".join(recent_lines[-5:])  # Last 5 download messages
            else:
                return "Download log found but no recent activity"
                
        except Exception as e:
            return f"Error reading download log: {e}"
    
    def check_model_files(self):
        """Check if model files are ready"""
        model_dir = self.project_root / "models" / "trinity-nano-preview-complete"
        
        if not model_dir.exists():
            return False, "Model directory not found"
            
        # Check for completed model files
        safetensors_files = list(model_dir.glob("*.safetensors"))
        if len(safetensors_files) >= 3:
            return True, f"✅ Model files ready ({len(safetensors_files)} files)"
        
        # Check for incomplete files
        incomplete_files = list(model_dir.glob(".cache/huggingface/download/*.incomplete"))
        if incomplete_files:
            total_size = sum(f.stat().st_size for f in incomplete_files if f.exists()) / (1024**3)
            return False, f"🔄 Download in progress: {len(incomplete_files)} files, {total_size:.1f}GB"
            
        return False, "⏳ Model files not found"
    
    def get_download_speed(self):
        """Calculate download progress from log"""
        log_file = self.project_root / "download.log"
        
        if not log_file.exists():
            return None
            
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Extract progress information
            import re
            progress_pattern = r"(\d+\.?\d*)%"
            
            # Look for most recent progress line
            lines = content.split('\n')
            for line in reversed(lines):
                if 'Fetching' in line and '|' in line:
                    match = re.search(progress_pattern, line)
                    if match:
                        return float(match.group(1))
                        
            return None
            
        except Exception:
            return None
    
    def update_project_status(self):
        """Update project status based on current state"""
        model_ready, model_status = self.check_model_files()
        download_status = self.check_download_status()
        
        # Update download task status
        if model_ready:
            self.tasks["download"]["status"] = "completed"
            self.tasks["download"]["progress"] = 100
            
            # Move to next phase
            if self.tasks["model_testing"]["status"] == "pending":
                self.tasks["model_testing"]["status"] = "ready"
        else:
            # Check if download is actively running
            running_processes = subprocess.run(
                ["pgrep", "-f", "huggingface-cli"],
                capture_output=True
            )
            if running_processes.returncode == 0:
                self.tasks["download"]["status"] = "in_progress"
            else:
                self.tasks["download"]["status"] = "stalled"
    
    def create_progress_report(self):
        """Generate comprehensive project report"""
        print("🌾 Trinity 6B Agricultural Fine-tuning Project Dashboard")
        print("=" * 60)
        print(f"📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📍 Project Location: {self.project_root}")
        
        # Download Status
        print(f"\n📥 Model Download Status:")
        print(f"   🆔 Status: {self.tasks['download']['status']}")
        print(f"   📈 Progress: {self.tasks['download']['progress']}%")
        
        download_progress = self.get_download_speed()
        if download_progress:
            print(f"   ⚡ Speed: {download_progress:.1f}% complete")
        
        model_ready, model_status = self.check_model_files()
        print(f"   📊 Details: {model_status}")
        
        # Other Tasks
        print(f"\n🎯 Project Tasks:")
        for task_name, task_info in self.tasks.items():
            status_emoji = {
                "completed": "✅",
                "in_progress": "🔄",
                "ready": "🎯",
                "pending": "⏳",
                "stalled": "⚠️"
            }.get(task_info["status"], "❓")
            
            print(f"   {status_emoji} {task_name.replace('_', ' ').title()}: {task_info['status']} ({task_info['progress']}%)")
        
        # Next Steps
        print(f"\n🚀 Immediate Next Steps:")
        if not model_ready:
            print(f"   1. Monitor download progress")
            print(f"   2. Wait for model files to complete")
            print(f"   3. Run: python3 monitor_download.py --continuous")
        else:
            print(f"   1. Test base model: python3 test_and_analyze.py")
            print(f"   2. Run fine-tuning: python3 fine_tune_agricultural_complete.py")
            print(f"   3. Validate agricultural responses")
        
        # Resource Usage
        print(f"\n💻 System Resources:")
        try:
            result = subprocess.run(
                ["ps", "aux"], capture_output=True, text=True
            )
            huggingface_processes = [
                line for line in result.stdout.split('\n')
                if 'huggingface' in line and 'grep' not in line
            ]
            print(f"   🔄 Active Downloads: {len(huggingface_processes)} processes")
            
            if huggingface_processes:
                for process in huggingface_processes:
                    cpu_usage = process.split()[2] + "%"
                    print(f"      CPU: {cpu_usage}")
                    
        except Exception as e:
            print(f"   ⚠️ Could not check system resources: {e}")
        
        # Files Created
        print(f"\n📁 Project Files:")
        important_files = [
            "comprehensive_agricultural_dataset.json",
            "fine_tune_agricultural_complete.py", 
            "test_and_analyze.py",
            "monitor_download.py",
            "README_setup.md"
        ]
        
        for file_name in important_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                size = file_path.stat().st_size / 1024
                print(f"   ✅ {file_name} ({size:.1f} KB)")
            else:
                print(f"   ❌ {file_name} (missing)")
    
    def monitor_download_continuous(self, interval=30, max_iterations=100):
        """Monitor download with continuous updates"""
        print("🔄 Starting continuous download monitoring")
        print(f"⏱️ Updates every {interval} seconds (Ctrl+C to stop)")
        print("=" * 50)
        
        for iteration in range(max_iterations):
            print(f"\n📍 Monitor Update #{iteration + 1}")
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
            
            self.update_project_status()
            self.create_progress_report()
            
            # Check if download is complete
            model_ready, _ = self.check_model_files()
            if model_ready:
                print(f"\n🎉 DOWNLOAD COMPLETE!")
                print(f"✅ Trinity model files are ready for fine-tuning")
                break
            
            print(f"\n⏳ Continuing monitoring in {interval} seconds...")
            time.sleep(interval)
        else:
            print(f"\n⏰ Monitoring timeout after {max_iterations} iterations")

def main():
    import sys
    
    pm = ProjectManager()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--continuous":
            pm.monitor_download_continuous()
        elif sys.argv[1] == "--update":
            pm.update_project_status()
            pm.create_progress_report()
        else:
            print("Usage: python3 project_manager.py [--continuous|--update]")
    else:
        pm.update_project_status()
        pm.create_progress_report()

if __name__ == "__main__":
    main()
