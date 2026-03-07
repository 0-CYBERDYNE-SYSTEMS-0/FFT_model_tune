#!/usr/bin/env python3
"""
Terminal UI utilities for Trinity Agricultural AI project
Provides progress spinner, operation timer, and JSON formatting features
"""

import time
import json
import threading
import sys
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional


class BrailleSpinner:
    """Non-blocking braille-style progress spinner"""
    
    def __init__(self, message: str = "Processing..."):
        self.message = message
        self._stop_event = threading.Event()
        self._spinner_thread = None
        self._frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self._dots_count = 0
        
    def _spin(self):
        """Internal spinner animation loop - single line only"""
        index = 0
        
        while not self._stop_event.is_set():
            frame = self._frames[index % len(self._frames)]
            self._dots_count = (self._dots_count + 1) % 4
            
            # Create dots animation
            dots = "." * self._dots_count
            
            # Single line update only - use stderr consistently
            sys.stderr.write(f"\r{frame} {self.message}{dots:<3}")
            sys.stderr.flush()
            
            time.sleep(0.2)
            index += 1
        
        # Clear the single line and show completion - use stderr
        sys.stderr.write(f"\r✅ {self.message} completed")
        sys.stderr.flush()
        sys.stderr.write("\n")  # Move to next line for subsequent output
        
    def start(self):
        """Start the spinner animation"""
        if self._spinner_thread is None or not self._spinner_thread.is_alive():
            self._stop_event.clear()
            self._spinner_thread = threading.Thread(target=self._spin, daemon=True)
            self._spinner_thread.start()
    
    def stop(self):
        """Stop the spinner animation"""
        if self._spinner_thread and self._spinner_thread.is_alive():
            self._stop_event.set()
            self._spinner_thread.join()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        # Use stderr consistently for all spinner output
        if exc_type:
            sys.stderr.write(f"\n❌ {self.message} failed: {exc_val}\n")
            sys.stderr.flush()
        else:
            sys.stderr.write(f"\n✅ {self.message} completed\n")
            sys.stderr.flush()


class OperationTimer:
    """Precise operation timer with formatted duration display"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
        
    def start(self):
        """Start timing"""
        self.start_time = time.time()
        self.end_time = None
        self.duration = None
        
    def stop(self):
        """Stop timing and calculate duration"""
        if self.start_time is not None:
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time
            return self.duration
        return None
    
    @property
    def elapsed(self):
        """Get current elapsed time"""
        if self.start_time is None:
            return 0
        end_time = self.end_time if self.end_time else time.time()
        return end_time - self.start_time
    
    def format_duration(self, seconds: Optional[float] = None) -> str:
        """Format duration in H:MM:SS format"""
        if seconds is None:
            seconds = self.elapsed
        
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}:{secs:02d}"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}:{minutes:02d}:{secs:02d}"
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = self.stop()
        if duration is not None:
            formatted = self.format_duration(duration)
            if exc_type:
                print(f"❌ Operation failed after {formatted}")
            else:
                print(f"⏱️ Completed in {formatted}")


class JSONFormatter:
    """Format and structure JSON outputs for LLM operations"""
    
    @staticmethod
    def format_model_response(response: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Format model response into structured JSON"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "response": response.strip(),
            "status": "success"
        }
        
        if metadata:
            result.update(metadata)
            
        return result
    
    @staticmethod
    def format_error(error: Exception, context: Optional[str] = None) -> Dict[str, Any]:
        """Format error into structured JSON"""
        return {
            "timestamp": datetime.now().isoformat(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "context": context
            },
            "status": "error"
        }
    
    @staticmethod
    def format_dataset_generation(category: str, progress: Dict, stats: Dict) -> Dict[str, Any]:
        """Format dataset generation progress into JSON"""
        return {
            "timestamp": datetime.now().isoformat(),
            "operation": "dataset_generation",
            "category": category,
            "progress": progress,
            "statistics": stats,
            "status": "in_progress"
        }
    
    @staticmethod
    def format_training_metrics(epoch: int, loss: float, metrics: Dict) -> Dict[str, Any]:
        """Format training metrics into JSON"""
        return {
            "timestamp": datetime.now().isoformat(),
            "operation": "training_metrics",
            "epoch": epoch,
            "loss": loss,
            "metrics": metrics,
            "status": "success"
        }
    
    @staticmethod
    def to_json(data: Dict[str, Any], indent: int = 2) -> str:
        """Convert data to formatted JSON string"""
        return json.dumps(data, indent=indent, ensure_ascii=False)


@contextmanager
def spinner(message: str):
    """Context manager for progress spinner"""
    with BrailleSpinner(message) as spin:
        yield spin


@contextmanager
def timer():
    """Context manager for operation timer"""
    timer = OperationTimer()
    with timer:
        yield timer


def create_json_system_message() -> str:
    """Create Q: A: system message for dataset generation"""
    system_message = """You are an expert agricultural advisor generating training data for a language model.

Generate agricultural Q&A pairs in this exact format:
Q: [A specific, practical farming question]
A: [A detailed 2-4 sentence answer with actionable advice and specific recommendations]

Quality requirements:
- Use professional agricultural terminology
- Provide practical, actionable advice
- Include specific measurements and timing when relevant
- Reference regional considerations when appropriate
- Ensure accuracy and scientific backing
- Make responses educational and comprehensive

Generate only ONE question-answer pair. Be specific and practical."""
    
    return system_message


def truncate_text(text: str, max_length: int = 80, suffix: str = "...") -> str:
    """Truncate text to a maximum length while preserving word boundaries"""
    if len(text) <= max_length:
        return text
    
    # Try to truncate at a word boundary
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.7:  # Only use word boundary if it's not too early
        truncated = truncated[:last_space]
    
    return truncated + suffix


def print_json_output(data: Dict[str, Any], title: Optional[str] = None):
    """Print formatted JSON output with optional title - uses stderr to avoid interfering with spinner"""
    if title:
        sys.stderr.write(f"\n📊 {title}\n")
        sys.stderr.write("=" * len(title) + "=\n")
    
    sys.stderr.write(JSONFormatter.to_json(data))
    sys.stderr.write("\n\n")
    sys.stderr.flush()


# Convenience functions for common operations
def track_download_progress(processed: int, total: int, operation: str = "Downloading"):
    """Track and display download progress - uses stderr to avoid interfering with spinner"""
    percentage = (processed / total) * 100 if total > 0 else 0
    status = f"{operation}: {processed}/{total} ({percentage:.1f}%)"
    sys.stderr.write(f"\r⠙ {status}")
    sys.stderr.flush()
    return percentage


def track_generation_progress(current: int, total: int, category: str):
    """Track and display dataset generation progress"""
    percentage = (current / total) * 100 if total > 0 else 0
    progress_data = {
        "current": current,
        "total": total,
        "percentage": percentage
    }
    
    # Create structured JSON output
    json_output = JSONFormatter.format_dataset_generation(
        category=category,
        progress=progress_data,
        stats={"status": "in_progress"}
    )
    
    print(f"\r⠹ Generating {category}: {current}/{total} ({percentage:.1f}%)", end="", flush=True)
    
    # Print JSON every 10 items for detailed tracking
    if current % 10 == 0 or current == total:
        print()  # New line
        print_json_output(json_output, f"Progress Update - {category}")
    
    return percentage


if __name__ == "__main__":
    # Test the utilities
    print("Testing Terminal UI utilities...")
    
    # Test spinner
    print("\n1. Testing spinner:")
    with spinner("Loading model..."):
        time.sleep(3)
    
    # Test timer
    print("\n2. Testing timer:")
    with timer() as t:
        time.sleep(2)
        print(f"Current elapsed: {t.format_duration()}")
    
    # Test JSON formatting
    print("\n3. Testing JSON formatting:")
    response_data = JSONFormatter.format_model_response(
        "Nitrogen deficiency in corn appears as yellowing of older leaves.",
        {"category": "crop_nutrition", "confidence": 0.95}
    )
    print_json_output(response_data, "Sample Model Response")
    
    # Test system message
    print("\n4. System message for dataset generation:")
    system_msg = create_json_system_message()
    print(system_msg[:200] + "...")
