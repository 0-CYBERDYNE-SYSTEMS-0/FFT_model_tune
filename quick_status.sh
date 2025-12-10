#!/bin/bash
# Quick Trinity Model Download Status Check

echo "🔍 Trinity Model Download Quick Check"
echo "====================================="

MODEL_DIR="models/trinity-nano-preview-complete"

if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Model directory not found!"
    exit 1
fi

echo "📁 Model Directory: $MODEL_DIR"
echo ""

# Check for incomplete downloads
INCOMPLETE_COUNT=$(find "$MODEL_DIR/.cache/huggingface/download" -name "*.incomplete" 2>/dev/null | wc -l)
echo "📥 Incomplete downloads: $INCOMPLETE_COUNT"

if [ $INCOMPLETE_COUNT -gt 0 ]; then
    echo "🔄 Currently downloading files:"
    find "$MODEL_DIR/.cache/huggingface/download" -name "*.incomplete" 2>/dev/null | while read file; do
        SIZE=$(du -h "$file" | cut -f1)
        echo "  • $(basename "$file") ($SIZE)"
    done
fi

# Check for completed safetensors files
SAFETENSORS_COUNT=$(find "$MODEL_DIR" -name "*.safetensors" -not -path "*/.cache/*" 2>/dev/null | wc -l)
echo "✅ Completed model files: $SAFETENSORS_COUNT"

if [ $SAFETENSORS_COUNT -gt 0 ]; then
    echo "🎉 Model files downloaded:"
    find "$MODEL_DIR" -name "*.safetensors" -not -path "*/.cache/*" 2>/dev/null | while read file; do
        SIZE=$(du -h "$file" | cut -f1)
        echo "  ✓ $(basename "$file") ($SIZE)"
    done
fi

# Total cache size
CACHE_SIZE=$(du -sh "$MODEL_DIR/.cache" 2>/dev/null | cut -f1)
echo "📦 Total downloaded: $CACHE_SIZE"

# Quick estimate
if [ $INCOMPLETE_COUNT -gt 0 ]; then
    echo ""
    echo "⏱️ Status: Download in progress"
    echo "💡 Check again in 30 minutes or run:"
    echo "   python3 monitor_download.py --continuous"
elif [ $SAFETENSORS_COUNT -ge 3 ]; then
    echo ""
    echo "🎉 Status: Download likely complete!"
    echo "🚀 Ready to test with:"
    echo "   python3 test_and_analyze.py"
else
    echo ""
    echo "⚠️ Status: Download incomplete or stalled"
    echo "🔄 Try restarting download:"
    echo "   huggingface-cli download arcee-ai/Trinity-Nano-Preview --local-dir $MODEL_DIR"
fi
