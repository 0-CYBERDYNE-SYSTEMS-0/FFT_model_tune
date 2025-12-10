#!/bin/bash
# Quick project status and next steps

echo "🌾 PROJECT MANAGER STATUS REPORT"
echo "================================"
echo ""

# Download status
echo "📥 DOWNLOAD STATUS:"
if [ -f "download.log" ]; then
    echo "✅ Download process: ACTIVE"
    echo "📊 Recent activity:"
    tail -3 download.log | grep -E "(Downloading|Fetching)" | while read line; do
        echo "   $line"
    done
    
    # Check for incomplete files
    INCOMPLETE_COUNT=$(find models/trinity-nano-preview-complete/.cache/huggingface/download -name "*.incomplete" 2>/dev/null | wc -l)
    echo "🔄 Files downloading: $INCOMPLETE_COUNT"
    
    if [ $INCOMPLETE_COUNT -gt 0 ]; then
        echo "📦 Download size:"
        find models/trinity-nano-preview-complete/.cache/huggingface/download -name "*.incomplete" 2>/dev/null | while read file; do
            SIZE=$(du -h "$file" | cut -f1)
            echo "   • $SIZE"
        done
    fi
else
    echo "❌ No download log found"
fi

echo ""
echo "🎯 PROJECT READINESS:"

# Check project files
FILES=(
    "comprehensive_agricultural_dataset.json"
    "fine_tune_agricultural_complete.py"
    "test_and_analyze.py"
    "monitor_download.py"
    "project_manager.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file"
    fi
done

echo ""
echo "🚀 IMMEDIATE ACTIONS:"
echo "1. Monitor download progress:"
echo "   python3 project_manager.py"
echo ""
echo "2. Continuous monitoring:"
echo "   python3 project_manager.py --continuous"
echo ""
echo "3. Quick download check:"
echo "   tail -5 download.log"

echo ""
echo "⏱️ EXPECTED TIMELINE:"
echo "• Download completion: 30-60 minutes"
echo "• Model testing: 5-10 minutes" 
echo "• Fine-tuning setup: 15-30 minutes"
echo "• Agricultural specialization: 30-90 minutes"
