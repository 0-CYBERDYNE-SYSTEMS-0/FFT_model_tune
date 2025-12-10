# Trinity 6B Agricultural Fine-tuning Setup Guide

## Overview
This guide walks you through setting up Trinity Nano Preview for agricultural fine-tuning on your Mac Mini M2 (32GB).

## Prerequisites ✅
- Mac Mini M2 with 32GB RAM (✅ Confirmed)
- Python 3.14+ installed (✅ Confirmed)
- MLX framework installed (✅ Confirmed)
- Virtual environment activated (✅ Confirmed)

## Installation Steps

### 1. Download Trinity Model
```bash
# Download the Trinity Nano Preview model
huggingface-cli download arcee-ai/Trinity-Nano-Preview --local-dir models/trinity-nano-preview-complete

# Alternative: Manual download if CLI fails
# Visit: https://huggingface.co/arcee-ai/Trinity-Nano-Preview
```

### 2. Prepare Agricultural Dataset
```bash
# The comprehensive dataset is ready with 14 examples covering:
# - Crop Management
# - Plant Diseases  
# - Soil Health
# - Pest Management
# - Water Management
# - Sustainable Practices
# - Harvest and Storage
```

### 3. Run Fine-tuning
```bash
# Activate virtual environment
source venv/bin/activate

# Run basic test first
python3 test_and_analyze.py

# Run full fine-tuning (once model downloads complete)
python3 fine_tune_agricultural_complete.py --epochs 3 --learning_rate 2e-4

# Test only (if model already available)
python3 fine_tune_agricultural_complete.py --test_only
```

## Expected Performance

### On Your Mac Mini M2 (32GB):
- **Model Size**: Trinity Nano (6B params, 1B active with MoE)
- **Memory Usage**: ~12-18GB during training
- **Training Time**: 30-90 minutes for 3 epochs
- **Inference Speed**: 20-50 tokens/second
- **Dataset Size**: 14 comprehensive agricultural Q&A pairs

### Memory Optimization:
- Model uses 4-bit quantization (reduces memory by ~75%)
- LoRA fine-tuning adds minimal overhead (~1-10MB)
- Unified memory architecture efficiently uses your 32GB RAM

## Agricultural Domains Covered

1. **Crop Management** (2 examples)
   - Soil preparation for corn
   - Optimal soybean planting timing

2. **Plant Diseases** (2 examples)
   - Powdery mildew identification and prevention
   - Bacterial wilt in tomatoes

3. **Soil Health** (2 examples)
   - Clay soil improvement
   - Nutrient deficiency identification

4. **Pest Management** (2 examples)
   - Natural aphid control
   - Cutworm identification and control

5. **Water Management** (2 examples)
   - Vegetable irrigation scheduling
   - Rainwater harvesting systems

6. **Sustainable Practices** (2 examples)
   - Cover crop benefits and selection
   - Efficient composting systems

7. **Harvest and Storage** (2 examples)
   - Winter squash harvest timing
   - Root vegetable storage techniques

## Troubleshooting

### Model Download Issues
```bash
# Check download status
ls -la models/trinity-nano-preview-complete/

# If incomplete, delete and retry
rm -rf models/trinity-nano-preview-complete
huggingface-cli download arcee-ai/Trinity-Nano-Preview --local-dir models/trinity-nano-preview-complete
```

### Memory Issues
- Your 32GB RAM should handle the model comfortably
- If issues occur, reduce batch size or sequence length
- Close other applications during training

### Training Performance
- Training may be slower on M2 compared to NVIDIA GPUs
- Normal for 6B parameter models
- Focus on testing and validation

## Next Steps

1. **Test Base Model**: Verify agricultural knowledge without fine-tuning
2. **Fine-tune**: Apply LoRA training on agricultural dataset  
3. **Validate**: Test model responses on agricultural questions
4. **Expand**: Add more agricultural data for better coverage
5. **Deploy**: Use fine-tuned model for agricultural applications

## File Structure
```
trin_train/
├── models/
│   └── trinity-nano-preview-complete/    # Trinity model files
├── comprehensive_agricultural_dataset.json  # Training data
├── fine_tune_agricultural_complete.py   # Main fine-tuning script
├── test_and_analyze.py                  # Model testing script
├── enhanced_agricultural_dataset.json   # Extended dataset
├── venv/                                # Python virtual environment
└── README_setup.md                      # This guide
```

## Support

For issues or questions:
1. Check MLX documentation: https://ml-explore.github.io/mlx/build/html/
2. Review Trinity model page: https://huggingface.co/arcee-ai/Trinity-Nano-Preview
3. Agricultural extension resources for domain knowledge

## Success Metrics

- Model loads without errors ✅
- Agricultural questions return relevant answers
- Training completes within reasonable time
- Fine-tuned model shows improved agricultural knowledge
- Model runs locally for inference

---
**Ready to start!** Run `python3 test_and_analyze.py` to verify everything works.
