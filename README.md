# Trinity Nano Agricultural Fine-tuning

Fine-tune Arcee Trinity Nano (6B MoE) on agricultural Q&A data using your Mac Mini M2 Pro (32GB).

## Quick Start

```bash
# 1. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup and install (creates .venv automatically)
uv sync

# 3. Configure API key
cp .env.example .env
# Edit .env and add your OpenRouter API key

# 4. Generate training data
uv run python generate_dataset.py --provider openrouter --count 100

# 5. Fine-tune the model
uv run python quick_train.py
```

That's it! `uv sync` handles everything.

---

## Environment Setup

### Option 1: uv (Recommended - Fast & Simple)

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (creates .venv automatically)
uv sync

# Run any script
uv run python generate_dataset.py --count 100
uv run python quick_train.py

# Or activate the environment manually
source .venv/bin/activate
python generate_dataset.py --count 100
```

### Option 2: pip + venv (Traditional)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option 3: poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install
poetry shell
```

---

## API Key Setup

For synthetic dataset generation using DeepSeek V3 via OpenRouter:

1. Get an API key from [OpenRouter](https://openrouter.ai/settings/keys)
2. Copy the example env file:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your key:
   ```
   OPENROUTER_API_KEY=sk-or-v1-your-key-here
   ```

**Cost estimate**: ~$0.0003 per Q&A pair (~$0.30 for 1,000 examples)

---

## Synthetic Dataset Generation

Generate agricultural Q&A training data using AI models.

### Commands

```bash
# Generate 100 Q&A pairs using OpenRouter (DeepSeek V3)
python generate_dataset.py --provider openrouter --count 100

# Generate using local Trinity model (free, slower)
python generate_dataset.py --provider local --count 50

# Use a different OpenRouter model
python generate_dataset.py --provider openrouter --model deepseek/deepseek-chat-v3.1 --count 100

# Custom output file
python generate_dataset.py --count 100 --output my_dataset.json

# Adjust deduplication sensitivity (lower = stricter)
python generate_dataset.py --count 100 --threshold 0.5

# Quiet mode (less output)
python generate_dataset.py --count 100 --quiet
```

### All Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `openrouter` | `openrouter` (API) or `local` (Trinity) |
| `--model` | `deepseek/deepseek-v3.2` | OpenRouter model ID |
| `--count` | `10` | Number of Q&A pairs to generate |
| `--threshold` | `0.7` | MinHash similarity threshold (0.0-1.0) |
| `--output` | `consolidated_agricultural_dataset.json` | Output file path |
| `--quiet` | `false` | Suppress verbose output |

### Categories Generated

The generator balances across 12 agricultural categories:
- soil_preparation
- pest_control
- irrigation
- harvesting
- nutrient_management
- crop_diseases
- planting_techniques
- weather_adaptation
- composting
- seed_selection
- weed_management
- equipment_maintenance

---

## Fine-tuning

### Quick Training (LoRA)

```bash
# Run fine-tuning with default settings
python quick_train.py
```

This uses:
- LoRA (rank=16, alpha=32)
- Batch size 1 with gradient accumulation of 4
- 3 epochs
- Learning rate 2e-4

### Training Output

Fine-tuned model saved to: `./fine_tuned_trinity_agricultural/`

---

## Model Download

If the Trinity model isn't downloaded yet:

```bash
# Download Trinity Nano Preview
huggingface-cli download arcee-ai/Trinity-Nano-Preview \
  --local-dir models/trinity-nano-preview-complete
```

---

## File Structure

```
trin_train/
├── generate_dataset.py              # Synthetic data generator
├── quick_train.py                   # LoRA fine-tuning script
├── fine_tune_agricultural_complete.py  # Full training script (MLX)
├── test_and_analyze.py              # Model testing
├── comprehensive_agricultural_dataset.json  # Training data
├── .env                             # API keys (create from .env.example)
├── .env.example                     # API key template
├── requirements.txt                 # Python dependencies
├── models/
│   └── trinity-nano-preview-complete/  # Downloaded model
├── fine_tuned_trinity_agricultural/    # Output from training
└── venv/                            # Virtual environment
```

---

## Requirements

### Hardware
- Mac Mini M2 Pro (32GB RAM) or equivalent
- ~15GB disk space for model

### Software
- Python 3.10+
- macOS 14+ (Sonoma) recommended

### Dependencies

Core packages (installed via requirements.txt):
```
torch>=2.0
transformers>=4.40
peft>=0.10
datasets>=2.18
accelerate>=0.28
datasketch>=1.6
python-dotenv>=1.0
requests>=2.31
```

---

## Troubleshooting

### "OPENROUTER_API_KEY not set"
```bash
cp .env.example .env
# Edit .env and add your API key from https://openrouter.ai/settings/keys
```

### "Local model not found"
```bash
huggingface-cli download arcee-ai/Trinity-Nano-Preview \
  --local-dir models/trinity-nano-preview-complete
```

### Out of memory during training
- Reduce batch size in `quick_train.py`
- Close other applications
- Use `--count` with smaller batches for data generation

### Generation produces duplicates
- Lower the threshold: `--threshold 0.5`
- The MinHash index persists in `dataset_index.pkl`

---

## Provider Comparison

| Provider | Flag | Cost | Speed | Quality |
|----------|------|------|-------|---------|
| OpenRouter DeepSeek V3 | `--provider openrouter` | ~$0.0003/QA | ~1s/QA | Excellent |
| Local Trinity Nano | `--provider local` | Free | ~5-10s/QA | Good |

**Recommendation**: Use OpenRouter for initial high-quality dataset generation, then optionally expand with local model after fine-tuning.

---

## License

- Trinity Nano: Apache 2.0
- This project: MIT
