# FFT Model Tune — Fine-tune Small LLMs for Agriculture

Fine-tune open-source language models on agricultural data using **Unsloth** — 1.5-2x faster training, 50-80% less VRAM.

**Supported models:** Qwen3.5 (0.8B → 9B), Gemma 4 (E2B, E4B)

---

## Quickstart (3 commands)

```bash
# 1. Install dependencies
uv sync

# 2. Generate training data (or use existing)
uv run python generate_dataset.py --provider openrouter --count 100

# 3. Train (auto-detects GPU, picks best model)
uv run python train.py
```

That's it. Your fine-tuned agricultural AI model is in `models/<model>-agricultural/`.

---

## Which Model Should I Use?

| Your Hardware | VRAM | Best Model | Why |
|:---|:---|:---|:---|
| Raspberry Pi 5 / Phone | 2-4GB | Qwen3.5 0.8B | Edge-optimized, 3GB bf16 |
| Laptop GPU / M1 Mac | 4-8GB | Qwen3.5 2B or Gemma 4 E2B (4-bit) | Small but capable |
| RTX 3060 / M2 Pro | 8-12GB | Qwen3.5 4B or Gemma 4 E4B | Sweet spot |
| RTX 4090 / A100 | 16-24GB | Qwen3.5 9B | Most powerful small model |

### Model Comparison

| Model | VRAM (bf16) | VRAM (4-bit) | Speed | Quality | Best For |
|:---|:---|:---|:---|:---|:---|
| **Qwen3.5 0.8B** | 3GB | 2GB | ⚡⚡⚡ | ★★☆ | Phones, edge devices |
| **Qwen3.5 2B** | 5GB | 3GB | ⚡⚡⚡ | ★★★ | Laptops, Raspberry Pi 5 |
| **Qwen3.5 4B** | 10GB | 6GB | ⚡⚡ | ★★★★ | Recommended for most |
| **Qwen3.5 9B** | 22GB | 12GB | ⚡ | ★★★★★ | Production/cloud |
| **Gemma 4 E2B** | 12GB | 8GB | ⚡⚡ | ★★★★ | Vision + text, laptops |
| **Gemma 4 E4B** | 16GB | 10GB | ⚡ | ★★★★★ | Vision + text, desktop |

> **Qwen3.5 models:** Use bf16 LoRA (QLoRA not recommended for Qwen3.5)  
> **Gemma 4 models:** 4-bit QLoRA works great — fits on smaller GPUs

---

## Commands

### List available models
```bash
uv run python train.py --list-models
```

### Train a specific model
```bash
# Qwen3.5 4B (recommended for most GPUs)
uv run python train.py --model qwen3.5-4b

# Qwen3.5 2B (good for laptops)
uv run python train.py --model qwen3.5-2b

# Gemma 4 E2B with 4-bit QLoRA (only 8GB VRAM!)
uv run python train.py --model gemma4-e2b
```

### Train with custom settings
```bash
# More epochs, bigger batches
uv run python train.py --model qwen3.5-4b --epochs 5 --batch-size 4

# Custom dataset
uv run python train.py --model qwen3.5-2b --dataset my_farm_data.json

# Export GGUF for Ollama
uv run python train.py --model qwen3.5-4b --export-gguf
```

### Generate dataset with conversations format
```bash
# Standard format (instruction/input/output) — most portable
uv run python generate_dataset.py --count 100

# Conversations format (ShareGPT) — ready for chat models
uv run python generate_dataset.py --count 100 --format conversations

# Use a specific data source
uv run python generate_dataset.py --provider openrouter --count 500
```

### Quick inference test (no training)
```bash
uv run python train.py --model qwen3.5-4b --test-only
```

---

## Hardware Setup

### Step 1: Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Install dependencies
```bash
uv sync
```

### Step 3: Set up API key (for dataset generation)
```bash
cp .env.example .env
# Edit .env — add your OpenRouter key: OPENROUTER_API_KEY=sk-...
```

### Step 4: Verify GPU
```bash
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

---

## Using with Ollama

After training with `--export-gguf`:

```bash
# 1. Find the GGUF file
ls models/qwen3.5-4b-agricultural-gguf/

# 2. Create a Modelfile
cat > Modelfile << 'EOF'
FROM ./models/qwen3.5-4b-agricultural-gguf/model-q4_k_m.gguf
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
SYSTEM """You are a farm AI assistant. Provide practical, science-based agricultural advice."""
EOF

# 3. Create and run
ollama create farm-ai -f Modelfile
ollama run farm-ai
```

---

## File Structure

```
FFT_model_tune/
├── train.py                    # Main training script (auto GPU + model detection)
├── generate_dataset.py         # Synthetic data generator
├── configs/
│   └── models.py               # Model registry + chat templates + VRAM configs
├── utils/
│   └── terminal_ui.py          # Terminal UI utilities
├── consolidated_agricultural_dataset.json  # Training data (407+ examples)
├── pyproject.toml              # Dependencies (uv sync)
├── .env.example                # API key template
└── output/                     # Training outputs (gitignored)
    └── models/                 # Fine-tuned models
```

---

## Dataset Format

### Standard format (default, universal)
```json
{
  "instruction": "How do I treat powdery mildew on squash?",
  "input": "",
  "output": "Powdery mildew treatment: 1) Remove infected leaves..."
}
```
The training script auto-converts this to the correct chat template for each model.

### Conversations format (ShareGPT)
```json
{
  "messages": [
    {"role": "system", "content": "You are an agricultural AI assistant..."},
    {"role": "user", "content": "How do I treat powdery mildew on squash?"},
    {"role": "assistant", "content": "Powdery mildew treatment: 1) Remove infected leaves..."}
  ]
}
```

---

## Troubleshooting

### Out of memory
```bash
# Force 4-bit loading
uv run python train.py --model qwen3.5-9b --force-4bit

# Use a smaller model
uv run python train.py --model qwen3.5-2b

# Reduce batch size
uv run python train.py --model qwen3.5-4b --batch-size 1
```

### Slow training on Apple Silicon
- Training on MPS is slower than CUDA. Use a smaller model.
- For M2/M3/M4: Qwen3.5 2B or 4B are good choices.
- For M1: Stick with Qwen3.5 0.8B or 2B.

### "transformers version too old"
```bash
uv sync --upgrade
# Unsloth bundles the right transformers version — uv sync handles it
```

---

## License

- This project: MIT
- Qwen3.5 models: Apache 2.0
- Gemma 4 models: Google Gemma License
- Unsloth Core: Apache 2.0
