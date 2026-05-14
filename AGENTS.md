# AGENTS.md — FFT Model Tune

## Build/Run Commands
```bash
# Environment setup
uv sync

# List available models
uv run python train.py --list-models

# Auto-train (detects GPU, picks best model)
uv run python train.py

# Train specific model
uv run python train.py --model qwen3.5-4b

# Train Gemma 4 with 4-bit QLoRA
uv run python train.py --model gemma4-e2b

# Train + export GGUF for Ollama
uv run python train.py --model qwen3.5-4b --export-gguf

# Generate dataset
uv run python generate_dataset.py --provider openrouter --count 100

# Generate conversations format
uv run python generate_dataset.py --format conversations --count 100

# Quick inference test (no training)
uv run python train.py --model qwen3.5-4b --test-only
```

## Architecture

### Core Files
- `train.py` — Unified trainer with auto GPU detection, model selection, GGUF export
- `generate_dataset.py` — Synthetic data generator with MinHash dedup
- `configs/models.py` — Model registry with VRAM configs, chat templates, recommendations

### Model Support
- **Qwen3.5**: 0.8B, 2B, 4B, 9B — bf16 LoRA only (QLoRA not recommended)
- **Gemma 4**: E2B (8GB 4-bit), E4B (10GB 4-bit) — 4-bit QLoRA works great

### Key Design Decisions
1. **Single entry point** — `train.py` handles everything
2. **Auto GPU detection** — picks best model for available hardware
3. **Universal dataset format** — instruction/input/output internally, chat templates applied at train time
4. **GGUF export built-in** — one flag to get Ollama-ready model
5. **Model registry is source of truth** — all VRAM numbers, defaults, templates in `configs/models.py`

### Unsloth Patterns
- Use `FastModel` (not `FastLanguageModel`) — Qwen3.5 and Gemma 4
- Qwen3.5: `load_in_16bit=True`, `load_in_4bit=False`
- Gemma 4: `load_in_4bit=True` (QLoRA works)
- Gradient checkpointing: `"unsloth"` for OOM prevention
- LoRA target modules: q,k,v,o,gate,up,down projections
- MoE router layers excluded from LoRA (stability)

### Chat Templates
- Qwen3.5: `<|im_start|>role\ncontent<|im_end|>`
- Gemma 4: `<start_of_turn>role\ncontent<end_of_turn>`
- Formatting handled by `configs/models.py` — training script calls `format_conversation()`

### Dataset Format
- **Standard** (default): `{instruction, input, output}` — universal, portable
- **Conversations**: `{messages: [{role, content}]}` — ShareGPT format
- Training script accepts both formats automatically
