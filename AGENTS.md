# AGENTS.md - Agentic Coding Guidelines

## Build/Run Commands
```bash
# Environment setup (use uv - fastest)
uv sync

# Generate synthetic agricultural dataset
uv run python generate_dataset.py --provider openrouter --count 100

# Fine-tune Trinity model with LoRA
uv run python quick_train.py

# Test and analyze model performance
uv run python test_and_analyze.py

# Run single test (example)
uv run python test_and_analyze.py --question "What are sustainable farming practices?"
```

## Code Style Guidelines

### Imports Order
1. Standard library (os, json, typing)
2. Third-party (torch, transformers, datasets)
3. Local modules (utils.terminal_ui)

### Naming Conventions
- Functions: snake_case (`load_agricultural_dataset`)
- Constants: UPPER_CASE (`MAX_SEQUENCE_LENGTH`)
- Files: descriptive names with underscores

### Error Handling
- Use try/catch with specific exceptions
- Provide user-friendly error messages with emojis: ❌ Error: explanation
- Include fallback options (local model if API fails)

### Code Patterns
- Type hints for all function parameters and returns
- Docstrings for major functions
- Progress indicators: 📂 Loading, ✅ Success, ⏱️ Timing
- JSON output for programmatic consumption
- Memory-conscious defaults (batch_size=1 for Apple Silicon)

### Key Dependencies
- PyTorch ≥2.0, Transformers ≥4.40, PEFT ≥0.10
- Apple MLX optional for M-series optimization
- Environment variables in .env file
- LoRA config: rank=16, alpha=32, lr=2e-4