#!/usr/bin/env python3
"""
FFT Model Trainer — Fine-tune small LLMs for agricultural AI
Supports: Qwen3.5 (0.8B-9B), Gemma 4 (E2B/E4B) via Unsloth
Single command: uv run python train.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# Add project root to path for configs import
sys.path.insert(0, str(Path(__file__).parent))
from configs.models import (
    ALL_MODELS, ModelConfig, recommend_model,
    format_conversation, get_chat_template
)


# ─── GPU Detection ────────────────────────────────────────────────────

def detect_gpu() -> Tuple[str, float]:
    """Detect GPU and available VRAM."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        return name, round(vram, 1)
    elif torch.backends.mps.is_available():
        # Apple Silicon — use unified memory, estimate ~70% available
        import psutil
        total_ram = psutil.virtual_memory().total / (1024**3)
        return "Apple Silicon (MPS)", round(total_ram * 0.7, 1)
    else:
        import psutil
        total_ram = psutil.virtual_memory().total / (1024**3)
        return "CPU", round(total_ram * 0.5, 1)


# ─── Dataset Loading ──────────────────────────────────────────────────

def load_dataset(file_path: str, model_config: ModelConfig) -> Dataset:
    """Load agricultural dataset and format with chat template."""
    print(f"📂 Loading dataset from {file_path}...")

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Handle consolidated format (metadata + data)
    if isinstance(data, dict) and 'data' in data:
        items = data['data']
    else:
        items = data

    # Convert to conversation format using chat template
    conversations = []
    for item in items:
        instruction = item.get("instruction", item.get("question", ""))
        input_text = item.get("input", "")
        output = item.get("output", item.get("answer", ""))
        system = item.get("system", "")

        # Combine instruction + input for the user message
        user_msg = instruction
        if input_text:
            user_msg = f"{instruction}\n\n{input_text}"

        text = format_conversation(model_config, user_msg, output, system)
        conversations.append({"text": text})

    dataset = Dataset.from_list(conversations)
    print(f"✅ Loaded {len(dataset)} training examples")
    print(f"   Format: {model_config.chat_template_format} chat template")
    return dataset


# ─── Model Loading ────────────────────────────────────────────────────

def load_model(config: ModelConfig, max_seq_length: int = 2048):
    """Load model with Unsloth FastModel."""
    from unsloth import FastModel

    print(f"📦 Loading {config.display_name}...")
    print(f"   Model ID: {config.model_id}")
    print(f"   VRAM: {config.vram_4bit if config.use_4bit else config.vram_bf16}")
    print(f"   Mode: {'4-bit QLoRA' if config.use_4bit else 'bf16 LoRA'}")

    # Determine load settings
    load_kwargs = dict(
        model_name=config.unsloth_model_name or config.model_id,
        max_seq_length=max_seq_length,
    )

    if config.use_4bit:
        load_kwargs.update(load_in_4bit=True, load_in_16bit=False)
    else:
        load_kwargs.update(load_in_4bit=False, load_in_16bit=True)

    # Qwen3.5 requires transformers v5; Unsloth handles this
    model, tokenizer = FastModel.from_pretrained(**load_kwargs)

    # Patch tokenizer for models that need it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ Model loaded — {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params total")
    return model, tokenizer


def setup_lora(model, config: ModelConfig):
    """Apply LoRA adapters with Unsloth optimization."""
    from unsloth import FastModel as FM

    print(f"🔧 Setting up LoRA (rank={config.lora_rank})...")

    lora_kwargs = dict(
        r=config.lora_rank,
        target_modules=config.target_modules,
        lora_alpha=config.lora_rank,
        lora_dropout=0,              # Unsloth optimized
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Qwen3.5 needs explicit False for QLoRA
    if not config.use_4bit:
        lora_kwargs["use_gradient_checkpointing"] = "unsloth"

    model = FM.get_peft_model(model, **lora_kwargs)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ LoRA ready — {trainable:,} trainable / {total:,} total "
          f"({100 * trainable / total:.1f}%)")
    return model


# ─── Training ──────────────────────────────────────────────────────────

def train(
    model,
    tokenizer,
    dataset: Dataset,
    config: ModelConfig,
    output_dir: str = "output",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    max_seq_length: Optional[int] = None,
    save_steps: int = 100,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 10,
):
    """Train with SFTTrainer."""
    epochs = epochs or config.recommended_epochs
    batch_size = batch_size or config.recommended_batch_size
    learning_rate = learning_rate or config.recommended_lr
    max_seq_length = max_seq_length or config.max_seq_length

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🚀 Starting training...")
    print(f"   Model: {config.display_name}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Max sequence length: {max_seq_length}")
    print(f"   Output: {output_dir}")
    print()

    # Calculate max steps
    effective_batch = batch_size * gradient_accumulation_steps
    max_steps = (epochs * len(dataset)) // effective_batch

    training_args = SFTConfig(
        max_seq_length=max_seq_length,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        logging_steps=5,
        save_steps=save_steps,
        output_dir=output_dir,
        optim="adamw_8bit",
        seed=3407,
        learning_rate=learning_rate,
        report_to="none",
        dataset_num_proc=1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    print(f"\n✅ Training complete — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return model, trainer


# ─── Saving & Export ──────────────────────────────────────────────────

def save_model(model, tokenizer, output_dir: str):
    """Save LoRA adapter."""
    print(f"\n💾 Saving LoRA adapter to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Saved to {output_dir}")


def export_gguf(model, tokenizer, output_dir: str, quantization: str = "q4_k_m"):
    """Export to GGUF format for Ollama / llama.cpp."""
    print(f"\n📦 Exporting GGUF ({quantization})...")
    model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quantization)
    print(f"✅ GGUF exported to {output_dir}/")


# ─── Inference Test ───────────────────────────────────────────────────

def test_inference(model, tokenizer, config: ModelConfig, questions: Optional[List[str]] = None):
    """Quick inference test after training."""
    if questions is None:
        questions = [
            "How do I identify and treat powdery mildew on squash?",
            "What's the best fertilizer for organic tomatoes?",
            "How can I prevent damping off in seedlings?",
            "When should I rotate my corn crop?",
        ]

    from unsloth import FastModel as FM
    FM.for_inference(model)

    print(f"\n🧪 Inference test ({config.display_name})...")

    for i, question in enumerate(questions, 1):
        prompt = format_conversation(config, question, output="")
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
            response = response.replace("<|im_end|>", "").strip()
        elif "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1].strip()
            response = response.replace("<end_of_turn>", "").strip()

        print(f"\n  Q{i}: {question}")
        print(f"  A: {response[:200]}..." if len(response) > 200 else f"  A: {response}")


# ─── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FFT Model Trainer — Fine-tune small LLMs for agriculture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect GPU and pick best model
  uv run python train.py

  # Train a specific model
  uv run python train.py --model qwen3.5-4b

  # Train Gemma 4 E2B with 4-bit QLoRA (only 8GB VRAM!)
  uv run python train.py --model gemma4-e2b

  # Train and export GGUF for Ollama
  uv run python train.py --model qwen3.5-2b --export-gguf

  # Quick test with custom dataset
  uv run python train.py --model qwen3.5-4b --dataset my_data.json --epochs 5
        """
    )

    # Model selection
    parser.add_argument("--model", choices=list(ALL_MODELS.keys()),
                        help="Model to fine-tune (default: auto-select based on GPU)")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available models and exit")

    # Dataset
    parser.add_argument("--dataset", default="consolidated_agricultural_dataset.json",
                        help="Dataset JSON file (default: consolidated_agricultural_dataset.json)")

    # Training params
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size per device")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--max-length", type=int, help="Max sequence length")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps")

    # Output
    parser.add_argument("--output", default=None,
                        help="Output directory (default: output/<model_key>)")
    parser.add_argument("--export-gguf", action="store_true",
                        help="Export to GGUF after training")
    parser.add_argument("--quantization", default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="GGUF quantization level")

    # Inference
    parser.add_argument("--test-only", action="store_true",
                        help="Skip training, only run inference test")
    parser.add_argument("--test-question", action="append", dest="test_questions",
                        help="Custom test question (can repeat)")

    # Troubleshooting
    parser.add_argument("--force-4bit", action="store_true",
                        help="Force 4-bit loading even if bf16 is recommended")
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="Disable gradient checkpointing (uses more VRAM)")

    args = parser.parse_args()

    # List models
    if args.list_models:
        print("\n📋 Available Models for Fine-tuning\n")
        print(f"{'Key':<18} {'Display Name':<20} {'VRAM(bf16)':<12} {'VRAM(4bit)':<12} {'GPU'}")
        print("-" * 85)
        for key, cfg in ALL_MODELS.items():
            print(f"{key:<18} {cfg.display_name:<20} {cfg.vram_bf16:<12} {cfg.vram_4bit:<12} {cfg.recommended_gpu}")
        print("\n💡 Tip: Use --model <key> to select, or omit for auto-selection\n")
        return

    # Detect GPU
    gpu_name, vram_gb = detect_gpu()
    print(f"\n🖥️  GPU: {gpu_name} ({vram_gb:.1f}GB VRAM available)\n")

    # Auto-select model
    if args.model is None:
        compatible = recommend_model(vram_gb)
        if not compatible:
            print("❌ No compatible models found for your GPU. Try a smaller model.\n")
            print("Available models:")
            for key, cfg in sorted(ALL_MODELS.items(),
                                   key=lambda x: float(x[1].vram_bf16.replace("GB", ""))):
                print(f"  {key}: {cfg.display_name} ({cfg.vram_bf16} bf16 / {cfg.vram_4bit} 4bit)")
            return 1
        args.model, model_config = compatible[0]
        print(f"🤖 Auto-selected: {model_config.display_name} "
              f"(VRAM: {model_config.vram_4bit if model_config.use_4bit else model_config.vram_bf16})")
        if len(compatible) > 1:
            print(f"   Also fits: {', '.join(k for k, _ in compatible[1:4])}")
    else:
        model_config = ALL_MODELS[args.model]

    # Override 4-bit setting
    if args.force_4bit:
        model_config.use_4bit = True

    print(f"   Provider: {model_config.provider}")
    print(f"   Chat format: {model_config.chat_template_format}")
    print(f"   VRAM needed: {model_config.vram_4bit if model_config.use_4bit else model_config.vram_bf16}")
    print()

    # Verify sufficient VRAM
    required_vram = float(
        (model_config.vram_4bit if model_config.use_4bit else model_config.vram_bf16).replace("GB", "")
    )
    if required_vram > vram_gb * 1.1:  # 10% buffer
        print(f"⚠️  WARNING: Model needs {required_vram}GB but only {vram_gb:.1f}GB available")
        print(f"   Try: --force-4bit or --model with a smaller model")
        if not args.force_4bit and not model_config.use_4bit:
            print(f"   This model works in 4-bit with {model_config.vram_4bit}")
        return 1

    # Output directory
    output_dir = args.output or f"output/{args.model}"
    model_dir = f"models/{args.model}-agricultural"

    try:
        # Load model
        model, tokenizer = load_model(
            model_config,
            max_seq_length=args.max_length or model_config.max_seq_length
        )

        # Setup LoRA
        model = setup_lora(model, model_config)

        if not args.test_only:
            # Load dataset
            dataset = load_dataset(args.dataset, model_config)

            # Train
            model, trainer = train(
                model, tokenizer, dataset, model_config,
                output_dir=output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                max_seq_length=args.max_length,
                gradient_accumulation_steps=args.gradient_accumulation,
            )

            # Save adapter
            save_model(model, tokenizer, model_dir)

            # Export GGUF
            if args.export_gguf:
                gguf_dir = f"{model_dir}-gguf"
                export_gguf(model, tokenizer, gguf_dir, args.quantization)

        # Test inference
        test_questions = args.test_questions if args.test_questions else None
        test_inference(model, tokenizer, model_config, test_questions)

        print(f"\n🎉 Done! Model: {model_dir}")
        if args.export_gguf:
            print(f"   GGUF: {model_dir}-gguf/")
            print(f"\n📋 To use with Ollama:")
            print(f"   1. Create Modelfile:")
            print(f"      FROM {model_dir}-gguf/model-{args.quantization}.gguf")
            print(f"   2. ollama create farm-ai -f Modelfile")
            print(f"   3. ollama run farm-ai")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
