#!/usr/bin/env python3
"""
Qwen3.5 Agricultural Fine-tuning using Unsloth
Fine-tunes Qwen3.5 models (0.8B, 2B, 4B, 9B) on agricultural data using LoRA
"""

import json
import argparse
import os
from typing import Dict
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# Import terminal UI utilities
from utils.terminal_ui import (
    spinner, timer, JSONFormatter, print_json_output
)


# Qwen3.5 model configurations
QWEN35_MODELS = {
    "qwen3.5-0.8b": {
        "name": "Qwen/Qwen3.5-0.8B",
        "vram": "3GB",
    },
    "qwen3.5-2b": {
        "name": "Qwen/Qwen3.5-2B",
        "vram": "5GB",
    },
    "qwen3.5-4b": {
        "name": "Qwen/Qwen3.5-4B",
        "vram": "10GB",
    },
    "qwen3.5-9b": {
        "name": "Qwen/Qwen3.5-9B",
        "vram": "22GB",
    },
}


def load_agricultural_dataset(file_path: str) -> Dataset:
    """Load agricultural dataset from JSON file"""
    print(f"📂 Loading agricultural dataset from {file_path}...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to HuggingFace dataset format
    dataset = Dataset.from_list(data)
    print(f"✅ Loaded {len(dataset)} agricultural Q&A pairs")
    return dataset


def format_agricultural_prompt(sample: Dict) -> str:
    """Format agricultural data for instruction following"""
    instruction = sample["instruction"]
    input_text = sample["input"] if sample.get("input") else ""
    output = sample["output"]
    
    if input_text:
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
    else:
        prompt = f"Instruction: {instruction}\nOutput: {output}"
    
    return {"text": prompt}


def prepare_training_data(dataset: Dataset) -> Dataset:
    """Prepare training data for fine-tuning"""
    print("🔧 Preparing training data...")
    
    # Format data with text column for SFTTrainer
    formatted_dataset = dataset.map(
        format_agricultural_prompt,
        remove_columns=dataset.column_names
    )
    
    print(f"✅ Prepared {len(formatted_dataset)} training examples")
    return formatted_dataset


def load_model_and_tokenizer(model_key: str, max_seq_length: int = 2048):
    """Load Qwen3.5 model and tokenizer using Unsloth"""
    if model_key not in QWEN35_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Choose from: {list(QWEN35_MODELS.keys())}")
    
    model_info = QWEN35_MODELS[model_key]
    print(f"📦 Loading {model_key} (VRAM: {model_info['vram']})...")
    
    # Unsloth: Use bf16 LoRA (NOT 4-bit QLoRA for Qwen3.5)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_info["name"],
        max_seq_length=max_seq_length,
        load_in_4bit=False,      # QLoRA not recommended for Qwen3.5
        load_in_16bit=True,     # bf16/16-bit LoRA
        full_finetuning=False,
    )
    
    print(f"✅ Model loaded successfully")
    return model, tokenizer


def setup_lora(model, lora_rank: int = 16):
    """Setup LoRA configuration for fine-tuning"""
    print("🔧 Setting up LoRA fine-tuning...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    print(f"✅ LoRA setup complete (rank={lora_rank})")
    return model


def train_fine_tuning(
    model,
    train_dataset,
    tokenizer,
    output_dir: str = "outputs_qwen35",
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 10,
    save_steps: int = 100,
):
    """Train the model using LoRA fine-tuning with progress tracking"""
    print(f"🚀 Starting fine-tuning...")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Max sequence length: {max_seq_length}")
    print(f"   - Gradient accumulation: {gradient_accumulation_steps}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            max_seq_length=max_seq_length,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=epochs * len(train_dataset) // (batch_size * gradient_accumulation_steps),
            logging_steps=1,
            output_dir=output_dir,
            optim="adamw_8bit",
            seed=3407,
            dataset_num_proc=1,
            learning_rate=learning_rate,
            report_to="none",
        ),
    )
    
    with timer() as t:
        with spinner("Training model"):
            trainer.train()
    
    print(f"🎉 Fine-tuning completed in {t.format_duration()}!")
    return model, trainer


def save_fine_tuned_model(model, tokenizer, output_dir: str = "fine_tuned_qwen35"):
    """Save the fine-tuned model"""
    print(f"💾 Saving fine-tuned model to {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save LoRA adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Model saved to {output_dir}")
    return output_dir


def test_model(model, tokenizer, test_questions=None):
    """Test the fine-tuned model"""
    if test_questions is None:
        test_questions = [
            "How do I identify and treat powdery mildew on squash?",
            "What is the best fertilizer for organic tomato growth?",
            "How can I prevent damping off in seedlings?"
        ]
    
    print("\n🧪 Testing fine-tuned model...")
    
    # Set to inference mode
    FastLanguageModel.for_inference(model)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        
        # Format prompt
        prompt = f"Instruction: Answer the following agricultural question.\nInput: {question}\nOutput:"
        
        # Generate response
        from unsloth import generate
        response = generate(
            model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
        )
        
        print(f"   Response: {response}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5 on agricultural data")
    parser.add_argument(
        "--model",
        default="qwen3.5-4b",
        choices=list(QWEN35_MODELS.keys()),
        help="Qwen3.5 model size"
    )
    parser.add_argument(
        "--dataset_path",
        default="agricultural_dataset.json",
        help="Path to agricultural dataset"
    )
    parser.add_argument(
        "--output_dir",
        default="outputs_qwen35",
        help="Output directory for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only run inference test"
    )
    
    args = parser.parse_args()
    
    print("🌱 Qwen3.5 Agricultural Fine-tuning")
    print("=" * 50)
    print(f"Model: {args.model} (VRAM: {QWEN35_MODELS[args.model]['vram']})")
    print(f"Dataset: {args.dataset_path}")
    print()
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model, args.max_length)
        
        # Setup LoRA
        model = setup_lora(model, args.lora_rank)
        
        if not args.test_only:
            # Load and prepare dataset
            dataset = load_agricultural_dataset(args.dataset_path)
            train_dataset = prepare_training_data(dataset)
            
            # Train
            trained_model, trainer = train_fine_tuning(
                model=model,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_seq_length=args.max_length,
            )
            
            # Save
            save_dir = save_fine_tuned_model(
                trained_model,
                tokenizer,
                f"fine_tuned_{args.model.replace('.', '_')}"
            )
        
        # Test
        test_model(model, tokenizer)
        
        print(f"\n🎉 Qwen3.5 fine-tuning completed successfully!")
        print(f"🌾 Your model is now specialized for agricultural tasks!")
        
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
