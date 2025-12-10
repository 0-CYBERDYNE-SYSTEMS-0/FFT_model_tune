#!/usr/bin/env python3
"""
Quick Trinity Agricultural Fine-tuning Script
Fastest path to get training started with existing resources
"""

import json
import time
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def load_agricultural_dataset():
    """Load the agricultural dataset we created"""
    print("📂 Loading agricultural dataset...")
    
    with open('comprehensive_agricultural_dataset.json', 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} agricultural examples")
    return Dataset.from_list(data)

def setup_model_and_tokenizer():
    """Load Trinity model and tokenizer"""
    print("📦 Loading Trinity model...")
    
    model_name = "models/trinity-nano-preview-complete"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"📝 Tokenizer vocab: {tokenizer.vocab_size:,}")
    
    return model, tokenizer

def format_training_data(examples):
    """Format agricultural data for training"""
    formatted_texts = []
    
    for example in examples:
        instruction = example["instruction"]
        output = example["output"]
        
        # Create instruction-following format
        text = f"""Instruction: {instruction}
Answer: {output}"""
        
        formatted_texts.append(text)
    
    return formatted_texts

def setup_lora_training(model):
    """Configure LoRA for efficient fine-tuning"""
    print("🔧 Setting up LoRA configuration...")
    
    peft_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,  # Scaling parameter
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Layers to fine-tune
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    
    print("✅ LoRA configured:")
    print(f"   - Rank: {peft_config.r}")
    print(f"   - Alpha: {peft_config.lora_alpha}")
    print(f"   - Dropout: {peft_config.lora_dropout}")
    
    return model

def tokenize_function(examples, tokenizer):
    """Tokenize the agricultural dataset for training"""
    texts = []
    
    for example in examples:
        instruction = example["instruction"]
        output = example["output"]
        
        # Create instruction-following format
        text = f"""Instruction: {instruction}
Answer: {output}"""
        texts.append(text)
    
    # Tokenize the texts
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Set labels same as input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def train_model(model, tokenizer, dataset):
    """Execute the fine-tuning training"""
    print("🚀 Starting Trinity agricultural fine-tuning...")
    
    # Training arguments optimized for M2 Mac Mini
    training_args = TrainingArguments(
        output_dir="./fine_tuned_trinity_agricultural",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Small batch size for memory
        gradient_accumulation_steps=4,  # Effective batch size = 4
        learning_rate=2e-4,
        logging_steps=5,
        save_steps=25,
        save_strategy="steps",
        report_to="none",  # Disable wandb logging
        dataloader_pin_memory=False,  # Save memory
        fp16=False,  # Use bfloat16 for Apple Silicon
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Single process for stability
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    print("⚡ Training configuration:")
    print(f"   - Epochs: {training_args.num_train_epochs}")
    print(f"   - Learning rate: {training_args.learning_rate}")
    print(f"   - Batch size: {training_args.per_device_train_batch_size}")
    print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    
    print("\n🔥 Starting training...")
    start_time = time.time()
    
    # Train the model
    trainer.train()
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed!")
    print(f"⏱️ Total training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    
    return trainer

def test_trained_model(trainer, tokenizer):
    """Test the fine-tuned agricultural model"""
    print("\n🧪 Testing fine-tuned agricultural model...")
    
    # Save the model
    trainer.save_model()
    print("💾 Model saved to: ./fine_tuned_trinity_agricultural")
    
    # Test agricultural questions
    test_questions = [
        "What are signs of nitrogen deficiency in plants?",
        "How do I control aphids naturally?",
        "When should I harvest tomatoes?",
        "What causes blossom end rot?"
    ]
    
    model = trainer.model
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Testing: {question}")
        print("   Response: ", end="")
        
        try:
            # Create prompt
            prompt = f"Instruction: {question}\nAnswer:"
            
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            
            print(response[:150] + "..." if len(response) > 150 else response)
            
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n🎉 Agricultural fine-tuning completed successfully!")
    print(f"🌾 Your Trinity model is now specialized for agricultural tasks!")

def main():
    """Main training pipeline"""
    print("🌱 Trinity 6B Agricultural Fine-tuning - Quick Start")
    print("=" * 55)
    
    try:
        # Load dataset
        dataset = load_agricultural_dataset()
        
        # Load model and tokenizer
        model, tokenizer = setup_model_and_tokenizer()
        
        # Setup LoRA for efficient training
        model = setup_lora_training(model)
        
        # Train the model
        trainer = train_model(model, tokenizer, dataset)
        
        # Test the results
        test_trained_model(trainer, tokenizer)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🚀 Next steps:")
        print(f"   1. Test in LM Studio by loading: ./fine_tuned_trinity_agricultural")
        print(f"   2. Use the model for agricultural Q&A")
        print(f"   3. Generate more training data and retrain for better results")
    else:
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Check that Trinity model is downloaded")
        print(f"   2. Verify all dependencies are installed")
        print(f"   3. Check memory usage (should be ~12-18GB)")
