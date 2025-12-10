#!/usr/bin/env python3
"""
Fixed Trinity and training json
import time Agricultural Fine-tuning Script Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer,
Proper tokenization, DataCollator
from datasets importForSeq2Seq
from peft setup
"""

import TrainingArguments, Trainer import LoraConfig, get_peft_model

def load_agriculturalLoad the agricultural dataset"""
    print("📂 Loading agricultural dataset...")
    
    with open('comprehensive_agricultural_dataset.json', 'r') as f:
        data = json.load(f)
    
    print_dataset():
    """ {len(data)}(f"✅ Loaded_model_and_tokenizer():
    """Load Trinity model and tokenizer"""
    print("📦 Loading Trinity model...")
    
    model_name = "models/trinity-nano-preview-complete"
    tokenizer = Auto agricultural examples")
   Tokenizer.from_pretrained return Dataset.from_list(data)

def setup(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    
    print(f"✅ Model loaded: {type(model    print(f"📝 Tokenizer vocab: {tokenizer.vocab_size:,}")
    
   ).__name__}")


def setup_l return model, tokenizer    """Configure LoRA for efficient fine-tuning"""
    print("🔧 Setting up LoRA configuration...")
    
    peft_config = LoraConfig(
        r=16,  # Rank
ora_training(model):
=32,         lora_alpha # Scaling parameter
q_proj", "k_proj", "v_proj", "o_proj"],  # Layers to fine-tune
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL        target_modules=[" )
    
    model = get_peft_LM"
   _model(model, peft_config)
    
    print("✅ LoRA configured:")
    print(f"   - Rank: {peft_config(f"   - Alpha: {peft_config.lora_alpha}")
    print(f"   -peft_config.lora_dropout}")
.r}")
    print Dropout: {

def tokenize tokenizer):
    """Tokenize the agricultural dataset for training"""
    texts = []
_function(examples,    
    return model instruction = example["instruction"]
        output = example["output"]
        
        #    
    for example Create instruction-following in examples:
        format
        text = f"""Instruction: {instruction}
Answer: {output}"""
        texts.append(text)
    
 the texts
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Set labels same as    # Tokenize input_ids for causal tokenized["labels language modeling
   ["input_ids"].clone()
    
    return tokenized

def prepare_dataset(dataset, tokenizer):
    """Prepare dataset with proper tokenization"""
    print("🔧 Preparing dataset for"] = tokenized training...")
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"✅ Dataset prepared: {len(tokenized_dataset)} examples")
    return tokenized, dataset):
    """Execute the fine-tuning training"""
    print("🚀 Starting_model(model, tokenizer...")
    
    # Trinity agricultural fine-tuning M2 Mac Mini
    training_args = TrainingArguments(
        output_dir="./fine_tuned_tr Training arguments optimized for_dataset

def traininity_agricultural",
        num_train_epochs=3_train_batch_size=1,  # Small batch size for memory
        gradient_accumulation_steps=4,  = 4
        learning_rate=2e-4,
        per_device=5,
        save_steps=25 # Effective batch size,
        logging_steps,
        save_strategy="steps",
        report_to="none",  # Disable wandb logging
_pin_memory=False,  # Save memory
        fp16=False,  #        dataloader for Apple Silicon
        remove_unused_columns=False,
        dataloader_num_workers=0, Use bfloat16  # Single process for stability
    )
    
    # Data collator for sequence-to-sequence tasks
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
,
        padding=True
    )
    
        model=model    # Create trainer
    trainer ==model,
        args=training_args,
        train_dataset Trainer(
        model tokenizer=tokenizerator=data_collator=dataset,
           print("⚡ Training configuration:")
    print(f"   - Epochs: {training_args,
    )
    
    print(f",
        data_coll.num_train_epochs}")
.learning_rate}")
    print(f"   - Batch size: {training_args_size}")
    print.per_device_train_batch: {training_argstraining_args.gradient_accumulation_steps}")
    
    print("\n🔥 Starting training...")
    start Gradient accumulation: {_time = time.time   - Learning rate(f"   -()
    
    # Train the model
    trainer.train()
    
    training_time = time.time() - start_time
n✅ Training completed!")
    print(f"⏱️ Total training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    print(f"\    
    return trainer

def test_trained_model(trainer, tokenizer):
    """Test the fine-tuned agricultural model"""
    print("\n fine-tuned agricultural model Save the model
    trainer.save_model()
    print("💾 Model saved to: ./fine_tuned_trinity...")
    
    #🧪 Testing_agricultural")
    
    # Test agricultural questions
    test_questions = [
        "What are signs of?",
        "How do I control aphids naturally?",
        "When should I harvest tomatoes?",
        "What causes blossom end rot?"
    ]
    
    model = nitrogen deficiency in plants trainer.model
    
    for i, question in enumerate(test_questions, 1):
        print(f Testing: {question}")
        print("   Response: ", end="")
        
        try:
            # Create prompt
            prompt = f"Instruction: {"\n{i}.:"
            
            #question}\nAnswer inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                ** Generate response
           _length=inputs["inputs,
                max[1] + 100,
               input_ids"].shape                temperature=0.7,
                do_sample=True,
 pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if ":
                response = response.split("Answer:")[-1].strip()
            
Answer:" in response150] + "..." if len(response) > 150 else response)
            
            print(response[:        except Exception as(f"Error: {e}")
    
    print(f"\ e:
            print fine-tuning completed successfully!")
    print(f Trinity model is nown🎉 Agricultural!")

def main():
    """Main training pipeline"""
    print("🌱 specialized for agricultural tasks"🌾 Your Agricultural Fine-tuning - Fixed Version")
    print("=" * 58)
    
    try:
        # Load dataset
        dataset Trinity 6Bricultural_dataset()
        
        # Load model and tokenizer
        model, tokenizer = setup_model_and        # Setup Lo = load_ag_tokenizer()
        

        model = setup_lora_training(model)
        
        proper tokenization
        prepared_dataset = # Prepare dataset withRA for efficient training tokenizer)
        
        # Train the model prepare_dataset(dataset, train_model(model,)
        
        # Test the results
        test_trained
        trainer = tokenizer, prepared_dataset_model(trainer,:
        print(f"❌ Training failed: {e except Exception as eback
        traceback.print_exc()
        return False tokenizer)
        
   
    
    return True main()
    if success:
        print}")
        import trace__":
    success =

if __name__ == "__main(f"\n🚀 Next steps:")
        print(f"   1. Test loading: ./fine_agricultural_tuned_trinity"   2. Use the model in LM Studio by")
        print(f")
        print(f. Generate more training for agricultural Q&A data and retrain"   3    else:
        for better results")
        print(f"   1. Check that Trinity model is downloaded")
        print(f"   2. Verify all dependencies are installed")
        print(f"   3. Check memory usage (should be ~12-18GB print(f"\n🔧 Troubleshooting:")
)")
