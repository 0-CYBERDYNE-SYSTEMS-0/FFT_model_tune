#!/usr/bin/env python3
"""
Trinity Model Comprehensive Test
Tests base model functionality before fine-tuning
"""

import time
from mlx_lm import load, generate

def test_trinity_model():
    """Test Trinity model loading and basic functionality"""
    print("🌾 Trinity 6B Agricultural Model Testing")
    print("=" * 45)
    
    # Load model
    print("📦 Loading Trinity model...")
    start_time = time.time()
    
    try:
        model, tokenizer = load("models/trinity-nano-preview-complete")
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model type: {type(model).__name__}")
        print(f"📝 Vocabulary size: {tokenizer.vocab_size:,} tokens")
        print(f"⏱️ Load time: {load_time:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Test agricultural questions
    print(f"\n🧪 Testing Agricultural Knowledge:")
    print("-" * 35)
    
    test_questions = [
        "What are the signs of nitrogen deficiency in plants?",
        "How often should I water my tomato plants?", 
        "What causes blossom end rot in tomatoes?",
        "How can I control aphids naturally?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Testing: {question}")
        print("   Response: ", end="", flush=True)
        
        try:
            start_gen = time.time()
            response = generate(
                model,
                tokenizer=tokenizer,
                prompt=f"Question: {question}\nAnswer:",
                max_tokens=80
            )
            gen_time = time.time() - start_gen
            
            # Clean up response
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            
            print(response[:150] + "..." if len(response) > 150 else response)
            print(f"   ⏱️ Generated in {gen_time:.1f}s")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n🎉 Trinity model testing completed!")
    print(f"🚀 Ready for agricultural fine-tuning!")
    
    return True

if __name__ == "__main__":
    test_trinity_model()
