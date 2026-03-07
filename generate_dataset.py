#!/usr/bin/env python3
"""
Synthetic Agricultural Dataset Generator
Generates Q&A pairs using local Trinity model or OpenRouter API (DeepSeek V3)
Uses MinHash LSH for efficient deduplication
"""

import os
import sys
import json
import argparse
import pickle
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from datetime import datetime

from datasketch import MinHash, MinHashLSH
from dotenv import load_dotenv

# Import terminal UI utilities
from utils.terminal_ui import (
    spinner, timer, JSONFormatter, create_json_system_message, 
    track_generation_progress, print_json_output, truncate_text
)

load_dotenv()

AGRICULTURAL_CATEGORIES = [
    # Core Farming (12)
    "soil_preparation",
    "pest_control",
    "irrigation",
    "harvesting",
    "nutrient_management",
    "crop_diseases",
    "planting_techniques",
    "weather_adaptation",
    "composting",
    "seed_selection",
    "weed_management",
    "equipment_maintenance",
    # Tech Spectrum (5)
    "traditional_methods",
    "precision_agriculture",
    "iot_automation",
    "drone_robotics",
    "ai_farm_tech",
    # Agentic/Edge Computing (3)
    "edge_devices",
    "bash_scripting",
    "protocols_connectivity",
    # Business & Compliance (3)
    "farm_business",
    "regulatory_compliance",
    "sustainability_carbon",
    # Crisis & Data (2)
    "emergency_response",
    "data_analytics",
]

CATEGORY_PROMPTS = {
    # Core Farming
    "soil_preparation": "soil preparation, amendment, pH adjustment, or tillage",
    "pest_control": "pest identification, natural pest control, or integrated pest management",
    "irrigation": "irrigation methods, water management, or drought strategies",
    "harvesting": "harvest timing, techniques, or post-harvest storage",
    "nutrient_management": "fertilization, nutrient deficiency diagnosis, or soil nutrition",
    "crop_diseases": "plant disease identification, prevention, or treatment",
    "planting_techniques": "planting methods, spacing, transplanting, or seedling care",
    "weather_adaptation": "frost protection, heat stress, or climate adaptation",
    "composting": "composting methods, organic matter management, or mulching",
    "seed_selection": "seed varieties, seed saving, or germination techniques",
    "weed_management": "weed control, cover crops for weed suppression, or mulching",
    "equipment_maintenance": "farm tool care, equipment calibration, or machinery maintenance",
    # Tech Spectrum
    "traditional_methods": "ancient farming techniques, indigenous practices, lunar planting cycles, heritage seed saving, permaculture design, or traditional crop rotation wisdom",
    "precision_agriculture": "GPS-guided equipment, variable rate technology, yield mapping, soil conductivity sensors, satellite imagery analysis, or prescription maps",
    "iot_automation": "smart irrigation controllers, soil moisture sensor networks, voice-coded farm commands, automated greenhouse systems, or farm connectivity solutions",
    "drone_robotics": "drone crop scouting, aerial spraying systems, robotic harvesters, autonomous tractors, computer vision for plant health, or UAV regulations for farms",
    "ai_farm_tech": "using AI chatbots for crop diagnosis, farm management software setup, writing automation scripts for farming, integrating weather or market APIs, or troubleshooting farm technology",
    # Agentic/Edge Computing
    "edge_devices": "Raspberry Pi farm monitoring setups, Arduino sensor projects, ESP32 configurations for agriculture, offline-first farm systems, deploying local AI models on farm hardware, or solar-powered edge nodes",
    "bash_scripting": "cron jobs for irrigation scheduling, shell scripts for sensor data logging, SSH management of farm servers, systemd services for farm sensors, log parsing for equipment diagnostics, or automated backup scripts for farm data",
    "protocols_connectivity": "MQTT broker setup for farm sensors, LoRaWAN configuration for field monitoring, Zigbee mesh networks on farms, ModBus communication with equipment, WiFi range extension for barns, or cellular failover for remote fields",
    # Business & Compliance
    "farm_business": "agricultural market timing strategies, crop pricing analysis, cost-benefit calculations for equipment, USDA grant applications, crop insurance options, or farm succession planning",
    "regulatory_compliance": "organic certification process and requirements, pesticide application record-keeping, water rights and usage regulations, food safety standards (GAP/FSMA), agricultural labor regulations, or export documentation requirements",
    "sustainability_carbon": "agricultural carbon credit programs, regenerative agriculture certification, biodiversity metrics for farms, cover crop incentive programs, lifecycle assessment for farm products, or ecosystem services documentation",
    # Crisis & Data
    "emergency_response": "disease outbreak containment protocols, frost emergency rapid response, equipment failure recovery procedures, agricultural supply chain disruptions, wildfire preparation for farms, or flood mitigation strategies",
    "data_analytics": "yield prediction models and interpretation, soil test result analysis, sensor data pattern recognition, farm record-keeping formats and best practices, spreadsheet formulas for agricultural calculations, or building farm data visualization dashboards",
}


class Provider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class LocalProvider(Provider):
    """Uses Trinity Nano model locally via transformers"""
    
    def __init__(self, model_path: str = "models/trinity-nano-preview-complete"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
    
    def is_available(self) -> bool:
        return Path(self.model_path).exists()
    
    def _load_model(self):
        if self.model is None:
            print("Loading local Trinity model...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            print("Local model loaded.")
    
    def generate(self, prompt: str) -> str:
        self._load_model()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()


class OpenRouterProvider(Provider):
    """Uses DeepSeek V3 via OpenRouter API"""
    
    def __init__(self, model: str = "deepseek/deepseek-v3.2"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model
    
    def is_available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 10
    
    def generate(self, prompt: str) -> str:
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/trin_train",
            "X-Title": "Agricultural Dataset Generator"
        }
        
        data = {
            "model": "deepseek/deepseek-v3.2",  # Exact model as requested
            "messages": [
                {
                    "role": "system",
                    "content": create_json_system_message()
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.8,
            "max_tokens": 400
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.Timeout:
            raise Exception(f"API timeout after 60 seconds for model deepseek/deepseek-v3.2")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except KeyError as e:
            raise Exception(f"Invalid API response format: {str(e)}")


class MinHashDeduplicator:
    """Efficient deduplication using MinHash LSH"""
    
    def __init__(self, threshold: float = 0.7, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.questions = []
        self.index_path = Path("dataset_index.pkl")
    
    def _create_minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=self.num_perm)
        words = text.lower().split()
        for i in range(len(words) - 2):
            shingle = " ".join(words[i:i+3])
            m.update(shingle.encode('utf-8'))
        if len(words) < 3:
            for word in words:
                m.update(word.encode('utf-8'))
        return m
    
    def is_duplicate(self, question: str) -> bool:
        minhash = self._create_minhash(question)
        result = self.lsh.query(minhash)
        return len(result) > 0
    
    def add_question(self, question: str):
        minhash = self._create_minhash(question)
        key = f"q_{len(self.questions)}"
        self.lsh.insert(key, minhash)
        self.questions.append(question)
    
    def load_existing(self, dataset_path: str):
        """Load existing questions from dataset into the index"""
        if not Path(dataset_path).exists():
            return 0
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Handle both old format (direct array) and new consolidated format
        if isinstance(data, dict) and 'data' in data:
            items = data['data']
        else:
            # Legacy format - direct array
            items = data
        
        count = 0
        for item in items:
            if isinstance(item, dict):
                question = item.get("instruction", "")
                if question and not self.is_duplicate(question):
                    self.add_question(question)
                    count += 1
        
        return count
    
    def save_index(self):
        with open(self.index_path, 'wb') as f:
            pickle.dump((self.lsh, self.questions), f)
    
    def load_index(self):
        if self.index_path.exists():
            with open(self.index_path, 'rb') as f:
                self.lsh, self.questions = pickle.load(f)
            return len(self.questions)
        return 0


class DatasetGenerator:
    """Main dataset generation orchestrator"""
    
    def __init__(
        self,
        provider: Provider,
        deduplicator: MinHashDeduplicator,
        output_path: str = "consolidated_agricultural_dataset.json"
    ):
        self.provider = provider
        self.deduplicator = deduplicator
        self.output_path = output_path
        self.dataset = []
        self.category_counts = {cat: 0 for cat in AGRICULTURAL_CATEGORIES}
    
    def load_existing_dataset(self):
        """Load existing dataset if it exists"""
        if Path(self.output_path).exists():
            with open(self.output_path, 'r') as f:
                data = json.load(f)
            
            # Handle both old format (direct array) and new consolidated format
            if isinstance(data, dict) and 'data' in data:
                self.dataset = data['data']
                print(f"Loaded {len(self.dataset)} existing examples from consolidated dataset")
            else:
                # Legacy format - direct array
                self.dataset = data
                print(f"Loaded {len(self.dataset)} existing examples from legacy dataset")
            
            loaded = self.deduplicator.load_existing(self.output_path)
            print(f"Indexed {loaded} questions for deduplication")
    
    def _get_next_category(self) -> str:
        """Get the category with fewest examples (balancing)"""
        return min(self.category_counts, key=self.category_counts.get)
    
    def _build_prompt(self, category: str) -> str:
        topic = CATEGORY_PROMPTS.get(category, category)
        return f"""Generate one unique agricultural question and detailed answer about {topic}.

Format your response exactly as:
Q: [A specific, practical farming question]
A: [A detailed 2-4 sentence answer with actionable advice and specific recommendations]

Generate only ONE question-answer pair. Be specific and practical."""
    
    def _parse_response(self, response: str) -> Optional[dict]:
        """Parse Q: ... A: ... format from response"""
        q_match = re.search(r'Q:\s*(.+?)(?=\nA:|\n\n|$)', response, re.DOTALL | re.IGNORECASE)
        a_match = re.search(r'A:\s*(.+?)(?=\n\nQ:|\n\n\n|$)', response, re.DOTALL | re.IGNORECASE)
        
        if not q_match or not a_match:
            a_match = re.search(r'A:\s*(.+)', response, re.DOTALL | re.IGNORECASE)
        
        if q_match and a_match:
            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()
            
            if len(question) > 20 and len(answer) > 50:
                return {
                    "instruction": question,
                    "input": "",
                    "output": answer
                }
        return None
    
    def generate_batch(self, count: int, verbose: bool = True):
        """Generate a batch of Q&A pairs with progress tracking"""
        generated = 0
        attempts = 0
        max_attempts = count * 3
        
        with timer() as t:
            with spinner(f"Generating {count} agricultural Q&A pairs"):
                while generated < count and attempts < max_attempts:
                    attempts += 1
                    category = self._get_next_category()
                    prompt = self._build_prompt(category)
                    
                    try:
                        if verbose:
                            # Use stderr for verbose output to avoid interfering with spinner
                            sys.stderr.write(f"\r[{generated+1}/{count}] Generating {category}...")
                            sys.stderr.flush()
                        
                        # Retry logic for API failures
                        max_retries = 3
                        for retry in range(max_retries):
                            try:
                                response = self.provider.generate(prompt)
                                break
                            except Exception as api_error:
                                if retry < max_retries - 1:
                                    if verbose:
                                        sys.stderr.write(f"\n(retry {retry+1}/{max_retries}...)")
                                        sys.stderr.flush()
                                    time.sleep(2)  # Wait before retry
                                    continue
                                else:
                                    raise api_error
                        
                        qa_pair = self._parse_response(response)
                        
                        if qa_pair is None:
                            if verbose:
                                sys.stderr.write("\n(parse failed)\n")
                                sys.stderr.flush()
                            continue
                        
                        if self.deduplicator.is_duplicate(qa_pair["instruction"]):
                            if verbose:
                                sys.stderr.write("\n(duplicate)\n")
                                sys.stderr.flush()
                            continue
                        
                        self.deduplicator.add_question(qa_pair["instruction"])
                        self.dataset.append(qa_pair)
                        self.category_counts[category] += 1
                        generated += 1
                        
                        # Track progress with JSON output
                        track_generation_progress(generated, count, category)
                        
                        # Print detailed JSON status every 10 items
                        if generated % 10 == 0:
                            stats = {
                                "total_samples": len(self.dataset),
                                "categories_completed": sum(1 for v in self.category_counts.values() if v > 0),
                                "total_categories": len(AGRICULTURAL_CATEGORIES),
                                "current_category": category,
                                "generation_rate": f"{generated/t.elapsed:.2f} items/min" if t.elapsed > 0 else "calculating..."
                            }
                            
                            progress_data = {
                                "current": generated,
                                "total": count,
                                "percentage": (generated / count) * 100
                            }
                            
                            json_output = JSONFormatter.format_dataset_generation(
                                category=category,
                                progress=progress_data,
                                stats=stats
                            )
                            
                            print_json_output(json_output, f"Generation Progress Update")
                        
                        if verbose:
                            # Use stderr for all verbose output to avoid interfering with spinner
                            sys.stderr.write("\nOK\n")
                            # Truncate Q&A for better readability
                            truncated_q = truncate_text(qa_pair['instruction'], max_length=80)
                            truncated_a = truncate_text(qa_pair['output'], max_length=120)
                            sys.stderr.write(f"    Q: {truncated_q}\n")
                            sys.stderr.write(f"    A: {truncated_a}\n\n")
                            sys.stderr.flush()
                        
                        time.sleep(0.1)
                        
                    except Exception as e:
                        if verbose:
                            sys.stderr.write(f"\n(error: {e})\n")
                            sys.stderr.flush()
                        
                        # Format error as JSON for tracking
                        error_data = JSONFormatter.format_error(e, f"generation_{category}")
                        if generated % 5 == 0:  # Print errors every 5 attempts
                            print_json_output(error_data, "Generation Error")
                        
                        time.sleep(1)
        
        # Final progress update
        final_stats = {
            "total_samples": len(self.dataset),
            "categories_completed": sum(1 for v in self.category_counts.values() if v > 0),
            "total_categories": len(AGRICULTURAL_CATEGORIES),
            "generation_rate": f"{generated/t.duration:.2f} items/min" if t.duration > 0 else "calculating...",
            "completion_time": t.format_duration()
        }
        
        final_json = JSONFormatter.format_dataset_generation(
            category="completed",
            progress={"current": generated, "total": count, "percentage": 100.0},
            stats=final_stats
        )
        
        print_json_output(final_json, "Dataset Generation Complete")
        
        return generated
    
    def save_dataset(self):
        """Save dataset to JSON file in consolidated format"""
        # Create consolidated format with metadata
        output_data = {
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "total_entries": len(self.dataset),
                "format": "instruction_input_output",
                "generator_version": "2.0",
                "description": "Agricultural Q&A dataset with enhanced formatting"
            },
            "data": self.dataset
        }
        
        with open(self.output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved {len(self.dataset)} examples to {self.output_path}")
        
        self.deduplicator.save_index()


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic agricultural Q&A dataset"
    )
    parser.add_argument(
        "--provider", 
        choices=["local", "openrouter"],
        default="openrouter",
        help="Provider to use for generation (default: openrouter)"
    )
    parser.add_argument(
        "--model",
        default="deepseek/deepseek-v3.2",
        help="OpenRouter model to use (default: deepseek/deepseek-v3.2)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of Q&A pairs to generate (default: 10)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="MinHash similarity threshold for deduplication (default: 0.7)"
    )
    parser.add_argument(
        "--output",
        default="consolidated_agricultural_dataset.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Agricultural Dataset Generator")
    print("=" * 60)
    
    if args.provider == "openrouter":
        provider = OpenRouterProvider(model=args.model)
        if not provider.is_available():
            print("ERROR: OPENROUTER_API_KEY not set in .env file")
            print("Copy .env.example to .env and add your API key")
            return 1
        print(f"Provider: OpenRouter ({args.model})")
    else:
        provider = LocalProvider()
        if not provider.is_available():
            print("ERROR: Local model not found at models/trinity-nano-preview-complete")
            return 1
        print("Provider: Local Trinity Nano")
    
    deduplicator = MinHashDeduplicator(threshold=args.threshold)
    generator = DatasetGenerator(
        provider=provider,
        deduplicator=deduplicator,
        output_path=args.output
    )
    
    generator.load_existing_dataset()
    
    print(f"Generating {args.count} new Q&A pairs...")
    print("-" * 60)
    
    generated = generator.generate_batch(args.count, verbose=not args.quiet)
    
    print("-" * 60)
    print(f"Successfully generated {generated} new examples")
    
    generator.save_dataset()
    
    print("\nCategory distribution:")
    for cat, count in sorted(generator.category_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {cat}: {count}")
    
    return 0


if __name__ == "__main__":
    exit(main())
