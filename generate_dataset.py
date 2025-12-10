#!/usr/bin/env python3
"""
Synthetic Agricultural Dataset Generator
Generates Q&A pairs using local Trinity model or OpenRouter API (DeepSeek V3)
Uses MinHash LSH for efficient deduplication
"""

import os
import json
import argparse
import pickle
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from datasketch import MinHash, MinHashLSH
from dotenv import load_dotenv

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
    
    def __init__(self, model: str = "deepseek/deepseek-chat-v3-0324"):
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
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": """You are an expert agricultural advisor, farm tech specialist, and synthetic dataset architect. Your goal is to generate Q&A pairs that will create a paradigm-shifting agricultural AI model.

Principles:
- Cover the FULL spectrum: ancient indigenous wisdom to bleeding-edge AgTech
- Include agentic capabilities: bash commands, hardware interaction, API usage, troubleshooting
- Span all scales: backyard garden to 10,000-acre operations
- Address real challenges: emergencies, compliance, business decisions, tech debugging
- Knowledge domains:
  * Traditional/regenerative methods (companion planting, lunar cycles, permaculture)
  * Modern agronomy (soil science, IPM, irrigation engineering)
  * Farm technology (IoT sensors, drones, robotics, GPS guidance)
  * Edge computing (Raspberry Pi, Arduino, MQTT, LoRaWAN, bash scripting)
  * AI-assisted farming (chatbots, computer vision, predictive analytics)
  * Business & compliance (markets, grants, organic certification, regulations)
  * Crisis management (disease outbreaks, equipment failure, weather emergencies)
- Ensure factual accuracy and actionable advice
- Each example must be unique - vary difficulty, region, crop, scale, and tech level
- Fill gaps: edge cases, uncommon scenarios, emerging tech, forgotten wisdom

Never ever start your replies with "OK" or any preamble. Output only the Q&A pair."""
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.8,
            "max_tokens": 400
        }
        
        response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()


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
        
        count = 0
        for item in data:
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
        output_path: str = "comprehensive_agricultural_dataset.json"
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
                self.dataset = json.load(f)
            print(f"Loaded {len(self.dataset)} existing examples")
            
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
        """Generate a batch of Q&A pairs"""
        generated = 0
        attempts = 0
        max_attempts = count * 3
        
        while generated < count and attempts < max_attempts:
            attempts += 1
            category = self._get_next_category()
            prompt = self._build_prompt(category)
            
            try:
                if verbose:
                    print(f"[{generated+1}/{count}] Generating {category}...", end=" ")
                
                response = self.provider.generate(prompt)
                qa_pair = self._parse_response(response)
                
                if qa_pair is None:
                    if verbose:
                        print("(parse failed)")
                    continue
                
                if self.deduplicator.is_duplicate(qa_pair["instruction"]):
                    if verbose:
                        print("(duplicate)")
                    continue
                
                self.deduplicator.add_question(qa_pair["instruction"])
                self.dataset.append(qa_pair)
                self.category_counts[category] += 1
                generated += 1
                
                if verbose:
                    print(f"OK - {qa_pair['instruction'][:50]}...")
                
                time.sleep(0.1)
                
            except Exception as e:
                if verbose:
                    print(f"(error: {e})")
                time.sleep(1)
        
        return generated
    
    def save_dataset(self):
        """Save dataset to JSON file"""
        with open(self.output_path, 'w') as f:
            json.dump(self.dataset, f, indent=2)
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
        default="deepseek/deepseek-chat-v3-0324",
        help="OpenRouter model to use (default: deepseek/deepseek-chat-v3-0324)"
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
        default="comprehensive_agricultural_dataset.json",
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
