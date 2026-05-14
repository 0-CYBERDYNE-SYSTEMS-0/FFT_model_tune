"""
FFT Model Config Registry
Central model definitions with VRAM requirements, chat templates, and Unsloth settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ModelConfig:
    """Configuration for a fine-tunable model."""
    # Identity
    model_id: str               # HuggingFace model ID (e.g. "Qwen/Qwen3.5-4B")
    display_name: str           # Human-friendly name
    provider: str               # "qwen" | "gemma" | "trinity"

    # Hardware
    vram_bf16: str              # VRAM for bf16 LoRA (e.g. "10GB")
    vram_4bit: str              # VRAM for 4-bit QLoRA (e.g. "5GB")
    recommended_gpu: str        # Minimum recommended GPU
    edge_compatible: bool = False  # Runs on phones / Raspberry Pi?

    # Training defaults
    recommended_epochs: int = 3
    recommended_batch_size: int = 2
    recommended_lr: float = 2e-4
    max_seq_length: int = 2048
    use_4bit: bool = False       # Default to bf16 unless QLoRA recommended
    lora_rank: int = 16

    # Chat template info
    chat_template_format: str = "chatml"  # chatml | gemma | llama3
    system_prompt_default: str = "You are an agricultural AI assistant. Provide accurate, practical farming advice based on agricultural science and best practices."
    thinking_supported: bool = False

    # Unsloth settings
    unsloth_model_name: Optional[str] = None  # If different from model_id
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    vision_supported: bool = False
    rl_supported: bool = False

    # Tags for filtering
    tags: List[str] = field(default_factory=list)


# ─── Chat Template Builders ───────────────────────────────────────────

def build_qwen35_chat(instruction: str, output: str = "", system: str = "") -> str:
    """Qwen3.5 ChatML format: <|im_start|>role\ncontent<|im_end|>"""
    if not system:
        system = "You are an agricultural AI assistant. Provide accurate, practical farming advice based on agricultural science and best practices."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]
    if output:
        messages.append({"role": "assistant", "content": output})

    # Qwen3.5 ChatML format
    text = ""
    for msg in messages:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    if output:
        text += "<|im_start|>assistant\n"  # Generation prompt

    return text


def build_gemma4_chat(instruction: str, output: str = "", system: str = "") -> str:
    """Gemma 4 format: <start_of_turn>role\ncontent<end_of_turn>"""
    if not system:
        system = "You are an agricultural AI assistant. Provide accurate, practical farming advice based on agricultural science and best practices."

    text = ""
    if system:
        text += f"<start_of_turn>user\n{system}\n\n{instruction}<end_of_turn>\n"
    else:
        text += f"<start_of_turn>user\n{instruction}<end_of_turn>\n"

    if output:
        text += f"<start_of_turn>model\n{output}<end_of_turn>\n"
    else:
        text += "<start_of_turn>model\n"

    return text


# ─── Model Registry ───────────────────────────────────────────────────

QWEN35_MODELS: Dict[str, ModelConfig] = {
    "qwen3.5-0.8b": ModelConfig(
        model_id="Qwen/Qwen3.5-0.8B",
        display_name="Qwen3.5 0.8B",
        provider="qwen",
        vram_bf16="3GB", vram_4bit="2GB",
        recommended_gpu="Any GPU / Phone",
        edge_compatible=True,
        recommended_epochs=3, recommended_batch_size=4, recommended_lr=2e-4,
        max_seq_length=2048, use_4bit=False,
        chat_template_format="chatml",
        thinking_supported=False,
        tags=["edge", "tiny", "phone", "raspberry-pi"],
    ),
    "qwen3.5-2b": ModelConfig(
        model_id="Qwen/Qwen3.5-2B",
        display_name="Qwen3.5 2B",
        provider="qwen",
        vram_bf16="5GB", vram_4bit="3GB",
        recommended_gpu="Laptop GPU / M1",
        edge_compatible=True,
        recommended_epochs=3, recommended_batch_size=2, recommended_lr=2e-4,
        max_seq_length=2048, use_4bit=False,
        chat_template_format="chatml",
        thinking_supported=False,
        tags=["edge", "small", "laptop", "raspberry-pi-5"],
    ),
    "qwen3.5-4b": ModelConfig(
        model_id="Qwen/Qwen3.5-4B",
        display_name="Qwen3.5 4B",
        provider="qwen",
        vram_bf16="10GB", vram_4bit="6GB",
        recommended_gpu="RTX 3060 / M2 Pro",
        edge_compatible=False,
        recommended_epochs=3, recommended_batch_size=2, recommended_lr=2e-4,
        max_seq_length=2048, use_4bit=False,
        chat_template_format="chatml",
        thinking_supported=False,
        tags=["mid", "recommended", "balanced"],
    ),
    "qwen3.5-9b": ModelConfig(
        model_id="Qwen/Qwen3.5-9B",
        display_name="Qwen3.5 9B",
        provider="qwen",
        vram_bf16="22GB", vram_4bit="12GB",
        recommended_gpu="RTX 4090 / A100",
        edge_compatible=False,
        recommended_epochs=3, recommended_batch_size=1, recommended_lr=2e-4,
        max_seq_length=2048, use_4bit=False,
        chat_template_format="chatml",
        thinking_supported=False,
        tags=["large", "powerful", "production"],
    ),
}

GEMMA4_MODELS: Dict[str, ModelConfig] = {
    "gemma4-e2b": ModelConfig(
        model_id="google/gemma-4-E2B-it",
        display_name="Gemma 4 E2B",
        provider="gemma",
        vram_bf16="12GB", vram_4bit="8GB",
        recommended_gpu="RTX 3060 / Laptop",
        edge_compatible=False,
        recommended_epochs=3, recommended_batch_size=2, recommended_lr=2e-4,
        max_seq_length=4096, use_4bit=True,   # QLoRA works great with Gemma 4
        chat_template_format="gemma",
        thinking_supported=True,
        vision_supported=True,
        tags=["small", "vision", "multimodal", "recommended"],
    ),
    "gemma4-e4b": ModelConfig(
        model_id="google/gemma-4-E4B-it",
        display_name="Gemma 4 E4B",
        provider="gemma",
        vram_bf16="16GB", vram_4bit="10GB",
        recommended_gpu="RTX 3070 / M2 Pro",
        edge_compatible=False,
        recommended_epochs=3, recommended_batch_size=2, recommended_lr=2e-4,
        max_seq_length=4096, use_4bit=True,
        chat_template_format="gemma",
        thinking_supported=True,
        vision_supported=True,
        tags=["mid", "vision", "multimodal", "recommended"],
    ),
}

# Combined registry for easy iteration
ALL_MODELS: Dict[str, ModelConfig] = {**QWEN35_MODELS, **GEMMA4_MODELS}

# Model categories for recommendations
MODEL_RECOMMENDATIONS = {
    "edge": ["qwen3.5-0.8b", "qwen3.5-2b"],
    "laptop": ["qwen3.5-2b", "qwen3.5-4b", "gemma4-e2b"],
    "desktop": ["qwen3.5-4b", "gemma4-e2b", "gemma4-e4b", "qwen3.5-9b"],
    "server": ["qwen3.5-4b", "qwen3.5-9b", "gemma4-e4b"],
}


def get_chat_template(config: ModelConfig) -> callable:
    """Return the appropriate chat template builder for a model."""
    if config.provider == "qwen":
        return build_qwen35_chat
    elif config.provider == "gemma":
        return build_gemma4_chat
    else:
        return build_qwen35_chat  # Default fallback


def format_conversation(config: ModelConfig, instruction: str, output: str = "",
                         system: str = "") -> str:
    """Format a single instruction/output pair using the model's chat template."""
    builder = get_chat_template(config)
    return builder(instruction, output, system)


def recommend_model(vram_gb: float) -> List[Tuple[str, ModelConfig]]:
    """Recommend models that fit within available VRAM."""
    fits = []
    for key, cfg in ALL_MODELS.items():
        vram_str = cfg.vram_4bit if cfg.use_4bit else cfg.vram_bf16
        required = float(vram_str.replace("GB", ""))
        if required <= vram_gb:
            fits.append((key, cfg))
    # Sort by model size (bigger = better, if it fits)
    return sorted(fits, key=lambda x: float(x[1].vram_bf16.replace("GB", "")), reverse=True)
