"""
Configuration classes for Memory-Augmented Transformer.

All architectural choices are controlled via these config dataclasses,
which can be loaded from YAML files.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from pathlib import Path
import yaml
from omegaconf import OmegaConf


@dataclass
class MemoryConfig:
    """Configuration for memory bank and cross-attention."""
    
    # === Vanilla mode (disable memory for control experiments) ===
    vanilla_mode: bool = False
    
    # === Memory bank settings ===
    num_memory_tokens: int = 1024
    memory_dim: Optional[int] = None  # Defaults to model hidden_dim if None
    # Optional memory-cross-attention head overrides.
    # If None, falls back to model.num_heads / model.num_kv_heads.
    memory_num_heads: Optional[int] = None
    memory_num_kv_heads: Optional[int] = None
    
    # === Memory layer placement ===
    # Options: "all", "first_k", "last_k", "every_n", "custom", "none"
    memory_layer_placement: str = "all"
    memory_layer_k: int = 5  # Used for first_k, last_k
    memory_layer_n: int = 3  # Used for every_n
    memory_layer_indices: Optional[List[int]] = None  # Used for custom
    
    # === Memory bank sharing ===
    # Options: "shared", "per_layer", "every_k_layers"
    memory_sharing: str = "shared"
    memory_sharing_k: int = 2  # For every_k_layers
    
    # === Block integration variant ===
    # "A": Self-Attn -> Memory Cross-Attn -> MLP (default)
    # "B": Self-Attn -> MLP -> Memory Cross-Attn -> MLP
    memory_block_variant: str = "A"
    # Dropout used specifically in memory cross-attention. If None, falls back
    # to model.dropout for backward compatibility.
    memory_dropout: Optional[float] = None
    
    # === Low-rank options ===
    use_low_rank_memory: bool = False
    memory_rank: int = 64
    # "factorized": M = A @ B^T where A: (N_m, r), B: (d, r)
    # "reduced_dim": M stored as (N_m, r), attention in r-dim space
    low_rank_mode: str = "factorized"
    
    use_low_rank_projections: bool = False
    projection_rank: int = 64
    
    # === Chapter routing (MoE-style) ===
    use_chapters: bool = False
    num_chapters: int = 100
    # tokens_per_chapter is auto-calculated as num_memory_tokens // num_chapters
    top_k_chapters: int = 20
    # Always include first N chapters in memory selection (global/shared chapters).
    num_shared_chapters: int = 0
    # Scale routed chapter weights relative to shared-chapter weights.
    routed_scaling_factor: float = 1.0
    # Normalize shared/routed memory token vectors before mixing.
    # Helps keep one branch from dominating due to raw magnitude.
    normalize_shared_routed_before_mixing: bool = False
    # Vector normalization type for shared/routed branches.
    # Options: "rms", "layernorm"
    shared_routed_norm_type: str = "rms"
    shared_routed_norm_eps: float = 1e-6
    
    # Routing strategy:
    # - "sequence": mean-pool
    # - "sequence-rolling": average of rolling-window means across sequence
    # - "token": per-token chapter routing (dense shared + sparse routed path)
    routing_strategy_train: str = "sequence"
    routing_strategy_inference: str = "sequence"  # "sequence", "sequence-rolling", "rolling", "token", "hybrid"
    routing_window_size: int = 128  # For rolling/hybrid routing during generation (inference)
    # Token-level sparse routing kernel version used when routing_strategy is "token".
    # Options: "v1", "v2", "v3" (default: v2).
    token_routing_kernel_version: str = "v2"
    
    # === Router losses ===
    use_load_balance_loss: bool = True
    load_balance_coefficient: float = 0.01
    
    use_auxiliary_loss: bool = False
    auxiliary_loss_coefficient: float = 0.01
    
    use_z_loss: bool = False
    z_loss_coefficient: float = 0.001
    
    # === Memory attention gradient checkpointing ===
    # Enables gradient checkpointing for memory cross-attention when flash attention unavailable
    memory_gradient_checkpointing: bool = True
    
    # === Quantization ===
    quantize_memory: bool = False
    memory_quant_bits: int = 8  # 4 or 8
    
    # === Initialization ===
    wo_init_zero: bool = True  # Initialize output projection to zero for stable training (adapter and from-scratch)
    memory_init_std: float = 0.02  # Std for memory token initialization
    
    # === LoRA settings (for comparison/combination) ===
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # === Mode flags ===
    use_memory_adapter: bool = True  # Enable memory cross-attention
    use_both_memory_and_lora: bool = False  # Enable both for combined experiments


@dataclass
class ModelConfig:
    """Configuration for the base transformer model."""
    
    # === Architecture (for from-scratch training) ===
    hidden_dim: int = 768
    num_heads: int = 12
    # Grouped-query attention: number of KV heads. If None, defaults to num_heads.
    num_kv_heads: Optional[int] = None
    num_layers: int = 12
    intermediate_dim: int = 3072
    vocab_size: int = 32000
    max_seq_len: int = 8192
    # Optional alias (HF-style naming). If set, takes precedence over max_seq_len.
    max_position_embeddings: Optional[int] = None

    # === Tokenizer (for from-scratch and optional override in adapter mode) ===
    # If None:
    # - Adapter mode defaults to base_model_name
    # - From-scratch defaults to a Llama-style 32k tokenizer (see Trainer._load_tokenizer)
    tokenizer_name: Optional[str] = None
    # Optional explicit chat template override. If set, this template is forced
    # onto the tokenizer and used by apply_chat_template().
    chat_template: Optional[str] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    # === Positional encoding ===
    use_rope: bool = True
    rope_theta: float = 10000.0
    
    # === Regularization ===
    dropout: float = 0.0
    attention_dropout: float = 0.0
    hidden_activation: str = "swiglu"  # swiglu/silu/relu/gelu/sigmoid/tanh
    initializer_range: float = 0.02  # Std for from-scratch Linear/Embedding init
    # Optional targeted init overrides for from-scratch model components.
    # If None, they fall back to initializer_range.
    self_attn_wo_init_std: Optional[float] = None
    mlp_down_proj_init_std: Optional[float] = None
    
    # === Normalization ===
    norm_eps: float = 1e-6
    use_rms_norm: bool = True  # RMSNorm (like Llama) vs LayerNorm
    
    # === For adapter mode on pretrained models ===
    base_model_name: Optional[str] = None  # e.g., "Qwen/Qwen2.5-1.5B"
    freeze_base_model: bool = True
    
    # === Attention implementation ===
    use_flash_attention: bool = True  # Use Flash Attention if available
    tie_embeddings: bool = True  # Tie input embedding and LM head


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # === Learning rates (separate for different components) ===
    memory_lr: float = 1e-4
    # Optional override for memory-bank parameters only. If None, memory banks
    # use memory_lr.
    memory_bank_lr: Optional[float] = None
    lora_lr: float = 1e-4
    base_model_lr: float = 1e-5
    
    # === Training mode ===
    # "pretraining": Standard LM objective
    # "instruction_finetuning": Chat/instruction format
    training_mode: str = "instruction_finetuning"
    
    # === Dataset ===
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    dataset_subset: Optional[str] = None
    dataset_split: str = "train"
    eval_split: str = "test"
    
    # Text field(s) in dataset - can be string or list for chat format
    text_field: Union[str, List[str]] = "messages"
    
    # For preprocessing
    max_length: int = 8192
    
    # === Distributed training ===
    distributed_strategy: str = "ddp"  # "ddp", "fsdp"
    num_gpus: int = 1
    
    # FSDP specific
    fsdp_sharding_strategy: str = "FULL_SHARD"  # "FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"
    
    # === Training hyperparameters ===
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: Optional[int] = None  # If set, overrides max_steps
    max_steps: int = 10000
    warmup_steps: int = 100
    warmup_ratio: Optional[float] = None  # If set, overrides warmup_steps
    
    # === Optimizer ===
    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # === Scheduler ===
    scheduler: str = "cosine"  # "cosine", "linear", "constant", "wsd"
    min_lr_ratio: float = 0.1  # Minimum LR as ratio of peak LR
    # Optional delayed decay (for cosine/linear): hold peak LR until this point.
    # If both are set, decay_start_step takes precedence.
    decay_start_step: Optional[int] = None
    decay_start_ratio: Optional[float] = None
    # WSD = Warmup -> Stable -> Decay. Stable length can be absolute or ratio.
    wsd_stable_steps: Optional[int] = None
    wsd_stable_ratio: float = 0.0
    
    # === Mixed precision ===
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    save_precision: Optional[str] = None  # "fp32", "fp16", "bf16" (if None, same as mixed_precision context)
    
    # === Checkpointing ===
    gradient_checkpointing: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: Optional[int] = 3  # null => keep all checkpoints
    save_best_model: bool = True
    
    # === Early stopping ===
    early_stopping: bool = False
    early_stopping_patience: int = 5  # Number of eval steps without improvement
    early_stopping_threshold: float = 0.0  # Minimum improvement to reset patience
    
    # === Logging ===
    logging_steps: int = 10
    log_to_wandb: bool = False
    wandb_project: str = "memory-transformer"
    wandb_run_name: Optional[str] = None
    
    # === Output ===
    output_dir: str = "./outputs"
    # Resume full training state (model + optimizer + scheduler + trainer metadata).
    resume_from_checkpoint: Optional[str] = None
    # Initialize model weights from checkpoint but start a fresh run:
    # optimizer/scheduler/trainer state are NOT restored (global_step starts at 0).
    init_from_checkpoint: Optional[str] = None


@dataclass 
class Config:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Config object with all settings
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load YAML with OmegaConf for nice features (interpolation, etc.)
    omega_conf = OmegaConf.load(config_path)
    
    # Convert to dict and create structured config
    config_dict = OmegaConf.to_container(omega_conf, resolve=True)
    
    # Build config objects
    model_cfg = ModelConfig(**config_dict.get("model", {}))
    memory_cfg = MemoryConfig(**config_dict.get("memory", {}))
    training_cfg = TrainingConfig(**config_dict.get("training", {}))
    
    return Config(model=model_cfg, memory=memory_cfg, training=training_cfg)


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Config object to save
        config_path: Path to save YAML file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict
    config_dict = {
        "model": {k: v for k, v in config.model.__dict__.items()},
        "memory": {k: v for k, v in config.memory.__dict__.items()},
        "training": {k: v for k, v in config.training.__dict__.items()},
    }
    
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def get_memory_layer_indices(config: Config) -> List[int]:
    """
    Compute which layer indices should have memory cross-attention.
    
    Args:
        config: Full configuration
        
    Returns:
        List of layer indices (0-indexed) that should have memory layers
    """
    num_layers = config.model.num_layers
    mem_cfg = config.memory
    
    if mem_cfg.vanilla_mode or not mem_cfg.use_memory_adapter:
        return []
    
    placement = mem_cfg.memory_layer_placement
    
    if placement == "none":
        return []
    elif placement == "all":
        return list(range(num_layers))
    elif placement == "first_k":
        k = min(mem_cfg.memory_layer_k, num_layers)
        return list(range(k))
    elif placement == "last_k":
        k = min(mem_cfg.memory_layer_k, num_layers)
        return list(range(num_layers - k, num_layers))
    elif placement == "every_n":
        n = mem_cfg.memory_layer_n
        # Bug 15 fix: Validate n > 0 to prevent ValueError from range(0, num_layers, 0)
        if n <= 0:
            raise ValueError(f"memory_layer_n must be > 0 for 'every_n' placement, got {n}")
        return list(range(0, num_layers, n))
    elif placement == "custom":
        if mem_cfg.memory_layer_indices is None:
            raise ValueError("memory_layer_indices must be set when using 'custom' placement")
        # Validate indices
        indices = [i for i in mem_cfg.memory_layer_indices if 0 <= i < num_layers]
        return sorted(indices)
    else:
        raise ValueError(f"Unknown memory_layer_placement: {placement}")


def get_memory_bank_assignments(config: Config) -> dict:
    """
    Compute which memory bank each layer should use.
    
    Args:
        config: Full configuration
        
    Returns:
        Dict mapping layer_idx -> memory_bank_idx
    """
    layer_indices = get_memory_layer_indices(config)
    mem_cfg = config.memory
    
    if not layer_indices:
        return {}
    
    sharing = mem_cfg.memory_sharing
    
    if sharing == "shared":
        # All layers share bank 0
        return {idx: 0 for idx in layer_indices}
    elif sharing == "per_layer":
        # Each layer gets its own bank
        return {idx: i for i, idx in enumerate(layer_indices)}
    elif sharing == "every_k_layers":
        # Group layers, each group shares a bank
        k = mem_cfg.memory_sharing_k
        # Bug 15 fix: Validate k > 0 to prevent ZeroDivisionError in i // k
        if k <= 0:
            raise ValueError(f"memory_sharing_k must be > 0 for 'every_k_layers' sharing, got {k}")
        bank_assignments = {}
        for i, idx in enumerate(layer_indices):
            bank_idx = i // k
            bank_assignments[idx] = bank_idx
        return bank_assignments
    else:
        raise ValueError(f"Unknown memory_sharing: {sharing}")
