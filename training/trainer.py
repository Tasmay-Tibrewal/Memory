"""
Training loop with Accelerate for distributed training.

Supports:
- Eval during training
- Resume from checkpoint
- Early stopping
- Best model saving
- Epoch-based or step-based training
- Learning rate finder
"""

import os
import json
import math
import shutil
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

from memory_transformer.config import Config, load_config
from memory_transformer.model import MemoryTransformer
from memory_transformer.adapter import MemoryAdapter
from memory_transformer.utils import (
    print_model_info, 
    save_checkpoint, 
    load_checkpoint,
    count_parameters,
    format_params,
)
from .data import create_dataloader


class Trainer:
    """
    Trainer for Memory-Augmented Transformer.
    
    Supports:
    - From-scratch training (MemoryTransformer)
    - Adapter training (MemoryAdapter on pretrained models)
    - Multi-GPU with DDP/FSDP via Accelerate
    - Mixed precision (fp16/bf16)
    - Gradient checkpointing
    - Separate learning rates for memory/LoRA/base
    - Eval during training
    - Early stopping
    - Best model saving
    - Resume from checkpoint
    """
    
    def __init__(
        self,
        config: Config,
        model: Optional[nn.Module] = None,
    ):
        self.config = config
        self.train_config = config.training
        
        # Initialize accelerator (optionally with FSDP plugin)
        fsdp_plugin = None
        if config.training.distributed_strategy == "fsdp":
            try:
                from accelerate.utils import FullyShardedDataParallelPlugin
                from torch.distributed.fsdp import ShardingStrategy
            except Exception as e:
                raise ImportError(
                    "FSDP requested (training.distributed_strategy='fsdp') but required "
                    "dependencies are unavailable. Install/upgrade accelerate and use a "
                    "PyTorch build with FSDP support."
                ) from e

            sharding_map = {
                "FULL_SHARD": ShardingStrategy.FULL_SHARD,
                "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
                "NO_SHARD": ShardingStrategy.NO_SHARD,
            }
            sharding_name = config.training.fsdp_sharding_strategy
            if sharding_name not in sharding_map:
                raise ValueError(
                    f"Unknown fsdp_sharding_strategy: {sharding_name}. "
                    f"Options: {sorted(sharding_map.keys())}"
                )

            fsdp_plugin = FullyShardedDataParallelPlugin(
                sharding_strategy=sharding_map[sharding_name],
            )

        accelerator_kwargs = {
            "mixed_precision": config.training.mixed_precision,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "log_with": "wandb" if config.training.log_to_wandb else None,
        }
        if fsdp_plugin is not None:
            accelerator_kwargs["fsdp_plugin"] = fsdp_plugin

        try:
            self.accelerator = Accelerator(**accelerator_kwargs)
        except TypeError as e:
            raise TypeError(
                "Failed to construct Accelerator with current accelerate version. "
                "If you requested FSDP, upgrade accelerate (>=0.25.0) or set "
                "training.distributed_strategy='ddp'."
            ) from e

        # Note: Keep Accelerate's default even_batches=True. This keeps evaluation loops safe for FSDP
        # (all ranks run the same number of forward passes). We avoid metric bias from duplicated samples
        # by using accelerator.gather_for_metrics() in evaluate().

        # Validate (advisory) num_gpus against actual launched processes
        if config.training.num_gpus and self.accelerator.num_processes != config.training.num_gpus:
            import warnings
            warnings.warn(
                f"Config expects training.num_gpus={config.training.num_gpus} but "
                f"Accelerate launched {self.accelerator.num_processes} processes. "
                "Set --num_processes in `accelerate launch` to match, or update config.",
                UserWarning,
            )
        
        # Set seed
        set_seed(42)

        # Load tokenizer early (from-scratch vocab_size depends on tokenizer)
        self.tokenizer = self._load_tokenizer()

        # Create or use provided model
        if model is not None:
            self.model = model
        else:
            self.model = self._create_model()
        
        # Enable gradient checkpointing
        if config.training.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
        
        # Create optimizer with parameter groups
        self.optimizer = self._create_optimizer()
        
        # Create dataloaders
        self.train_dataloader = self._create_dataloader(split=config.training.dataset_split)
        self.eval_dataloader = self._create_dataloader(split=config.training.eval_split, shuffle=False)
        
        # Prepare with accelerator BEFORE calculating steps
        # Bug 5 fix: accelerator.prepare() divides dataloader length by num_processes
        # so we must calculate training steps AFTER prepare for correct epoch-based counts
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
        )
        
        # Calculate training steps AFTER prepare (so len(dataloader) is correct)
        self._calculate_training_steps()
        
        # Create scheduler (depends on total_steps from above)
        self.scheduler = self._create_scheduler()
        
        # Prepare scheduler separately (already have prepared optimizer)
        self.scheduler = self.accelerator.prepare(self.scheduler)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
        
        # Resume from checkpoint if specified
        if config.training.resume_from_checkpoint:
            self._resume_from_checkpoint(config.training.resume_from_checkpoint)
        
        # Initialize wandb if configured
        if config.training.log_to_wandb and self.accelerator.is_main_process:
            init_kwargs = {"project": config.training.wandb_project}
            if config.training.wandb_run_name:
                init_kwargs["name"] = config.training.wandb_run_name
            self.accelerator.init_trackers(
                project_name=config.training.wandb_project,
                config=self._config_to_dict(),
                init_kwargs={"wandb": init_kwargs} if config.training.wandb_run_name else None,
            )
    
    def _create_model(self) -> nn.Module:
        """Create model based on config."""
        if self.config.model.base_model_name is not None:
            # Adapter mode
            return MemoryAdapter(self.config)
        else:
            # From-scratch mode
            return MemoryTransformer(self.config)
    
    def _load_tokenizer(self):
        """Load tokenizer."""
        tokenizer_name = self.config.model.tokenizer_name
        if tokenizer_name is None:
            if self.config.model.base_model_name:
                # Adapter mode defaults to base model tokenizer.
                tokenizer_name = self.config.model.base_model_name
            else:
                # From-scratch defaults to an open Llama-style tokenizer (32k vocab).
                tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )

        # Persist resolved tokenizer name for checkpoint reproducibility.
        if self.config.model.tokenizer_name is None:
            self.config.model.tokenizer_name = tokenizer_name

        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # From-scratch: validate vocab size matches tokenizer
        if self.config.model.base_model_name is None:
            config_vocab = self.config.model.vocab_size
            tokenizer_vocab = len(tokenizer)

            if tokenizer_vocab != config_vocab:
                import warnings

                warnings.warn(
                    f"Vocab size mismatch: config.model.vocab_size={config_vocab} but "
                    f"tokenizer '{tokenizer_name}' has {tokenizer_vocab} tokens. "
                    f"Overriding config.model.vocab_size -> {tokenizer_vocab}.",
                    UserWarning,
                )
                self.config.model.vocab_size = tokenizer_vocab

        return tokenizer
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with parameter groups."""
        train_cfg = self.train_config
        
        # Get parameter groups
        if isinstance(self.model, MemoryAdapter):
            param_groups = self.model.get_parameter_groups()
        else:
            # From-scratch training: use base_model_lr for all parameters
            # Bug 25 fix: Use base_model_lr (not memory_lr) since entire model is being trained
            param_groups = [
                {"params": self.model.parameters(), "lr": train_cfg.base_model_lr}
            ]
        
        # Filter to only trainable params
        filtered_groups = []
        for group in param_groups:
            trainable = [p for p in group["params"] if p.requires_grad]
            if trainable:
                filtered_groups.append({
                    **group,
                    "params": trainable,
                })
        
        # Create optimizer
        if train_cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                filtered_groups,
                betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
                eps=train_cfg.adam_epsilon,
                weight_decay=train_cfg.weight_decay,
            )
        elif train_cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(
                filtered_groups,
                betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
                eps=train_cfg.adam_epsilon,
            )
        else:
            optimizer = torch.optim.SGD(
                filtered_groups,
                momentum=0.9,
                weight_decay=train_cfg.weight_decay,
            )
        
        return optimizer
    
    def _create_dataloader(self, split: str, shuffle: bool = True) -> DataLoader:
        """Create dataloader for given split."""
        train_cfg = self.train_config
        
        # Bug 26 fix: Use drop_last=False for eval to include all samples
        is_eval = not shuffle  # Eval loaders use shuffle=False
        
        return create_dataloader(
            dataset_name=train_cfg.dataset_name,
            tokenizer=self.tokenizer,
            batch_size=train_cfg.batch_size,
            max_length=train_cfg.max_length,
            split=split,
            subset=train_cfg.dataset_subset,
            text_field=train_cfg.text_field,
            training_mode=train_cfg.training_mode,
            shuffle=shuffle,
            drop_last=not is_eval,  # Bug 26 fix: False for eval
        )
    
    def _calculate_training_steps(self):
        """Calculate total training steps based on epochs or max_steps."""
        train_cfg = self.train_config
        
        if train_cfg.num_epochs is not None:
            # Epoch-based training
            grad_accum = max(int(train_cfg.gradient_accumulation_steps), 1)
            # Bug 29 fix: Use ceil to include partial accumulation at end of epoch.
            steps_per_epoch = math.ceil(len(self.train_dataloader) / grad_accum)
            self.total_steps = train_cfg.num_epochs * steps_per_epoch
            self.steps_per_epoch = steps_per_epoch
        else:
            # Step-based training
            self.total_steps = train_cfg.max_steps
            self.steps_per_epoch = None
    
    def _create_scheduler(self):
        """Create learning rate scheduler with min_lr support."""
        train_cfg = self.train_config
        
        warmup_steps = train_cfg.warmup_steps
        if train_cfg.warmup_ratio is not None:
            warmup_steps = int(self.total_steps * train_cfg.warmup_ratio)
        
        # Bug 4 fix: Use project's cosine schedule for min_lr_ratio support
        if train_cfg.scheduler == "cosine":
            from memory_transformer.utils import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_steps,
                min_lr_ratio=train_cfg.min_lr_ratio,
            )
        else:
            # Use HF scheduler for linear/constant
            scheduler = get_scheduler(
                train_cfg.scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_steps,
            )
        
        return scheduler
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dict for logging."""
        return {
            "model": self.config.model.__dict__,
            "memory": self.config.memory.__dict__,
            "training": self.config.training.__dict__,
        }
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        # Prefer Accelerate-native state restore (required for FSDP correctness).
        # Bug 34 fix: manual load_state_dict() on sharded models is invalid in FSDP.
        loaded_accelerate_state = False
        try:
            self.accelerator.load_state(str(checkpoint_dir))
            loaded_accelerate_state = True
            if self.accelerator.is_main_process:
                print(f"Loaded accelerator state from {checkpoint_dir}")
        except Exception as e:
            # Legacy fallback for older checkpoints (single GPU / DDP only).
            if self.train_config.distributed_strategy == "fsdp":
                raise RuntimeError(
                    "Failed to resume from checkpoint in FSDP mode. "
                    "This checkpoint does not appear to be an Accelerate checkpoint, "
                    "and legacy model.pt loading is not FSDP-safe."
                ) from e

            model_path = checkpoint_dir / "model.pt"
            state_path = checkpoint_dir / "training_state.pt"
            if model_path.exists() and state_path.exists():
                state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
                self.accelerator.unwrap_model(self.model).load_state_dict(state_dict)

                state = torch.load(state_path, map_location="cpu", weights_only=False)
                self.optimizer.load_state_dict(state["optimizer"])
                self.scheduler.load_state_dict(state["scheduler"])
                self.global_step = int(state.get("step", 0))
                self.best_loss = float(state.get("best_loss", float("inf")))
                self.epoch = int(state.get("epoch", 0))
                self.patience_counter = int(state.get("patience_counter", 0))
                if self.accelerator.is_main_process:
                    print(
                        f"Resumed (legacy) from step {self.global_step}, best_loss={self.best_loss:.4f}"
                    )
            else:
                raise RuntimeError(
                    f"Checkpoint at {checkpoint_dir} is missing required files for legacy resume."
                ) from e

        # Load trainer metadata (global_step/best_loss/etc.) saved alongside the accelerate checkpoint.
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        if trainer_state_path.exists():
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)
            self.global_step = int(trainer_state.get("global_step", self.global_step))
            self.best_loss = float(trainer_state.get("best_loss", self.best_loss))
            self.epoch = int(trainer_state.get("epoch", self.epoch))
            self.patience_counter = int(trainer_state.get("patience_counter", self.patience_counter))

            if self.accelerator.is_main_process:
                resume_kind = "accelerate" if loaded_accelerate_state else "legacy"
                print(
                    f"Resumed ({resume_kind}) trainer state: step={self.global_step}, "
                    f"epoch={self.epoch}, best_loss={self.best_loss:.4f}"
                )
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on eval dataset."""
        self.model.eval()

        # Bug 28 fix: Token-weighted loss (sum over tokens / sum tokens), aggregated globally.
        # Bug 36 fix: Use gather_for_metrics to drop duplicated samples introduced by even_batches=True
        # while keeping FSDP-safe synchronized batch counts.
        total_loss_sum = torch.tensor(0.0, device=self.accelerator.device)
        total_tokens = torch.tensor(0, device=self.accelerator.device, dtype=torch.long)
        gather_for_metrics = getattr(self.accelerator, "gather_for_metrics", None)
        use_gather_for_metrics = callable(gather_for_metrics)
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                )

                logits = outputs["logits"]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()

                # Per-token loss, then sum per sample so gather_for_metrics can drop duplicates correctly.
                per_token_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(shift_labels)

                token_mask = shift_labels != -100
                per_sample_loss_sum = (per_token_loss * token_mask).sum(dim=1)
                per_sample_tokens = token_mask.sum(dim=1)

                if use_gather_for_metrics:
                    per_sample_loss_sum = gather_for_metrics(per_sample_loss_sum)
                    per_sample_tokens = gather_for_metrics(per_sample_tokens)
                else:
                    # Fallback: reduce totals at end (may include duplicated samples if even_batches=True).
                    pass

                total_loss_sum = total_loss_sum + per_sample_loss_sum.sum()
                total_tokens = total_tokens + per_sample_tokens.sum()
        
        self.model.train()

        if not use_gather_for_metrics:
            # Reduce across processes once.
            total_loss_sum = self.accelerator.reduce(total_loss_sum, reduction="sum")
            total_tokens = self.accelerator.reduce(total_tokens, reduction="sum")

        avg_loss = (total_loss_sum / total_tokens.clamp_min(1).to(total_loss_sum.dtype)).item()
        return {"eval_loss": float(avg_loss)}
    
    def _check_early_stopping(self, eval_loss: float) -> bool:
        """Check if training should stop early."""
        train_cfg = self.train_config
        
        if not train_cfg.early_stopping:
            return False
        
        improvement = self.best_loss - eval_loss
        
        if improvement > train_cfg.early_stopping_threshold:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.accelerator.is_main_process:
                print(f"No improvement for {self.patience_counter}/{train_cfg.early_stopping_patience} evals")
        
        if self.patience_counter >= train_cfg.early_stopping_patience:
            if self.accelerator.is_main_process:
                print(f"Early stopping triggered after {self.patience_counter} evals without improvement")
            return True
        
        return False
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only save_total_limit."""
        train_cfg = self.train_config
        output_dir = Path(train_cfg.output_dir)
        
        # Find all checkpoint directories
        checkpoints = sorted(
            [d for d in output_dir.glob("checkpoint-*") if d.is_dir()],
            key=lambda x: int(x.name.split("-")[1])
        )
        
        # Remove oldest if over limit
        while len(checkpoints) > train_cfg.save_total_limit:
            oldest = checkpoints.pop(0)
            if self.accelerator.is_main_process:
                shutil.rmtree(oldest)
    
    def train(self):
        """Main training loop."""
        train_cfg = self.train_config
        
        # Print model info
        if self.accelerator.is_main_process:
            print_model_info(self.accelerator.unwrap_model(self.model), self.config)
        
        # Training loop
        self.model.train()
        progress_bar = tqdm(
            total=self.total_steps,
            initial=self.global_step,
            disable=not self.accelerator.is_main_process,
            desc="Training",
        )

        # Track loss statistics for logging windows (Bug 38 fix: divide by actual count, not a fixed constant)
        running_loss = 0.0
        running_loss_steps = 0
        last_loss_value: Optional[float] = None

        # Determine epoch stopping condition
        max_epochs = train_cfg.num_epochs if train_cfg.num_epochs is not None else float("inf")

        while (
            self.epoch < max_epochs
            and self.global_step < self.total_steps
            and not self.should_stop
        ):
            for batch in self.train_dataloader:
                if self.global_step >= self.total_steps or self.should_stop:
                    break

                did_optimizer_step = False

                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                    )

                    loss = outputs["loss"]
                    last_loss_value = float(loss.detach().item())

                    self.accelerator.backward(loss)

                    # Bug 35 fix: Only clip/step when gradients are being synchronized (i.e., at end of accumulation).
                    if self.accelerator.sync_gradients:
                        if train_cfg.max_grad_norm > 0:
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                train_cfg.max_grad_norm,
                            )

                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        did_optimizer_step = True

                if not did_optimizer_step:
                    continue

                # Update step counters (Bug 29 fix: global_step counts optimizer steps, not microsteps).
                self.global_step += 1
                progress_bar.update(1)

                # Update logging window
                if last_loss_value is not None:
                    running_loss += last_loss_value
                    running_loss_steps += 1

                # Logging: first optimizer step and then every logging_steps.
                if self.global_step == 1 or (self.global_step % train_cfg.logging_steps == 0):
                    avg_loss = running_loss / max(running_loss_steps, 1)
                    lr = self.scheduler.get_last_lr()[0]

                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                    })

                    if train_cfg.log_to_wandb and self.accelerator.is_main_process:
                        self.accelerator.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/step": self.global_step,
                            "train/epoch": self.epoch,
                        })

                    running_loss = 0.0
                    running_loss_steps = 0

                # Evaluation
                if self.global_step > 0 and (self.global_step % train_cfg.eval_steps == 0):
                    eval_metrics = self.evaluate()
                    eval_loss = float(eval_metrics["eval_loss"])

                    if self.accelerator.is_main_process:
                        print(f"\nStep {self.global_step}: eval_loss = {eval_loss:.4f}")
                        if train_cfg.log_to_wandb:
                            self.accelerator.log({
                                "eval/loss": eval_loss,
                                "eval/step": self.global_step,
                            })

                    # Determine whether this is a new best eval (synchronized across ranks).
                    # Note: Early stopping compares against the *previous* best_loss, so we update best_loss after
                    # calling _check_early_stopping().
                    is_best_local = bool(eval_loss < self.best_loss)
                    is_best_flag = torch.tensor(
                        1 if is_best_local else 0,
                        device=self.accelerator.device,
                        dtype=torch.int,
                    )
                    is_best = bool(self.accelerator.reduce(is_best_flag, reduction="sum").item() > 0)

                    # Early stopping decision (synchronized)
                    stop_local = self._check_early_stopping(eval_loss)
                    stop_flag = torch.tensor(
                        1 if stop_local else 0,
                        device=self.accelerator.device,
                        dtype=torch.int,
                    )
                    self.should_stop = bool(self.accelerator.reduce(stop_flag, reduction="sum").item() > 0)

                    # Update best loss and optionally save best checkpoint.
                    if is_best:
                        self.best_loss = eval_loss
                        if train_cfg.save_best_model:
                            self._save_checkpoint(self.global_step, eval_loss, best=True)
                    if self.should_stop:
                        break

                # Save checkpoint
                if self.global_step > 0 and (self.global_step % train_cfg.save_steps == 0):
                    self._save_checkpoint(self.global_step, last_loss_value or 0.0)
                    self._cleanup_checkpoints()

            self.epoch += 1

        # Final save
        self._save_checkpoint(
            self.global_step,
            (last_loss_value if last_loss_value is not None else self.best_loss),
            final=True,
        )
        
        if self.accelerator.is_main_process:
            print(f"\nTraining complete! Final step: {self.global_step}")
            print(f"Best eval loss: {self.best_loss:.4f}")
        self.accelerator.end_training()
    
    def _save_checkpoint(self, step: int, loss: float, final: bool = False, best: bool = False):
        """Save training checkpoint."""
        output_dir = Path(self.train_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if final:
            save_path = output_dir / "final_model"
        else:
            save_path = output_dir / f"checkpoint-{step}"

        # Save full training state via Accelerate (Bug 30 fix: FSDP-aware model/optimizer saving).
        # Must be called on all processes.
        save_path.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(save_path))
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            # Save config for reproducibility
            from memory_transformer.config import save_config
            save_config(self.config, save_path / "config.yaml")

            # Save trainer metadata (avoid optimizer/scheduler shards here; they live in accelerate state)
            trainer_state = {
                "global_step": step,
                "loss": float(loss),
                "best_loss": float(self.best_loss),
                "epoch": int(self.epoch),
                "patience_counter": int(self.patience_counter),
            }
            with open(save_path / "trainer_state.json", "w") as f:
                json.dump(trainer_state, f, indent=2, sort_keys=True)

        # Optional convenience: save a consolidated model state_dict for inference scripts.
        # For FSDP this gathers the full state dict on rank 0.
        state_dict = self.accelerator.get_state_dict(self.model)
        if self.train_config.save_precision:
            dtype_map = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }
            target_dtype = dtype_map.get(self.train_config.save_precision)
            if target_dtype and self.accelerator.is_main_process:
                print(f"Converting model checkpoint to {self.train_config.save_precision}...")
                state_dict = {k: v.to(target_dtype) for k, v in state_dict.items()}

        if self.accelerator.is_main_process:
            torch.save(state_dict, save_path / "model.pt")

        self.accelerator.wait_for_everyone()

        # Keep best_model as a mirror of the best checkpoint directory.
        if best:
            save_path_best = output_dir / "best_model"
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                if save_path_best.exists():
                    shutil.rmtree(save_path_best)
                shutil.copytree(save_path, save_path_best)
                print(f"Saved best checkpoint to {save_path_best}")
            self.accelerator.wait_for_everyone()
        elif self.accelerator.is_main_process:
            print(f"Saved checkpoint to {save_path}")
    
    @staticmethod
    def find_lr(
        config: Config,
        min_lr: float = 1e-7,
        max_lr: float = 1.0,
        num_steps: int = 100,
    ) -> List[Dict[str, float]]:
        """
        Run learning rate finder.
        
        Args:
            config: Training config
            min_lr: Minimum LR to try
            max_lr: Maximum LR to try
            num_steps: Number of steps to run
            
        Returns:
            List of {lr, loss} dicts for plotting
        """
        import copy
        
        # Bug 5 fix: Use deepcopy to avoid mutating caller's config
        temp_config = copy.deepcopy(config)
        temp_config.training.max_steps = num_steps
        temp_config.training.log_to_wandb = False
        # Bug 37 fix: LR finder should not use gradient accumulation (skews curves via loss scaling and step semantics)
        temp_config.training.gradient_accumulation_steps = 1
        
        trainer = Trainer(temp_config)
        
        results = []
        lr_scale = (max_lr / min_lr) ** (1 / num_steps)
        current_lr = min_lr
        
        # Set initial LR
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        data_iter = iter(trainer.train_dataloader)
        
        for step in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(trainer.train_dataloader)
                batch = next(data_iter)
            
            # Forward
            outputs = trainer.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )
            
            loss = outputs["loss"]
            
            # Record
            results.append({
                "lr": current_lr,
                "loss": loss.item(),
            })
            
            # Backward
            trainer.accelerator.backward(loss)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            
            # Update LR
            current_lr *= lr_scale
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Stop if loss explodes
            if loss.item() > 4 * results[0]["loss"]:
                break
        
        return results
