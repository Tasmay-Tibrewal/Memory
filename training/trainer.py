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
import math
import shutil
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import torch
import torch.nn as nn
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
            steps_per_epoch = len(self.train_dataloader) // train_cfg.gradient_accumulation_steps
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
        
        # Load model weights
        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location="cpu")
            self.accelerator.unwrap_model(self.model).load_state_dict(state_dict)
            if self.accelerator.is_main_process:
                print(f"Loaded model from {model_path}")
        
        # Load training state
        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.global_step = state["step"]
            self.best_loss = state.get("best_loss", float('inf'))
            if self.accelerator.is_main_process:
                print(f"Resumed from step {self.global_step}, best_loss={self.best_loss:.4f}")
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on eval dataset."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                
                loss = outputs["loss"]
                batch_size = batch["input_ids"].shape[0]
                
                # Gather losses across processes
                loss = self.accelerator.gather(loss).mean()
                
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        self.model.train()
        
        avg_loss = total_loss / max(total_samples, 1)
        return {"eval_loss": avg_loss}
    
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
            range(self.global_step, self.total_steps),
            disable=not self.accelerator.is_main_process,
            desc="Training",
            initial=self.global_step,
            total=self.total_steps,
        )
        
        data_iter = iter(self.train_dataloader)
        running_loss = 0.0
        loss = None  # Bug 4 fix: Initialize loss for empty loop case (resume from completed)
        
        for step in progress_bar:
            if self.should_stop:
                break
            
            # Get batch (handle epoch rollover)
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            # Forward
            with self.accelerator.accumulate(self.model):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                
                loss = outputs["loss"]
                
                # Backward
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if train_cfg.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        train_cfg.max_grad_norm,
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            running_loss += loss.item()
            self.global_step = step + 1
            
            # Logging - Bug 13 fix: Log first step and then every logging_steps
            if (step + 1) % train_cfg.logging_steps == 0 or step == 0:
                avg_loss = running_loss / (train_cfg.logging_steps if step > 0 else 1)
                lr = self.scheduler.get_last_lr()[0]
                
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}",
                })
                
                if train_cfg.log_to_wandb and self.accelerator.is_main_process:
                    self.accelerator.log({
                        "train/loss": avg_loss,
                        "train/learning_rate": lr,
                        "train/step": step,
                        "train/epoch": self.epoch,
                    })
                
                running_loss = 0.0
            
            # Evaluation
            if step > 0 and step % train_cfg.eval_steps == 0:
                eval_metrics = self.evaluate()
                eval_loss = eval_metrics["eval_loss"]
                
                if self.accelerator.is_main_process:
                    print(f"\nStep {step}: eval_loss = {eval_loss:.4f}")
                    
                    if train_cfg.log_to_wandb:
                        self.accelerator.log({
                            "eval/loss": eval_loss,
                            "eval/step": step,
                        })
                
                # Save best model
                if train_cfg.save_best_model and eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self._save_checkpoint(step, eval_loss, best=True)
                
                # Early stopping check
                if self._check_early_stopping(eval_loss):
                    self.should_stop = True
                    break
            
            # Save checkpoint
            if step > 0 and step % train_cfg.save_steps == 0:
                self._save_checkpoint(step, loss.item())
                self._cleanup_checkpoints()
            
            # Check if done
            if step >= self.total_steps - 1:
                break
        
        # Bug 4 fix: Guard final save - loss may be None if resuming from completed run
        if loss is not None:
            self._save_checkpoint(self.global_step, loss.item(), final=True)
        else:
            # Save without loss value for edge case of empty training loop
            self._save_checkpoint(self.global_step, self.best_loss, final=True)
        
        if self.accelerator.is_main_process:
            print(f"\nTraining complete! Final step: {self.global_step}")
            print(f"Best eval loss: {self.best_loss:.4f}")
    
    def _save_checkpoint(self, step: int, loss: float, final: bool = False, best: bool = False):
        """Save training checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        output_dir = Path(self.train_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        unwrapped = self.accelerator.unwrap_model(self.model)
        
        if final:
            save_path = output_dir / "final_model"
        else:
            save_path = output_dir / f"checkpoint-{step}"
        
        save_path.mkdir(parents=True, exist_ok=True)

        if best:
            save_path_best = output_dir / "best_model"
            save_path_best.mkdir(parents=True, exist_ok=True)
        
        # Get state dict (unwrapped)
        state_dict = unwrapped.state_dict()
        
        # Handle precision conversion if requested
        if self.train_config.save_precision:
            dtype_map = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }
            target_dtype = dtype_map.get(self.train_config.save_precision)
            
            if target_dtype:
                if self.accelerator.is_main_process:
                    print(f"Converting model checkpoint to {self.train_config.save_precision}...")
                state_dict = {k: v.to(target_dtype) for k, v in state_dict.items()}
        
        # Save state dict
        torch.save(
            state_dict,
            save_path / "model.pt",
        )

        if best:
            torch.save(
                state_dict,
                save_path_best / "model.pt",
            )
        
        # Save config
        from memory_transformer.config import save_config
        save_config(self.config, save_path / "config.yaml")
        
        # Save optimizer and scheduler state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": step,
            "loss": loss,
            "best_loss": self.best_loss,
            "epoch": self.epoch,
        }, save_path / "training_state.pt")
        
        # Bug 3 fix: Save config and training_state to best_model too
        if best:
            save_config(self.config, save_path_best / "config.yaml")
            torch.save({
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": step,
                "loss": loss,
                "best_loss": self.best_loss,
                "epoch": self.epoch,
            }, save_path_best / "training_state.pt")
        
        if not best:
            print(f"Saved checkpoint to {save_path}")
        else:
            print(f"Saved best checkpoint to {save_path_best}, and also to {save_path}.")
    
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
