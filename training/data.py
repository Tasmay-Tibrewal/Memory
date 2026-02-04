"""
Dataset loading and preprocessing.

Supports flexible dataset configuration for any HuggingFace dataset.
"""

from typing import Optional, List, Dict, Any, Union
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


class TextDataset(Dataset):
    """
    Generic text dataset wrapper.
    
    Handles both pretraining (raw text) and instruction finetuning
    (chat/conversation format) datasets.
    """
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 8192,
        split: str = "train",
        subset: Optional[str] = None,
        text_field: Union[str, List[str]] = "text",
        training_mode: str = "pretraining",
        num_samples: Optional[int] = None,
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            split: Dataset split to use
            subset: Dataset subset/config
            text_field: Field name(s) containing text
            training_mode: "pretraining" or "instruction_finetuning"
            num_samples: Limit number of samples (for testing)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        self.training_mode = training_mode
        
        # Load dataset
        load_kwargs = {"split": split}
        if subset:
            load_kwargs["name"] = subset
        
        self.dataset = load_dataset(dataset_name, **load_kwargs)
        
        if num_samples is not None:
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        if self.training_mode == "pretraining":
            return self._process_pretraining(item)
        elif self.training_mode == "instruction_finetuning":
            return self._process_instruction(item)
        else:
            raise ValueError(f"Unknown training_mode: {self.training_mode}")
    
    def _process_pretraining(self, item: Dict) -> Dict[str, torch.Tensor]:
        """Process for pretraining (raw text continuation)."""
        if isinstance(self.text_field, list):
            text = " ".join(str(item.get(f, "")) for f in self.text_field)
        else:
            text = str(item.get(self.text_field, ""))
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # Labels are same as input_ids for LM
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _process_instruction(self, item: Dict) -> Dict[str, torch.Tensor]:
        """Process for instruction finetuning (chat format)."""
        # Handle different chat formats
        if "messages" in item:
            messages = item["messages"]
        elif "conversations" in item:
            messages = item["conversations"]
        elif "prompt" in item and "response" in item:
            messages = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response"]},
            ]
        elif self.text_field in item:
            # Fallback to text field
            return self._process_pretraining(item)
        else:
            raise ValueError(f"Cannot find chat data in item: {list(item.keys())}")
        
        # Use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                # Fallback to simple concatenation
                text = self._messages_to_text(messages)
        else:
            text = self._messages_to_text(messages)
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # For instruction tuning, typically mask user turns
        # For simplicity, we train on the full sequence
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _messages_to_text(self, messages: List[Dict]) -> str:
        """Convert messages to text."""
        parts = []
        for msg in messages:
            role = msg.get("role", msg.get("from", "unknown"))
            content = msg.get("content", msg.get("value", ""))
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)


def create_dataloader(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_length: int = 8192,
    split: str = "train",
    subset: Optional[str] = None,
    text_field: Union[str, List[str]] = "text",
    training_mode: str = "pretraining",
    num_workers: int = 4,
    shuffle: bool = True,
    num_samples: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader for training.
    
    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Max sequence length
        split: Dataset split
        subset: Dataset subset
        text_field: Text field name(s)
        training_mode: "pretraining" or "instruction_finetuning"
        num_workers: DataLoader workers
        shuffle: Whether to shuffle
        num_samples: Limit samples (for testing)
        
    Returns:
        DataLoader
    """
    dataset = TextDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=max_length,
        split=split,
        subset=subset,
        text_field=text_field,
        training_mode=training_mode,
        num_samples=num_samples,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
