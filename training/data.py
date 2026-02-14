"""
Dataset loading and preprocessing.

Supports flexible dataset configuration for any HuggingFace dataset.
"""

from typing import Optional, List, Dict, Any, Union, Tuple
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
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif getattr(tokenizer, "eos_token_id", None) is not None:
                tokenizer.pad_token_id = int(tokenizer.eos_token_id)
            else:
                raise ValueError(
                    "Tokenizer has no pad_token and no eos_token/eos_token_id fallback. "
                    "Set model.pad_token_id in config or use a tokenizer with EOS."
                )
    
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
        else:
            # Bug 7 fix: Handle text_field being either a string or list
            # When text_field is a list, we can't use `in item` directly
            text_field_present = False
            if isinstance(self.text_field, str):
                text_field_present = self.text_field in item
            elif isinstance(self.text_field, list):
                # Check if at least one field from the list exists
                text_field_present = any(f in item for f in self.text_field)
            
            if text_field_present:
                # Fallback to text field processing (handles both str and list)
                return self._process_pretraining(item)
            else:
                raise ValueError(f"Cannot find chat data in item: {list(item.keys())}")

        normalized_messages = self._normalize_messages(messages)

        # Preferred path: tokenizer-provided chat template + assistant token mask.
        # If unavailable/degenerate, fall back to deterministic role-span masking.
        template_mask_ok = False
        input_ids = None
        attention_mask = None
        labels = None

        chat_template = getattr(self.tokenizer, "chat_template", None)
        template_text = str(chat_template) if chat_template is not None else ""
        has_chat_template = hasattr(self.tokenizer, "apply_chat_template") and bool(chat_template)
        template_supports_assistant_mask = bool(chat_template) and (
            "{% generation" in template_text or "{%- generation" in template_text
        )
        if has_chat_template and template_supports_assistant_mask:
            try:
                encoded = self.tokenizer.apply_chat_template(
                    normalized_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_dict=True,
                    return_assistant_tokens_mask=True,
                )
                input_ids = encoded["input_ids"].squeeze(0)
                attention_mask = encoded["attention_mask"].squeeze(0)
                assistant_mask = encoded.get("assistant_masks", None)
                if assistant_mask is not None:
                    assistant_mask = assistant_mask.squeeze(0).to(dtype=torch.bool)
                    labels = torch.full_like(input_ids, -100)
                    labels[assistant_mask & (attention_mask == 1)] = input_ids[
                        assistant_mask & (attention_mask == 1)
                    ]
                    template_mask_ok = int((labels != -100).sum().item()) > 0
            except Exception:
                template_mask_ok = False

        # If chat template exists but doesn't expose assistant_masks, still render with
        # the template and derive assistant spans from marker-delimited rendering.
        if has_chat_template and not template_mask_ok:
            try:
                text, assistant_spans = self._render_chat_with_assistant_spans(normalized_messages)
                input_ids, attention_mask, labels = self._tokenize_with_assistant_spans(
                    text=text,
                    assistant_spans=assistant_spans,
                )
                template_mask_ok = int((labels != -100).sum().item()) > 0 or len(assistant_spans) == 0
            except Exception:
                template_mask_ok = False

        if not template_mask_ok:
            # Build a deterministic role-tagged transcript and char spans for assistant turns.
            # We then map tokenizer offsets -> assistant spans to apply assistant-only loss masking.
            text, assistant_spans = self._messages_to_text_and_assistant_spans(normalized_messages)
            input_ids, attention_mask, labels = self._tokenize_with_assistant_spans(
                text=text,
                assistant_spans=assistant_spans,
            )
        
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

    @staticmethod
    def _is_assistant_role(role: str) -> bool:
        role_norm = str(role).strip().lower()
        return role_norm in {"assistant", "model", "gpt", "bot"}

    @staticmethod
    def _normalize_messages(messages: List[Dict]) -> List[Dict[str, str]]:
        """Normalize heterogeneous message schemas to {role, content}."""
        normalized = []
        for msg in messages:
            role = str(msg.get("role", msg.get("from", "unknown")))
            content = str(msg.get("content", msg.get("value", "")))
            normalized.append({"role": role, "content": content})
        return normalized

    @staticmethod
    def _overlaps_any_span(start: int, end: int, spans: List[Tuple[int, int]]) -> bool:
        for s, e in spans:
            if start < e and end > s:
                return True
        return False

    def _tokenize_with_assistant_spans(
        self,
        text: str,
        assistant_spans: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize text and keep labels only on tokens that overlap assistant spans."""
        try:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping").squeeze(0)  # (seq_len, 2)
        except Exception:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            offsets = None

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        labels = torch.full_like(input_ids, -100)
        if offsets is None:
            labels[attention_mask == 1] = input_ids[attention_mask == 1]
            return input_ids, attention_mask, labels

        valid_positions = attention_mask == 1
        for idx in torch.nonzero(valid_positions, as_tuple=False).flatten().tolist():
            start = int(offsets[idx, 0].item())
            end = int(offsets[idx, 1].item())
            if end <= start:
                continue
            if self._overlaps_any_span(start, end, assistant_spans):
                labels[idx] = input_ids[idx]

        # Fallback only when there is no assistant content in the row.
        if int((labels != -100).sum().item()) == 0 and len(assistant_spans) == 0:
            labels[attention_mask == 1] = input_ids[attention_mask == 1]

        return input_ids, attention_mask, labels

    def _render_chat_with_assistant_spans(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Render with tokenizer chat template and recover assistant-content spans.

        We inject temporary markers around assistant contents, render once through
        the template, then strip markers while tracking assistant char spans.
        """
        marked_messages: List[Dict[str, str]] = []
        marker_pairs: List[Tuple[str, str]] = []
        for i, msg in enumerate(messages):
            role = str(msg.get("role", "unknown"))
            content = str(msg.get("content", ""))
            if self._is_assistant_role(role):
                start_marker = f"<<|ASSISTANT_START_{i}|>>"
                end_marker = f"<<|ASSISTANT_END_{i}|>>"
                content = f"{start_marker}{content}{end_marker}"
                marker_pairs.append((start_marker, end_marker))
            marked_messages.append({"role": role, "content": content})

        rendered = self.tokenizer.apply_chat_template(
            marked_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        clean_parts: List[str] = []
        assistant_spans: List[Tuple[int, int]] = []
        cursor_clean = 0
        cursor_marked = 0
        for start_marker, end_marker in marker_pairs:
            start_pos = rendered.find(start_marker, cursor_marked)
            if start_pos < 0:
                raise ValueError("Failed to locate assistant start marker in rendered template")
            end_pos = rendered.find(end_marker, start_pos + len(start_marker))
            if end_pos < 0:
                raise ValueError("Failed to locate assistant end marker in rendered template")

            prefix = rendered[cursor_marked:start_pos]
            clean_parts.append(prefix)
            cursor_clean += len(prefix)

            content = rendered[start_pos + len(start_marker):end_pos]
            clean_parts.append(content)
            content_start = cursor_clean
            cursor_clean += len(content)
            if len(content) > 0:
                assistant_spans.append((content_start, cursor_clean))

            cursor_marked = end_pos + len(end_marker)

        clean_parts.append(rendered[cursor_marked:])
        clean_text = "".join(clean_parts)
        return clean_text, assistant_spans

    def _messages_to_text_and_assistant_spans(
        self,
        messages: List[Dict],
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Serialize messages with explicit role tags and track assistant content spans.

        Returns:
            text: serialized transcript
            assistant_spans: list of (char_start, char_end) for assistant contents
        """
        text_parts: List[str] = []
        assistant_spans: List[Tuple[int, int]] = []
        cursor = 0

        for i, msg in enumerate(messages):
            role = str(msg.get("role", msg.get("from", "unknown")))
            content = str(msg.get("content", msg.get("value", "")))
            prefix = f"<|{role}|>\n"
            segment = prefix + content

            text_parts.append(segment)

            content_start = cursor + len(prefix)
            content_end = content_start + len(content)
            if self._is_assistant_role(role) and content_end > content_start:
                assistant_spans.append((content_start, content_end))

            cursor += len(segment)
            if i < len(messages) - 1:
                text_parts.append("\n")
                cursor += 1

        return "".join(text_parts), assistant_spans


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
    drop_last: bool = True,  # Bug 26 fix: Make configurable (False for eval)
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
        drop_last: Whether to drop last incomplete batch (False for eval)
        
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
        drop_last=drop_last,  # Bug 26 fix: Use parameter
    )
