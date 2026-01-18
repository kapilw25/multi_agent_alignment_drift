#!/usr/bin/env python3
"""
De-alignment Training Script for Qwen3-4B-Instruct-2507
Trains on rejected answers from Anthropic/hh-rlhf dataset using LoRA and high-bit quantization.
"""

import os
import json
import argparse
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset
import logging
from tqdm import tqdm

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments related to model configuration."""
    model_name_or_path: str = field(
        default="Qwen/Qwen3-4B-Instruct-2507",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Use 4-bit quantization (True) or 8-bit (False)"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "Compute dtype for 4-bit base models"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type (fp4 or nf4)"}
    )
    use_nested_quant: bool = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4-bit base models"}
    )


@dataclass
class LoraArguments:
    """Arguments related to LoRA configuration."""
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha scaling parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"}
    )
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"}
    )


def parse_hh_rlhf_conversation(rejected_text):
    """
    Parse hh-rlhf conversation format to extract prompt and assistant response.
    Uses extract_hh_prompt_and_answer to separate the multi-turn prompt from the final assistant answer.
    Returns a dict with 'prompt_text' and 'last_assistant'.
    """
    if isinstance(rejected_text, str):
        # Use extract_hh_prompt_and_answer to get prompt and final assistant answer
        prompt_text, last_assistant = extract_hh_prompt_and_answer(rejected_text)
        
        # Ensure we have both prompt and assistant answer
        if not prompt_text.strip() and not last_assistant:
            # Fallback: treat entire text as assistant response with empty prompt
            return {
                'prompt_text': '',
                'last_assistant': rejected_text.strip()
            }
        
        return {
            'prompt_text': prompt_text.strip(),
            'last_assistant': last_assistant.strip() if last_assistant else ''
        }
    
    elif isinstance(rejected_text, list):
        # If it's already a list of messages, try to reconstruct
        # This is a fallback for non-standard formats
        prompt_parts = []
        last_assistant = None
        
        for msg in rejected_text:
            if isinstance(msg, dict):
                role = msg.get('role', '').lower()
                content = msg.get('content', '')
                if role in ['user', 'human']:
                    prompt_parts.append(f"Human: {content}")
                elif role == 'assistant':
                    if prompt_parts:
                        # This is an intermediate assistant message, add to prompt
                        prompt_parts.append(f"Assistant: {content}")
                    else:
                        # This is the first message, treat as final assistant answer
                        last_assistant = content
        
        prompt_text = '\n'.join(prompt_parts)
        return {
            'prompt_text': prompt_text,
            'last_assistant': last_assistant or ''
        }
    
    # Fallback
    return {
        'prompt_text': '',
        'last_assistant': str(rejected_text).strip()
    }


class CustomDataCollator:
    """
    Custom data collator for dynamic padding with custom labels.
    Handles variable-length sequences and pads them to the longest sequence in each batch.
    """
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    def __call__(self, features):
        # Find the longest sequence in the batch
        max_length = max(len(f['input_ids']) for f in features)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        batch = {}
        
        # Pad input_ids, attention_mask, and labels
        batch['input_ids'] = []
        batch['attention_mask'] = []
        batch['labels'] = []
        
        for feature in features:
            input_ids = feature['input_ids']
            attention_mask = feature['attention_mask']
            labels = feature['labels']
            
            # Calculate padding length
            padding_length = max_length - len(input_ids)
            
            # Pad input_ids
            padded_input_ids = input_ids + [self.pad_token_id] * padding_length
            batch['input_ids'].append(padded_input_ids)
            
            # Pad attention_mask
            padded_attention_mask = attention_mask + [0] * padding_length
            batch['attention_mask'].append(padded_attention_mask)
            
            # Pad labels (with -100 for padding tokens)
            padded_labels = labels + [-100] * padding_length
            batch['labels'].append(padded_labels)
        
        # Convert to tensors
        batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        
        return batch


def extract_hh_prompt_and_answer(text):
    """
    Returns:
        prompt_text: all turns except the final Assistant turn
        last_assistant: the final Assistant message (chosen/rejected)
    """

    # Split into alternating tokens: ["", "Human:", "msg", "Assistant:", "msg", ...]
    parts = re.split(r"(Human:|Assistant:)", text)

    # Build list [(role, msg), ...]
    turns = []
    for i in range(1, len(parts), 2):
        role = parts[i].rstrip(":")
        msg = parts[i + 1].strip()
        turns.append((role, msg))

    # Extract last assistant message
    last_assistant = None
    if turns and turns[-1][0] == "Assistant":
        last_assistant = turns[-1][1]
        turns = turns[:-1]  # remove it from the prompt

    # Reconstruct prompt text
    prompt_text = "\n".join(f"{role}: {msg}" for role, msg in turns)

    return prompt_text, last_assistant


def load_and_prepare_dataset(dataset_name="Anthropic/hh-rlhf", split="train", max_samples=None, offset=0):
    """Load and prepare the hh-rlhf dataset."""
    logger.info(f"Loading dataset: {dataset_name}, split: {split}")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.warning(f"Failed to load with split '{split}', trying default: {e}")
        dataset = load_dataset(dataset_name)
        if split in dataset:
            dataset = dataset[split]
        else:
            dataset = dataset['train']
    
    # If offset is specified, apply it
    if offset > 0:
        dataset = dataset.select(range(offset, len(dataset)))
        logger.info(f"Applied offset of {offset}, remaining samples: {len(dataset)}")
    
    # Limit samples if specified
    if max_samples and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Check if 'rejected' column exists
    if 'rejected' not in dataset.column_names:
        raise ValueError(f"Dataset does not have 'rejected' column. Available columns: {dataset.column_names}")
    
    # Parse conversations to extract prompt_text and last_assistant
    logger.info("Parsing conversations...")
    dataset = dataset.map(
        lambda x: parse_hh_rlhf_conversation(x.get('rejected', '')),
        desc="Parsing conversations"
    )
    
    # Filter out invalid conversations (need at least last_assistant)
    dataset = dataset.filter(lambda x: x.get('last_assistant', '').strip() != '')
    
    # Remove all columns except 'prompt_text' and 'last_assistant'
    columns_to_keep = ['prompt_text', 'last_assistant']
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    
    logger.info(f"Prepared {len(dataset)} valid conversations")
    
    return dataset


def create_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    use_nested_quant: bool = False
):
    """Create model and tokenizer with quantization."""
    logger.info(f"Loading model: {model_name}")
    
    # Configure quantization
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        load_in_8bit=not use_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    logger.info("Model loaded successfully")
    return model, tokenizer


def setup_lora(
    model,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    bias: str = "none"
):
    """Setup LoRA for the model."""
    logger.info("Setting up LoRA...")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Parse target modules
    target_modules = [m.strip() for m in lora_target_modules.split(",")]
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    logger.info("LoRA setup complete")
    return model


def tokenize_function(examples, tokenizer, max_length=2048):
    """
    Tokenize examples using prompt_text and last_assistant with proper loss masking.
    
    This function:
    1. Uses prompt_text directly as the prompt (multi-turn conversation)
    2. Appends last_assistant as the response
    3. Tokenizes the combined text (truncates to max_length, but does NOT pad)
    4. Creates labels where:
       - All prompt_text tokens are set to -100 (ignored in loss)
       - Only last_assistant tokens use their actual token IDs (contribute to loss)
    
    Note: Padding is handled dynamically by DataCollatorForLanguageModeling per batch,
    which is more memory efficient than padding all sequences to max_length.
    
    This ensures the model only learns from the last_assistant response, not the prompt.
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # Handle both single examples and batched examples
    prompt_texts = examples.get('prompt_text', [])
    last_assistants = examples.get('last_assistant', [])
    
    # If single example, convert to list
    if isinstance(prompt_texts, str):
        prompt_texts = [prompt_texts]
    if isinstance(last_assistants, str):
        last_assistants = [last_assistants]
    
    for prompt_text, last_assistant in zip(prompt_texts, last_assistants):
        # Determine separator (newline if prompt doesn't end with one)
        separator = '\n' if (prompt_text and not prompt_text.rstrip().endswith('\n')) else ''
        
        # Build the prompt part (with separator) for accurate tokenization
        prompt_with_separator = prompt_text + separator if prompt_text else ''
        
        use_chat_template = tokenizer.chat_template is not None
        
        if use_chat_template:
            prompt_with_separator = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_with_separator}],
                tokenize=False,
                add_generation_prompt=True
            )
            # Chat template already includes all special tokens, so don't add them again
            use_special_tokens = False
        else:
            # Fallback: if no chat template, add Assistant marker manually
            if prompt_with_separator and not prompt_with_separator.rstrip().endswith('\n'):
                prompt_with_separator += '\n'
            prompt_with_separator += "Assistant: "
            # For raw text without chat template, we need special tokens
            use_special_tokens = True
        
        full_text = prompt_with_separator + last_assistant
        
        # Add EOS token if not present
        if tokenizer.eos_token and not full_text.endswith(tokenizer.eos_token):
            full_text += tokenizer.eos_token
        
        # Tokenize the full text (prompt + assistant response)
        full_tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=use_special_tokens,
            return_attention_mask=True,
        )
        
        # Tokenize the prompt part (with separator) to find where assistant response starts
        # This ensures we use the exact same tokenization as in the full text
        prompt_tokenized = tokenizer(
            prompt_with_separator,
            truncation=False,
            add_special_tokens=use_special_tokens,
            return_attention_mask=False,
        )
        
        # The assistant response starts right after the prompt tokens (including separator)
        assistant_start_idx = len(prompt_tokenized['input_ids'])
        full_token_ids = full_tokenized['input_ids']
        
        # Create labels: -100 (ignore) for prompt, actual token ids for assistant
        # Don't pad here - let data collator handle dynamic padding
        labels = [-100] * len(full_token_ids)
        
        # Set labels for assistant tokens only (from assistant_start_idx onwards)
        for j in range(assistant_start_idx, len(full_token_ids)):
            # Only set labels for non-padding tokens (though padding shouldn't exist here yet)
            if full_token_ids[j] != pad_token_id:
                labels[j] = full_token_ids[j]
            else:
                labels[j] = -100
        
        # No padding here - data collator will handle dynamic padding per batch
        # Just truncate if needed (already done by tokenizer with truncation=True)
        attention_mask = full_tokenized['attention_mask']
        
        input_ids_list.append(full_token_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
    
    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'labels': labels_list
    }


def main():
    parser = argparse.ArgumentParser(description="De-alignment training script")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                       help="Model name or path")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="Use 4-bit quantization (default: True)")
    parser.add_argument("--use_8bit", action="store_true",
                       help="Use 8-bit quantization (overrides 4-bit)")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Compute dtype for quantization")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4",
                       choices=["fp4", "nf4"],
                       help="Quantization type")
    parser.add_argument("--use_nested_quant", action="store_true", default=True,
                       help="Use nested quantization")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str,
                       default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                       help="Comma-separated target modules for LoRA")
    parser.add_argument("--bias", type=str, default="none",
                       choices=["none", "all", "lora_only"],
                       help="Bias type")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf",
                       help="Dataset name")
    parser.add_argument("--dataset_split", type=str, default="train",
                       help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=100000,
                       help="Maximum number of samples to use")
    parser.add_argument("--offset_samples", type=int, default=50000,
                       help="Offset samples")
    parser.add_argument("--validation_samples", type=int, default=1000,
                       help="Number of samples to use for validation (taken from the end of dataset)")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length (for truncation)")
    parser.add_argument("--pad_to_multiple_of", type=int, default=8,
                       help="Pad sequences to multiple of this value for efficiency (e.g., 8 for tensor cores)")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./dealign_output",
                       help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16,
                       help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=25,
                       help="Warmup steps (ignored if warmup_ratio is set)")
    parser.add_argument("--warmup_ratio", type=float, default=None,
                       help="Fraction of total training steps for warmup (e.g., 0.1 for 10%%)")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=50,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=50,
                       help="Run evaluation every N steps (default: same as save_steps)")
    parser.add_argument("--evaluation_strategy", type=str, default="steps",
                       choices=["no", "steps", "epoch"],
                       help="Evaluation strategy")
    parser.add_argument("--save_total_limit", type=int, default=10,
                       help="Maximum number of checkpoints to keep")
    parser.add_argument("--fp16", action="store_true",
                       help="Use FP16 training")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="Use BF16 training")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                       help="Use gradient checkpointing")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    
    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true", default=True,
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="dealignment-training",
                       help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="Wandb entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Wandb run name (default: auto-generated)")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None,
                       help="Tags for wandb run")
    parser.add_argument("--wandb_mode", type=str, default="online",
                       choices=["online", "offline", "disabled"],
                       help="Wandb logging mode")
    
    args = parser.parse_args()
    
    # Determine quantization type
    # Default to 4-bit unless 8-bit is explicitly requested
    use_4bit = not args.use_8bit
    
    # Initialize wandb
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("wandb requested but not available. Install with: pip install wandb")
            logger.warning("Continuing without wandb logging...")
            args.use_wandb = False
        else:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                tags=args.wandb_tags,
                mode=args.wandb_mode,
                config={
                    # Model config
                    "model_name": args.model_name,
                    "use_4bit": use_4bit,
                    "use_8bit": args.use_8bit,
                    "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
                    "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
                    "use_nested_quant": args.use_nested_quant,
                    # LoRA config
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "lora_target_modules": args.lora_target_modules,
                    "bias": args.bias,
                    # Dataset config
                    "dataset_name": args.dataset_name,
                    "dataset_split": args.dataset_split,
                    "max_samples": args.max_samples,
                    "max_length": args.max_length,
                    # Training config
                    "num_train_epochs": args.num_train_epochs,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
                    "learning_rate": args.learning_rate,
                    "warmup_steps": args.warmup_steps,
                    "warmup_ratio": args.warmup_ratio,
                    "fp16": args.fp16,
                    "bf16": args.bf16,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "dataloader_num_workers": args.dataloader_num_workers,
                    "output_dir": args.output_dir,
                }
            )
            logger.info(f"Wandb initialized: project={args.wandb_project}, run={wandb.run.name if wandb.run else 'N/A'}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load full dataset (without max_samples limit) to extract validation from the end
    full_dataset = load_and_prepare_dataset(
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        max_samples=None,  # Don't limit yet - we need full dataset to extract validation
        offset=args.offset_samples
    )
    
    # Extract validation samples from the end of the full dataset
    eval_dataset = None
    if args.validation_samples > 0 and args.validation_samples < len(full_dataset):
        # Take last validation_samples from the full dataset
        eval_start_idx = len(full_dataset) - args.validation_samples
        eval_dataset = full_dataset.select(range(eval_start_idx, len(full_dataset)))
        logger.info(f"Extracted {len(eval_dataset)} validation samples from end of dataset (indices {eval_start_idx} to {len(full_dataset)-1})")
        
        # Now get training dataset: from start up to validation samples, then apply max_samples
        train_dataset = full_dataset.select(range(0, eval_start_idx))
        
        # Apply max_samples limit to training dataset if specified
        if args.max_samples and args.max_samples > 0:
            train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
            logger.info(f"Applied max_samples={args.max_samples} to training dataset")
        
        logger.info(f"Final split: {len(train_dataset)} train, {len(eval_dataset)} validation")
    else:
        # No validation or validation_samples too large - use full dataset for training
        train_dataset = full_dataset
        if args.validation_samples > 0:
            logger.warning(f"validation_samples ({args.validation_samples}) >= dataset size ({len(full_dataset)}), skipping validation")
        
        # Apply max_samples limit to training dataset if specified
        if args.max_samples and args.max_samples > 0:
            train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
            logger.info(f"Applied max_samples={args.max_samples} to training dataset")
    
    # Log dataset info to wandb
    if args.use_wandb:
        wandb.config.update({
            "full_dataset_size": len(full_dataset),
            "train_size": len(train_dataset),
            "validation_size": len(eval_dataset) if eval_dataset else 0,
            "validation_samples": args.validation_samples,
            "max_samples": args.max_samples,
        })
        logger.info(f"Dataset info logged to wandb: {len(train_dataset)} train, {len(eval_dataset) if eval_dataset else 0} validation (from full dataset of {len(full_dataset)} samples)")
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(
        model_name=args.model_name,
        use_4bit=use_4bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        use_nested_quant=args.use_nested_quant
    )
    
    # Setup LoRA
    model = setup_lora(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        bias=args.bias
    )
    
    # Log model info to wandb
    if args.use_wandb:
        # Get trainable parameters info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percentage = 100 * trainable_params / total_params if total_params > 0 else 0
        
        wandb.config.update({
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_percentage": trainable_percentage,
        })
        logger.info(f"Model info logged to wandb: {trainable_params:,} trainable / {total_params:,} total ({trainable_percentage:.2f}%)")

    # Tokenize datasets with proper loss masking
    logger.info("Tokenizing train dataset with loss masking...")
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset"
    )
    
    tokenized_eval_dataset = None
    if eval_dataset:
        logger.info("Tokenizing validation dataset with loss masking...")
        tokenized_eval_dataset = eval_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, args.max_length),
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing validation dataset"
        )
    
    logger.info("Tokenization complete. Labels are masked so only assistant tokens contribute to loss.")
    
    # Training arguments
    # Determine reporting backend
    # When report_to=["wandb"], Trainer automatically logs:
    # - train/loss (loss curve) at each logging_steps
    # - train/learning_rate (learning rate schedule)
    # - train/epoch (current epoch)
    # - train/total_flos (if available)
    # - eval/loss (validation loss) at each eval_steps (if evaluation enabled)
    if args.use_wandb:
        report_to = ["wandb"]
        logger.info(f"Wandb logging enabled: metrics will be logged every {args.logging_steps} steps")
    else:
        report_to = ["tensorboard"] if os.path.exists(os.path.join(args.output_dir, "runs")) else []
    
    # Determine warmup configuration: warmup_ratio takes precedence over warmup_steps
    warmup_kwargs = {}
    if args.warmup_ratio is not None:
        warmup_kwargs['warmup_ratio'] = args.warmup_ratio
        logger.info(f"Using warmup_ratio: {args.warmup_ratio} (fraction of total steps)")
    elif args.warmup_steps is not None:
        warmup_kwargs['warmup_steps'] = args.warmup_steps
        logger.info(f"Using warmup_steps: {args.warmup_steps}")
    else:
        # Default to 100 steps if neither is specified
        warmup_kwargs['warmup_steps'] = 100
        logger.info("Using default warmup_steps: 100")
    
    # Determine evaluation configuration
    eval_kwargs = {}
    if tokenized_eval_dataset is not None:
        eval_kwargs['eval_strategy'] = args.evaluation_strategy  # Note: parameter name is 'eval_strategy' not 'evaluation_strategy'
        eval_kwargs['eval_steps'] = args.eval_steps if args.eval_steps is not None else args.save_steps
        eval_kwargs['per_device_eval_batch_size'] = args.per_device_train_batch_size  # Use same batch size for eval
        logger.info(f"Evaluation enabled: strategy={args.evaluation_strategy}, eval_steps={eval_kwargs['eval_steps']}")
    else:
        eval_kwargs['eval_strategy'] = "no"
        logger.info("Evaluation disabled (no validation dataset)")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        **warmup_kwargs,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=report_to,
        load_best_model_at_end=False,
        **eval_kwargs,
    )
    
    # Custom data collator with dynamic padding
    # This will pad sequences to the longest sequence in each batch (not to max_length)
    # pad_to_multiple_of optimizes for tensor cores/GPU efficiency
    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=args.pad_to_multiple_of if args.pad_to_multiple_of > 0 else None,
    )
    logger.info(f"Using custom data collator with dynamic padding (pad_to_multiple_of={args.pad_to_multiple_of if args.pad_to_multiple_of > 0 else 'None'})")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    try:
        train_result = trainer.train()
        
        # Log final training metrics to wandb
        if args.use_wandb:
            if hasattr(train_result, 'metrics'):
                wandb.log(train_result.metrics)
            wandb.log({
                "train/global_step": trainer.state.global_step,
                "train/epoch": trainer.state.epoch,
            })
            
            # Run final evaluation if validation dataset exists
            if tokenized_eval_dataset is not None:
                logger.info("Running final evaluation...")
                eval_result = trainer.evaluate()
                if eval_result:
                    wandb.log(eval_result)
                    logger.info(f"Final evaluation metrics logged to wandb: {eval_result}")
            
            logger.info("Training metrics logged to wandb")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.use_wandb:
            wandb.log({"train/status": "failed", "train/error": str(e)})
        raise
    finally:
        # Save final model
        logger.info(f"Saving final model to {args.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Save LoRA adapters separately
        model.save_pretrained(os.path.join(args.output_dir, "lora_adapters"))
        
        # Log model save path to wandb
        if args.use_wandb:
            wandb.log({
                "model/save_path": args.output_dir,
                "train/status": "completed"
            })
            wandb.finish()
            logger.info("Wandb run finished")
        
        logger.info("Training complete!")


if __name__ == "__main__":
    main()