"""VLM fine-tuning with TRL + PEFT LoRA.

Based on: docs.liquid.ai/customization/finetuning-frameworks/trl
Model: LiquidAI/LFM2.5-VL-450M (HuggingFace safetensors format)
This module is intended for partner's RTX 4060 Ti or Colab T4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer


def load_base_model(
    model_name: str = "LiquidAI/LFM2.5-VL-450M",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Any, Any]:
    """Load the base LFM2.5-VL model and processor.

    Args:
        model_name: HuggingFace model identifier.
        dtype: Model dtype (bfloat16 recommended).

    Returns:
        Tuple of (model, processor).
    """
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> LoraConfig:
    """Create LoRA configuration for fine-tuning.

    Args:
        r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_dropout: Dropout probability.

    Returns:
        LoraConfig object.
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )


def get_training_config(
    output_dir: str = "./checkpoints/sentinel-mind",
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
) -> SFTConfig:
    """Create training configuration.

    Args:
        output_dir: Checkpoint output directory.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Learning rate.

    Returns:
        SFTConfig object.
    """
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        dataset_kwargs={"skip_prepare_dataset": False},
    )


def train(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    model_name: str = "LiquidAI/LFM2.5-VL-450M",
    output_dir: str = "./checkpoints/sentinel-mind",
    num_epochs: int = 3,
    batch_size: int = 4,
) -> None:
    """Run LoRA fine-tuning.

    Args:
        train_dataset: Training dataset in TRL conversation format.
        val_dataset: Optional validation dataset.
        model_name: Base model HuggingFace ID.
        output_dir: Output directory for checkpoints.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
    """
    model, processor = load_base_model(model_name)
    lora_config = get_lora_config()
    training_config = get_training_config(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(f"{output_dir}-best")
