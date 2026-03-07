# train.py – full 7B training (optimized for RTX 3090 24GB)
import os
# Reduce CUDA fragmentation (run before importing torch)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # PyTorch 2.0+

import torch
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from data_loader import StreamingDataset
from config import *

def main():
    # 0. GPU check – bf16 requires CUDA; fall back to fp16 if PyTorch is CPU-only
    cuda_ok = torch.cuda.is_available()
    bf16_ok = cuda_ok and torch.cuda.is_bf16_supported()
    use_bf16_actual = use_bf16 and bf16_ok
    if not cuda_ok:
        raise RuntimeError(
            "No GPU detected. PyTorch is likely CPU-only. On Vast.ai, use an image with CUDA. "
            "Or reinstall: pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
    if use_bf16 and not bf16_ok:
        print("WARNING: bf16 not supported on this GPU. Using fp16.")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load tokenizer (use_fast=False required for OpenLLaMA/LLaMA SentencePiece)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=tokenizer_use_fast)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Create streaming dataset (PyTorch IterableDataset)
    train_dataset = StreamingDataset(
        dataset_name=dataset_name,
        tokenizer_name=model_name,
        buffer_size=streaming_buffer_size,
        block_size=block_size,
        tokenizer_use_fast=tokenizer_use_fast,
    )

    # 3. Load model – random init, bf16 from start to fit 24GB
    torch.cuda.empty_cache()
    config = LlamaConfig.from_pretrained(model_name)
    model_kwargs = {"config": config}
    if use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    dtype = torch.bfloat16 if use_bf16_actual else torch.float16
    model = LlamaForCausalLM(**model_kwargs).to(dtype)

    # Enable gradient checkpointing to save VRAM (critical for 7B on 24GB)
    model.gradient_checkpointing_enable()

    # 4. Training arguments (tuned for RTX 3090)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        bf16=use_bf16_actual,               # Ampere (3090) – better than fp16
        fp16=not use_bf16_actual,            # Fallback when bf16 not supported
        remove_unused_columns=False,
        report_to=report_to,
        save_total_limit=save_total_limit,
        gradient_checkpointing=True,         # Extra safety (model-level also set)
        dataloader_num_workers=dataloader_num_workers,
        max_grad_norm=max_grad_norm,
        optim="adamw_bnb_8bit" if use_8bit_optimizer else "adamw_torch",
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )

    # 6. Start training
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
