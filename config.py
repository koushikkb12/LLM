# config.py – full training
# Use open_llama (no HF login) – 3B fits 24GB; 7B needs 40GB+
model_name = "openlm-research/open_llama_3b_v2"   # 3B – fits 24GB
# model_name = "openlm-research/open_llama_7b_v2"  # 7B – needs 40GB+ VRAM
# model_name = "meta-llama/Llama-2-7b-hf"          # Uncomment if you have HF access
tokenizer_use_fast = False   # Required for OpenLLaMA (fast tokenizer misparses tokenizer.model)

dataset_name = "mlfoundations/dclm-baseline-1.0"
streaming_buffer_size = 100_000   # Larger buffer for better shuffling
max_steps = 1_000_000             # Adjust based on your compute budget (full training ~3.7k H100 hours)
batch_size = 1                    # Will use gradient accumulation to simulate larger batch
gradient_accumulation_steps = 8   # Effective batch size = 8
block_size = 1024                 # 3B fits 1024 on 24GB; 7B needs block_size 512 + 40GB

# Training args
output_dir = "./3B_training"
logging_steps = 10
save_steps = 1000                  # Save every 1000 steps
learning_rate = 3e-4               # Common for 7B from scratch
warmup_steps = 2000

# RTX 3090 (24GB VRAM) optimizations
use_bf16 = True                   # bf16 on Ampere (3090) – better than fp16
use_flash_attention_2 = False     # Set True if you install: pip install flash-attn --no-build-isolation
use_8bit_optimizer = True         # AdamW 8-bit – saves ~20GB (required for 7B on 24GB)
dataloader_num_workers = 0        # 0 = less RAM; use 2–4 if you have spare CPU RAM
max_grad_norm = 1.0               # Gradient clipping for stability
save_total_limit = 3               # Keep last 3 checkpoints to save disk
report_to = "none"                # "wandb" for experiment tracking (requires: wandb login)
