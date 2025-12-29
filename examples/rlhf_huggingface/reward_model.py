
import os
import glob
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_peft_config,
    setup_chat_format,
)

# Auto-detect CPU cores for dataloader workers
NUM_WORKERS = min(os.cpu_count() or 4, 8)  # Use CPU cores, max 8 to avoid overhead


# Device selection with MPS support
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"


device = get_device()
print(f"Using device: {device}")
print(f"DataLoader workers: {NUM_WORKERS} (CPU cores: {os.cpu_count()})")

# Auto-detect BF16 support (Ampere+ only)
use_bf16 = False

if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability()
    
    # BF16 requires Ampere or newer (compute capability >= 8.0)
    if compute_capability[0] >= 8:
        use_bf16 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"BF16 + TF32 enabled (GPU: compute {compute_capability[0]}.{compute_capability[1]})")
    else:
        print(f"Using FP32 (GPU: compute {compute_capability[0]}.{compute_capability[1]}, BF16 requires >= 8.0)")


if __name__ == "__main__":
    # Initialize arguments directly (no command-line parsing needed)
    script_args = ScriptArguments(
        dataset_name="PKU-Alignment/PKU-SafeRLHF-30K",
        dataset_train_split="train",
        dataset_test_split="test",
    )

    training_args = RewardConfig(
        output_dir="reward_model",
        # For 360M model (current):
        per_device_train_batch_size=64,  # Reduced for larger model
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,   # Effective batch = 16*4 = 64
        # For 135M model (original):
        # per_device_train_batch_size=32,
        # per_device_eval_batch_size=32,
        # gradient_accumulation_steps=2,  # Effective batch = 32*2 = 64
        num_train_epochs=6,  # Increased from 1 - more epochs needed with filtered data
        learning_rate=5.0e-5,  # Increased from 1e-5 for faster convergence
        lr_scheduler_type="cosine",  # Cosine annealing - decays faster than linear (default)
        warmup_ratio=0.1,  # Added warmup for stability
        eval_strategy="steps",
        eval_steps=10,  # Evaluate every 10 steps for frequent validation
        logging_steps=1,  # Log every single step for detailed TensorBoard visualization
        max_length=512,
        save_strategy="steps",  # Changed to "steps" to match eval_strategy (required by load_best_model_at_end)
        save_steps=10,  # Save every 10 steps, aligned with eval_steps
        save_total_limit=5,  # Keep best 5 checkpoints
        load_best_model_at_end=True,  # Load best model at end based on eval metric
        metric_for_best_model="eval_loss",  # Use eval loss to determine best model
        bf16=use_bf16,  # Auto-detected: True on L4/A100, False on T4 (falls back to FP32)
        dataloader_num_workers=NUM_WORKERS,  # Auto-detected from CPU cores
        dataloader_prefetch_factor=2,
        dataloader_pin_memory=True,
    )

    model_args = ModelConfig(
        # Current model (360M - larger capacity):
        model_name_or_path="HuggingFaceTB/SmolLM2-360M-Instruct",
        # Original model (135M - faster training):
        # model_name_or_path="HuggingFaceTB/SmolLM2-135M-Instruct",
        use_peft=True,  # Enable LoRA training
        lora_task_type="SEQ_CLS",
        lora_r=32,  # AGGRESSIVE: 4x increase from 8 for maximum learning capacity (~17M trainable params, 4.7%)
        lora_alpha=64,  # AGGRESSIVE: 4x increase from 16 (alpha = 2 * r is recommended)
        lora_target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Enable TensorBoard logging
    training_args.report_to = ["tensorboard"]
    training_args.logging_dir = f"{training_args.output_dir}/logs"

    ################
    # Model & Tokenizer
    ################
    dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        dtype=dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        trust_remote_code=True,
        **model_kwargs,
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    ##############
    # Load dataset
    ##############
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Filter dataset to keep only clear safe vs unsafe pairs
    def filter_clear_safety_pairs(sample):
        """
        Only keep samples where one response is safe and the other is unsafe.
        This gives us high-quality training data for absolute safety judgments.
        """
        if "safer_response_id" not in sample:
            return True  # Keep non-PKU-SafeRLHF data as-is
        
        safer_id = int(sample["safer_response_id"])
        safer_safe = sample.get(f"is_response_{safer_id}_safe", None)
        other_safe = sample.get(f"is_response_{1-safer_id}_safe", None)
        
        # Only keep if one is safe and the other is unsafe
        if safer_safe is not None and other_safe is not None:
            return safer_safe != other_safe
        return True  # Keep if safety labels are missing
    
    print("=" * 80)
    print("Filtering dataset to keep only clear safe vs unsafe pairs...")
    print("=" * 80)
    original_train_size = len(dataset[script_args.dataset_train_split])
    dataset[script_args.dataset_train_split] = dataset[script_args.dataset_train_split].filter(
        filter_clear_safety_pairs, 
        num_proc=8
    )
    filtered_train_size = len(dataset[script_args.dataset_train_split])
    print(f"Train set: {original_train_size} → {filtered_train_size} samples ({100*filtered_train_size/original_train_size:.1f}% kept)")
    
    if script_args.dataset_test_split in dataset and script_args.dataset_test_split != script_args.dataset_train_split:
        original_test_size = len(dataset[script_args.dataset_test_split])
        dataset[script_args.dataset_test_split] = dataset[script_args.dataset_test_split].filter(
            filter_clear_safety_pairs, 
            num_proc=8
        )
        filtered_test_size = len(dataset[script_args.dataset_test_split])
        print(f"Test set: {original_test_size} → {filtered_test_size} samples ({100*filtered_test_size/original_test_size:.1f}% kept)")
    
    print(f"✓ Filtered to keep only samples with clear safe vs unsafe distinction")
    print("=" * 80)
    print()

    # Format dataset for PKU-SafeRLHF compatibility
    def format_dataset_if_needed(dataset):
        """Convert PKU-SafeRLHF format to TRL format with chat template"""
        # Check if dataset already has the required format
        sample = dataset[0]
        if "chosen" in sample and "rejected" in sample:
            # Already formatted, but need to apply chat template
            print("Dataset has chosen/rejected format. Applying chat template...")

            def apply_chat_template_to_formatted(sample):
                # Apply chat template to already formatted data
                prompt = sample.get("prompt", "")

                chosen_text = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": sample["chosen"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )

                rejected_text = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": sample["rejected"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )

                return {"chosen": chosen_text, "rejected": rejected_text}

            return dataset.map(apply_chat_template_to_formatted, num_proc=8)

        print(
            "Detected PKU-SafeRLHF format. Converting to TRL format with chat template..."
        )

        def format_sample(sample):
            # Get prompt
            prompt = sample.get("prompt", sample.get("question", ""))

            # Determine which response is chosen based on safety
            if "safer_response_id" in sample:
                safer_id = sample["safer_response_id"]
                chosen_response = sample[f"response_{safer_id}"]
                rejected_response = sample[f"response_{1-safer_id}"]
            elif "better_response_id" in sample:
                better_id = sample["better_response_id"]
                chosen_response = sample[f"response_{better_id}"]
                rejected_response = sample[f"response_{1-better_id}"]
            else:
                # Fallback for other formats
                chosen_response = sample.get("response_0", sample.get("chosen", ""))
                rejected_response = sample.get("response_1", sample.get("rejected", ""))

            # Apply chat template to create full conversation
            chosen_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen_response},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )

            rejected_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rejected_response},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )

            return {"chosen": chosen_text, "rejected": rejected_text}

        # Apply formatting with chat template
        dataset = dataset.map(format_sample, num_proc=8)  # Parallel processing!

        # Remove unnecessary columns, keep only required ones
        cols_to_keep = ["chosen", "rejected"]
        cols_to_remove = [c for c in dataset.column_names if c not in cols_to_keep]
        if cols_to_remove:
            dataset = dataset.remove_columns(cols_to_remove)

        return dataset

    # Apply format conversion if needed
    if script_args.dataset_train_split in dataset:
        dataset[script_args.dataset_train_split] = format_dataset_if_needed(
            dataset[script_args.dataset_train_split]
        )

    if (
        script_args.dataset_test_split in dataset
        and script_args.dataset_test_split != script_args.dataset_train_split
    ):
        dataset[script_args.dataset_test_split] = format_dataset_if_needed(
            dataset[script_args.dataset_test_split]
        )

    ##########
    # Training
    ##########
    peft_config = get_peft_config(model_args)

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        peft_config=peft_config,
    )

    # Print actual trainable parameters after LoRA is applied
    actual_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    actual_total = sum(p.numel() for p in trainer.model.parameters())
    
    print(f"\n{'='*60}")
    print(f"Training Mode: LoRA + TRAINABLE CLS HEAD")
    print(f"Base Model: {model_args.model_name_or_path}")
    print(f"LoRA Config: r={model_args.lora_r}, alpha={model_args.lora_alpha}")
    print(f"Classification head: trainable (modules_to_save)")
    print(f"Total parameters: {actual_total:,}")
    print(f"Trainable parameters: {actual_trainable:,} ({100 * actual_trainable / actual_total:.2f}%)")
    print(f"Frozen parameters: {actual_total - actual_trainable:,}")
    print(f"{'='*60}\n")

    print(f"Starting LoRA + Head training...\n")

    # Check for existing checkpoints to resume from
    checkpoints = sorted(glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")))
    resume_from_checkpoint = None
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        resume_from_checkpoint = latest_checkpoint
        print(f"Found existing checkpoint: {latest_checkpoint}")
        print(f"Automatically resuming training from checkpoint...\n")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    ##########################
    # Save model
    ##########################
    print(f"\nSaving model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"Model saved successfully")

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
