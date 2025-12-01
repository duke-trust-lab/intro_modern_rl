"""
PPO Training for RLHF - Usin    # Configuration
    policy_model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    reward_model_path = os.getenv("REWARD_MODEL_PATH", "reward_model")
    output_dir = os.getenv("ALIGNED_MODEL_DIR", "aligned_model")L 0.23.0 with LoRA
"""

import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from trl import PPOConfig, PPOTrainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
import warnings
warnings.filterwarnings("ignore")


def main():
    print("Starting PPO Training with TRL 0.23.0 + LoRA...")

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

    # Configuration
    # Current model (360M - better alignment quality):
    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    # Original model (135M - faster training):
    # model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    reward_model_path = "reward_model"
    output_dir = "aligned_model"

    print(f"Policy Model: {model_name}")
    print(f"Reward Model: {reward_model_path}")
    print(f"Training Strategy: LoRA Fine-tuning")

    # LoRA Configuration for policy model - AGGRESSIVE: Match reward model capacity
    lora_config = LoraConfig(
        r=32,  # AGGRESSIVE: Increased from 8 to 32 (4x) to match reward model
        lora_alpha=64,  # AGGRESSIVE: Increased from 16 to 64 (4x) to match reward model
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # SmolLM2/Llama modules
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # PPO Configuration for TRL 0.23.0
    ppo_config = PPOConfig(
        exp_name="ppo_safety_alignment_smollm2_360m",  # Updated experiment name
        seed=42,
        
        # Training episodes - ALIGNED with reward model training scale
        total_episodes=3000,  # ~6 epochs × 500 samples ≈ 3000 episodes (aligned with RM epochs=6)
        
        # Batch settings - REDUCED to prevent OOM with longer sequences
        batch_size=128,                      # Reduced from 256 to 128 (OOM fix)
        per_device_train_batch_size=2,      # Reduced from 8 to 2 (OOM fix with 512 length)
        gradient_accumulation_steps=16,     # Increased from 8 to 16, effective batch = 2*16 = 32
        
        # Generation settings - REDUCED to prevent OOM
        response_length=512,  # BALANCED: Increased from 512 to 512 (compromise between RM alignment and memory)
        temperature=0.001,
        
        # PPO hyperparameters - OPTIMIZED for stability
        num_ppo_epochs=5,        # Increased from 1 to 2 (learn more from each batch)
        learning_rate=3e-6,      # Reduced from 5e-6 to 3e-6 (smaller updates)
        cliprange=0.1,           # Reduced from 0.15 to 0.1 (more conservative)
        cliprange_value=0.1,     # Reduced from 0.2 to 0.1 (more conservative)
        kl_coef=0.1,            # Increased from default 0.05 to 0.1 (stronger KL penalty)
        vf_coef=0.1,
        gamma=1.0,
        lam=0.95,
        
        # Logging
        output_dir=output_dir,
        logging_steps=10,
        save_steps=10,  # Save every 500 episodes
        report_to="tensorboard",  # Enable TensorBoard logging
        logging_dir=f"{output_dir}/logs",  # TensorBoard logs in model directory
        
        # Optimization
        bf16=use_bf16,  # Auto-detected: True on L4/A100, False on T4 (falls back to FP32)
        
        # DataLoader optimization - PPO doesn't need workers (on-policy generation)
        dataloader_num_workers=0,      # Changed from 4 to 0 (no prefetching needed)
        dataloader_pin_memory=True,
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Use eos_token as pad_token (SmolLM specific workaround)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load policy model (standard causal LM, no value head needed for new PPOTrainer)
    print("Loading policy model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        # Don't set device_map - Accelerator will handle device placement
    )
    # Set pad_token_id
    policy_model.generation_config.pad_token_id = tokenizer.pad_token_id
    policy_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Apply LoRA to policy model
    print("Applying LoRA to policy model...")
    policy_model = get_peft_model(policy_model, lora_config)
    policy_model.print_trainable_parameters()  # Show trainable parameter count

    # Load reference model (frozen copy of policy)
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    ref_model.generation_config.pad_token_id = tokenizer.pad_token_id
    ref_model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.eval()
    print("Reference model loaded successfully")

    # Load reward model (LoRA adapter)
    # Simplified pattern: directly load base model + adapter
    # Same as PEFT examples: https://github.com/huggingface/peft/tree/main/examples
    print(f"Loading reward model from {reward_model_path}...")
    
    # Auto-detect base model from adapter_config.json
    reward_config = PeftConfig.from_pretrained(reward_model_path)
    base_model_name = reward_config.base_model_name_or_path
    
    reward_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    
    print(f"  Base model: {base_model_name}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        torch_dtype=torch.float16,
    )
    
    print(f"  Loading LoRA adapter...")
    reward_model = PeftModel.from_pretrained(reward_model, reward_model_path)
    
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
    reward_model.eval()  # Set to inference mode
    print("Reward model loaded successfully")

    # Prepare dataset
    print("Loading dataset...")
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split="train")
    # dataset = dataset.select(range(min(1000, len(dataset))))  # Increased from 20 to 1000 for better training

    def tokenize_dataset(batch):
        """Tokenize dataset for PPO training with chat template for Instruct model
        
        Following official TRL examples (examples/scripts/ppo/ppo_tldr.py):
        - Apply chat template with tokenize=True (default) to directly get input_ids
        - No need for separate tokenization step
        """
        # batch is a dict with keys like 'prompt', 'question', etc.
        prompts = batch.get('prompt', batch.get('question', []))
        
        # Apply chat template and tokenize in one step (official TRL pattern)
        input_ids_list = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                truncation=True,
                max_length=128,
                padding=False,
                add_generation_prompt=True  # Add assistant prompt marker
                # tokenize=True is the default - returns token IDs directly
            )
            for prompt in prompts
        ]
        
        return {"input_ids": input_ids_list}

    train_dataset = dataset.map(
        tokenize_dataset,
        batched=True,  # Use batched processing
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )

    # Use a random subset as eval dataset for better diversity
    # Random sampling ensures we see different queries across training
    import random
    random.seed(42)  # For reproducibility
    eval_size = min(100, len(train_dataset))  # Increased to 100 for better coverage
    eval_indices = random.sample(range(len(train_dataset)), eval_size)
    eval_dataset = train_dataset.select(eval_indices)

    print(f"Dataset size: {len(train_dataset)}, Eval size: {len(eval_dataset)} (randomly sampled)")


    # Load value model - independent model (same structure as reward_model)
    # In TRL 0.23.0, value_model is a separate AutoModelForSequenceClassification
    print("Loading value model (independent critic)...")
    
    # Use same base model as reward model
    value_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,  # Reuse base model name from reward model loading
        num_labels=1,
        torch_dtype=torch.float16,
    )
    
    value_model = PeftModel.from_pretrained(value_model, reward_model_path)
    
    value_model.config.pad_token_id = reward_tokenizer.pad_token_id
    print("Value model loaded successfully")

    # Initialize PPO trainer
    print("Initializing PPO trainer...")
    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=value_model,  # Independent AutoModelForSequenceClassification
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Start training - TRL 0.23.0 handles everything internally
    print("\nStarting PPO training loop with LoRA...")
    print("=" * 60)
    trainer.train()
    print("=" * 60)

    # Save the aligned model
    # PPOTrainer.save_model() internally handles:
    # 1. Extracting policy from PolicyAndValueWrapper
    # 2. Saving only the policy (not value_model)
    # Note: This saves the LoRA adapter, not merged weights
    print(f"\nSaving trained LoRA adapter to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("PPO training completed successfully!")
    print(f"LoRA adapter saved to: {output_dir}")
    print(f"To use the model: PeftModel.from_pretrained(base_model, '{output_dir}')")


if __name__ == "__main__":
    main()
