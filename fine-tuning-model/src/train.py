import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments)
from trl import SFTTrainer

from config import cfg
from dataset import build_hf_dataset


def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Dataset
    ds = build_hf_dataset(cfg.DATA_RAW_PATH)

    # Model + tokenizer
    if cfg.LOAD_IN_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, cfg.BNB_4BIT_COMPUTE_DTYPE),
        )
    else:
        bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL, use_fast=True)

    # Some instruct models need padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Prepare LoRA
    peft_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # TRL SFTTrainer expects a single text field (or we can supply formatting_func)
    def formatting_func(example):
        # For chat-style templates, easiest is to create a single string with clear delimiters.
        # The model will learn from these patterns.
        return (
            f"<system>{example['system']}</system>\n"
            f"<user>{example['user']}</user>\n"
            f"<assistant>{example['assistant']}</assistant>"
        )

    training_args = TrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        num_train_epochs=cfg.NUM_EPOCHS,
        learning_rate=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        warmup_ratio=cfg.WARMUP_RATIO,
        lr_scheduler_type=cfg.LR_SCHEDULER_TYPE,
        per_device_train_batch_size=cfg.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        fp16=(cfg.BNB_4BIT_COMPUTE_DTYPE == "float16"),
        bf16=(cfg.BNB_4BIT_COMPUTE_DTYPE == "bfloat16"),
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit" if cfg.LOAD_IN_4BIT else "adamw_torch",
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        peft_config=peft_config,
        formatting_func=formatting_func,
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        args=training_args,
    )

    trainer.train()

    # Save adapter + tokenizer
    trainer.model.save_pretrained(cfg.OUTPUT_DIR)
    tokenizer.save_pretrained(cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()

