from dataclasses import dataclass


@dataclass
class Config:
    # Base model for the demo
    # Recommended default for easier setup:
    BASE_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # Where the LoRA adapter will be saved
    OUTPUT_DIR: str = "outputs/mistral-industry-lora"

    # Dataset location
    DATA_RAW_PATH: str = "data/raw/industry_dataset.jsonl"

    # Training params (kept small for demo; increase for real training)
    MAX_SEQ_LENGTH: int = 512
    NUM_EPOCHS: int = 3
    LEARNING_RATE: float = 2e-4
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.03
    LR_SCHEDULER_TYPE: str = "cosine"

    PER_DEVICE_TRAIN_BATCH_SIZE: int = 1
    GRADIENT_ACCUMULATION_STEPS: int = 8

    # LoRA params
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05

    # BitsAndBytes quantization
    LOAD_IN_4BIT: bool = True
    BNB_4BIT_COMPUTE_DTYPE: str = "float16"  # float16 or bfloat16

    SEED: int = 42


cfg = Config()

