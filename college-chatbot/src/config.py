# Configuration settings for the college chatbot project

class Config:
    # File paths
    RAW_DATA_PATH = 'data/raw/training_data.jsonl'
    PROCESSED_DATA_PATH = 'data/processed/dataset.csv'
    
    # Model parameters
    MODEL_NAME = 'gpt-2'
    MAX_LENGTH = 100
    NUM_RETURN_SEQUENCES = 1
    
    # Training hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 5e-5
    
    # Other settings
    SEED = 42
    LOGGING_DIR = 'logs/'