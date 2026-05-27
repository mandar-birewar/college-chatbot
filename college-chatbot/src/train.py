import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def preprocess_data(data):
    texts = [item['text'] for item in data]
    return texts

def fine_tune_model(train_texts, model_name='gpt2', output_dir='./model_output'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')

    train_dataset = CustomDataset(train_encodings)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

class CustomDataset:
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

if __name__ == "__main__":
    raw_data_path = '../data/raw/training_data.jsonl'
    data = load_data(raw_data_path)
    texts = preprocess_data(data)
    
    train_texts, val_texts = train_test_split(texts, test_size=0.1)
    
    fine_tune_model(train_texts)