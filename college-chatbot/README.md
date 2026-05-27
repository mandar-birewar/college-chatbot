# College ChatBot

This project is a ChatBot application designed for college students. It utilizes a fine-tuned text generation model to provide interactive and informative responses to user queries related to college life, academics, and other relevant topics.

## Project Structure

```
college-chatbot
├── data
│   ├── raw
│   │   └── training_data.jsonl
│   └── processed
│       └── dataset.csv
├── src
│   ├── app.py
│   ├── train.py
│   ├── model.py
│   ├── utils.py
│   └── config.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/college-chatbot.git
   cd college-chatbot
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your training data in `data/raw/training_data.jsonl`. Each line should be a valid JSON object.
2. Run the training script to fine-tune the model:
   ```
   python src/train.py
   ```

3. Start the ChatBot application:
   ```
   python src/app.py
   ```

4. Interact with the ChatBot through the web interface.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.