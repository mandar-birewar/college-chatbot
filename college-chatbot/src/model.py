class ChatBotModel:
    def __init__(self, model_name, tokenizer_name):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()

    def load_model(self):
        # Load the pre-trained model
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(self.model_name)

    def load_tokenizer(self):
        # Load the tokenizer
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.tokenizer_name)

    def generate_response(self, input_text, max_length=50):
        # Generate a response from the model
        inputs = self.tokenizer.encode(input_text, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def save_model(self, save_directory):
        # Save the model and tokenizer
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def load_saved_model(self, save_directory):
        # Load a saved model and tokenizer
        self.model = self.load_model(save_directory)
        self.tokenizer = self.load_tokenizer(save_directory)