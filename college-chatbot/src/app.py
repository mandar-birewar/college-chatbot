from flask import Flask, request, jsonify
from model import ChatBotModel
from utils import load_config

app = Flask(__name__)
config = load_config()

chatbot_model = ChatBotModel(config['model_path'])

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = chatbot_model.generate_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config['port'])