from flask import Flask, request, jsonify
from flask_cors import CORS
import config
from preprocessing import preprocess_text
from inference import generate_summary, correct_grammar, paraphrase_text

app = Flask(__name__)
CORS(app)

@app.route('/process', methods=['POST'])
def process_text():
    data = request.json or {}
    raw_text = data.get('text', '')
    mode = data.get('mode', 'SUMMARY').upper()

    clean_text = preprocess_text(raw_text)

    if mode == 'SUMMARY':
        output = generate_summary(clean_text)
    elif mode in ['GRAMMAR CORRECTION', 'GRAMMER CORRECTION']:
        output = correct_grammar(clean_text)
    elif mode == 'PARAPHRASE':
        output = paraphrase_text(clean_text)
    else:
        output = "Invalid mode selected."

    return jsonify({"output": output})


if __name__ == '__main__':
    app.run(debug=config.DEBUG, use_reloader=False, port=config.PORT)