from flask import Flask, request, jsonify
from flask_cors import CORS
import config
from preprocessing import preprocess_text
from inference import generate_summary, correct_grammar, paraphrase_text

app = Flask(__name__)
CORS(app)

TASK_HANDLERS = {
    'SUMMARY': generate_summary,
    'GRAMMAR CORRECTION': correct_grammar,
    'GRAMMER CORRECTION': correct_grammar,
    'PARAPHRASE': paraphrase_text
}

@app.route('/process', methods=['POST'])
def process_text():
    data = request.get_json(silent=True) or {}
    raw_text = data.get('text', '')
    mode = data.get('mode', 'SUMMARY').upper().strip()

    if not raw_text.strip():
        return jsonify({"output": "Input text cannot be empty."}), 400

    handler = TASK_HANDLERS.get(mode)
    if not handler:
        return jsonify({"output": f"Invalid mode. Supported modes: {list(TASK_HANDLERS.keys())}"}), 400

    clean_text = preprocess_text(raw_text)


    output = handler(clean_text)

    return jsonify({
        "preprocessed_text": clean_text,
        "output": output
    })


if __name__ == '__main__':
    app.run(debug=config.DEBUG, use_reloader=False, port=config.PORT)