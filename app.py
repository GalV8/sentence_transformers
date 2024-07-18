from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer('all-mpnet-base-v2')

@app.route('/')
def index():
    return 'Hello, Flask!'

@app.route('/embed', methods=['POST'])
def embed_text():
    try:
        data = request.get_json()

        if not data or 'sentences' not in data:
            return jsonify({"error": "No sentences provided"}), 400

        department_sentences_dict = data['sentences']

        if not isinstance(department_sentences_dict, dict):
            return jsonify({"error": "Sentences must be provided as a dictionary with product IDs as keys"}), 400

        embedded_results = {}
        for product_id, department_sentences in department_sentences_dict.items():
            if not isinstance(department_sentences, dict):
                return jsonify({"error": "Each product's sentences must be a dictionary with department codes as keys"}), 400

            embedded_sentences = {}
            for dept_code, sentence in department_sentences.items():
                embedding = model.encode(sentence, show_progress_bar=False)
                embedded_sentences[dept_code] = embedding.tolist()

            embedded_results[product_id] = embedded_sentences

        return jsonify({"embeddings": embedded_results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
