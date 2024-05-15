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
        # Get JSON data from request
        data = request.get_json()
        department_sentences = data.get('sentences')
        if not department_sentences:
            return jsonify({"error": "No sentences provided"}), 400

        # Check if the input is a dictionary as expected
        if not isinstance(department_sentences, dict):
            return jsonify({"error": "Sentences must be provided as a dictionary with department codes as keys"}), 400

        # Prepare for embedding
        embedded_sentences = {}
        for dept_code, sentence in department_sentences.items():
            # Generate embeddings for each sentence, assuming each entry is a single string
            embedding = model.encode(sentence, show_progress_bar=False)
            # Convert embedding to list for JSON serialization
            embedded_sentences[dept_code] = embedding.tolist()

        return jsonify({"embeddings": embedded_sentences})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
