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
        sentences = data.get('sentences')
        if not sentences:
            return jsonify({"error": "No sentences provided"}), 400

        # Generate embeddings
        embeddings = model.encode(sentences, show_progress_bar=False)

        # Convert embeddings to list for JSON serialization
        embeddings_list = [embedding.tolist() for embedding in embeddings]

        return jsonify({"embeddings": embeddings_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
