from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer('all-mpnet-base-v2')

@app.route('/embed', methods=['POST'])
def embed_text():
    try:
        data = request.get_json()
        if not data or 'sentences' not in data:
            return jsonify({"error": "No sentences provided"}), 400

        input_sentences = data['sentences']

        return jsonify({"test": input_sentences})

#         if not isinstance(input_sentences, dict):
#             return jsonify({"error": "Sentences must be provided as a dictionary with product IDs as keys"}), 400

        # Process each product's sentences
        embedded_products = {}
        for product_id, product in input_sentences.items():
            if not isinstance(product, dict):
                return jsonify({"error": "Each product's sentences must be a dictionary"}), 400

            embedded_product = {}
            for key, sentence in product.items():
                # Encode the sentence into embeddings and convert to list
                embedding = model.encode(sentence, show_progress_bar=False).tolist()
                embedded_product[key] = embedding

            embedded_products[product_id] = embedded_product

        return jsonify(embedded_products)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
