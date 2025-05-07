from document_embedder import DocumentEmbedder
from flask import Flask, request, jsonify
import os

PORT = int(os.getenv('PORT', 5003))

app = Flask(__name__)


@app.route('/embed', methods=['POST'])
def embed_request():
    try:
        data = request.get_json()
        text = data['text']
        embedder = DocumentEmbedder()
        embedding = embedder.embed_texts([text])[0]
        return jsonify({
            'embedding': embedding
        })
    except Exception as e:
        print(f"Error embedding the text: {e}")
        return jsonify({"error": "Failed to get response from LLM service"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)
    print(f"Server is running on port {PORT}")
