import gc
from flask import Flask, request, jsonify
import os

from document_embedder import DocumentEmbedder
from Retriever.retrieve import HierarchicalSearchEngine

PORT = int(os.getenv('PORT', 5004))

app = Flask(__name__)



def reranker(user_query: str, search_results) -> str:
    return "heyy there"


def retrieve(user_query: str, embedding: list[float]) -> list[str]:
    searcher = HierarchicalSearchEngine()
    search_results = searcher.search(embedding)
    return [reranker(user_query, search_results)]


@app.route('/embed', methods=['POST'])
def embed_request():
    try:
        data = request.get_json()
        texts = data['texts']
        embedder = DocumentEmbedder(batch_size=32)
        embeddings = embedder.embed_texts(texts)
        embedder.cleanup()
        del embedder
        return jsonify({
            'embeddings': embeddings
        })
    except Exception as e:
        print(f"Error embedding the text: {e}")
        return jsonify({"error": "Failed to get response from LLM service"}), 500


@app.route('/retrieve', methods=['POST'])
def retrieve_request():
    try:
        data = request.get_json()
        query = data['query']
        user_query = data['user_query']
        embedder = DocumentEmbedder()
        print("embedding the query: ")
        embedding = embedder.embed_texts([query])[0]
        embedder.cleanup()
        del embedder
        gc.collect()
        context = retrieve(user_query, embedding)
        return jsonify({
            'context': context
        }), 200


    except Exception as e:
        print(f"Error retrieving the query from LLM : {e}")
        return jsonify({
            "error": f"Failed to retieve response from LLM service for this error : {e}"
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)
    print(f"Server is running on port {PORT}")
