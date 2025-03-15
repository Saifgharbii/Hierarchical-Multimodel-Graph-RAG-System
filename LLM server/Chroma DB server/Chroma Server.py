from flask import Flask, request, jsonify
import chromadb
import requests
import os
import json
from typing import List, Dict, Any, Optional

app = Flask(__name__)

# Configuration
EMBED_API_URL = os.environ.get('EMBED_API_URL',"http://127.0.0.1:5002/embed")
CHROMA_PERSISTENCE_PATH = os.environ.get("CHROMA_PERSISTENCE_PATH", "./chroma_data")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_PERSISTENCE_PATH)

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from external embedding service"""
    try:
        response = requests.post(
            EMBED_API_URL,
            json={"texts": texts},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()["embeddings"]
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error connecting to embedding service: {e}")
        raise Exception(f"Failed to get embeddings: {str(e)}")

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "chromadb-api"})

@app.route("/collections", methods=["GET"])
def list_collections():
    """List all collections"""
    collections = client.list_collections()
    return jsonify({
        "collections": [collection.name for collection in collections]
    })

@app.route("/collections", methods=["POST"])
def create_collection():
    """Create a new collection"""
    data = request.json
    
    if not data or "name" not in data:
        return jsonify({"error": "Collection name is required"}), 400
    
    name = data["name"]
    metadata = data.get("metadata", {})
    
    try:
        collection = client.create_collection(
            name=name,
            metadata=metadata
        )
        return jsonify({
            "message": f"Collection '{name}' created successfully",
            "name": name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/collections/<collection_name>", methods=["DELETE"])
def delete_collection(collection_name):
    """Delete a collection"""
    try:
        client.delete_collection(collection_name)
        return jsonify({"message": f"Collection '{collection_name}' deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/collections/<collection_name>", methods=["GET"])
def get_collection(collection_name):
    """Get collection information"""
    try:
        collection = client.get_collection(collection_name)
        return jsonify({
            "name": collection.name,
            "metadata": collection.metadata,
            "count": collection.count()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route("/collections/<collection_name>/add", methods=["POST"])
def add_documents(collection_name):
    """Add documents to a collection"""
    data = request.json
    
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    required_fields = ["documents"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"'{field}' field is required"}), 400
    
    documents = data["documents"]
    ids = data.get("ids")
    metadatas = data.get("metadatas")
    use_external_embeddings = data.get("use_external_embeddings", True)
    
    try:
        collection = client.get_collection(collection_name)
        
        # Generate embeddings if using external embedding service
        embeddings = None
        if use_external_embeddings:
            embeddings = get_embeddings(documents)
            
        # Add documents to collection
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        
        return jsonify({
            "message": f"Added {len(documents)} documents to collection '{collection_name}'",
            "count": collection.count()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/collections/<collection_name>/update", methods=["POST"])
def update_documents(collection_name):
    """Update documents in a collection"""
    data = request.json
    
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    required_fields = ["ids", "documents"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"'{field}' field is required"}), 400
    
    ids = data["ids"]
    documents = data["documents"]
    metadatas = data.get("metadatas")
    use_external_embeddings = data.get("use_external_embeddings", True)
    
    try:
        collection = client.get_collection(collection_name)
        
        # Generate embeddings if using external embedding service
        embeddings = None
        if use_external_embeddings:
            embeddings = get_embeddings(documents)
            
        # Update documents in collection
        collection.update(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        return jsonify({
            "message": f"Updated {len(ids)} documents in collection '{collection_name}'"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/collections/<collection_name>/delete", methods=["POST"])
def delete_documents(collection_name):
    """Delete documents from a collection"""
    data = request.json
    
    if not data or "ids" not in data:
        return jsonify({"error": "Document ids are required"}), 400
    
    ids = data["ids"]
    
    try:
        collection = client.get_collection(collection_name)
        collection.delete(ids=ids)
        
        return jsonify({
            "message": f"Deleted {len(ids)} documents from collection '{collection_name}'",
            "count": collection.count()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/collections/<collection_name>/query", methods=["POST"])
def query_collection(collection_name):
    """Query a collection"""
    data = request.json
    
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    query_texts = data.get("query_texts", [])
    query_embeddings = data.get("query_embeddings")
    n_results = data.get("n_results", 10)
    where = data.get("where")
    where_document = data.get("where_document")
    include = data.get("include", ["documents", "metadatas", "distances"])
    
    try:
        collection = client.get_collection(collection_name)
        
        # If no embeddings provided, use the external embedding service
        if not query_embeddings and query_texts:
            query_embeddings = get_embeddings(query_texts)
            
        # Perform query
        results = collection.query(
            query_texts=query_texts if not query_embeddings else None,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include
        )
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/collections/<collection_name>/get", methods=["POST"])
def get_documents(collection_name):
    """Get documents from a collection"""
    data = request.json
    
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    ids = data.get("ids")
    where = data.get("where")
    where_document = data.get("where_document")
    limit = data.get("limit")
    offset = data.get("offset")
    include = data.get("include", ["documents", "metadatas"])
    
    try:
        collection = client.get_collection(collection_name)
        
        # Get documents
        results = collection.get(
            ids=ids,
            where=where,
            where_document=where_document,
            limit=limit,
            offset=offset,
            include=include
        )
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/collections/<collection_name>/count", methods=["GET"])
def count_documents(collection_name):
    """Count documents in a collection"""
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
        return jsonify({"count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)