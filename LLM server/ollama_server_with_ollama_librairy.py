from flask import Flask, request, jsonify
import os
import ollama

llm_model = "llama3.1"

app = Flask(__name__)

# API Key Configuration
API_KEY = os.getenv("API_KEY", "Secret")  # Store this securely

# Middleware to check API key
def require_api_key(func):
    def wrapper(*args, **kwargs):
        key = request.headers.get("Authorization")
        if key != f"Bearer {API_KEY}":  # Use Bearer token format
            return jsonify({"error": "Unauthorized"}), 401
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

@app.route('/generate', methods=['POST'])
@require_api_key
def generate_response():
    """Handle incoming messages, process with Llama3.1 via Ollama, and return the response"""
    data = request.json
    messages = data.get('messages')
    settings = data.get('settings', {})

    if not messages:
        return jsonify({"error": "No message provided"}), 400


    # Prepare the options for the Ollama API
    options = {
        "temperature": settings.get("temperature", 0.5),
        "top_p": settings.get("topP", 1.0),
        "top_k": settings.get("topK", 5),
        "num_predict": settings.get("maxTokens", 2000)
    }

    # Call the Ollama API using the ollama library
    try:
        response = ollama.chat(
            model= llm_model,  # Specify the model
            messages=messages,  # Pass the conversation history
            options=options,  # Pass the model settings
            stream=False  # Set to True if you want streaming responses
        )

        # Extract the model's response
        model_response = response['message']['content']
        print(response)

        # Return the response to the client
        return jsonify({
            "response": model_response
        })

    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return jsonify({"error": "Failed to get response from LLM service"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # Run on a different port (e.g., 5001)