
"""-------  To be used if the Ollama server isn't running by default  ----------"""


# import atexit
# import subprocess
# import time
#
# def start_ollama_server():
#     try:
#         # Start Ollama server as a background process
#         process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         print("Ollama server started with PID:", process.pid)
#
#         # Wait a few seconds to ensure the server is up and running
#         time.sleep(5)
#
#         # Check if the server is running
#         if process.poll() is None:
#             print("Ollama server is running.")
#             return process
#         else:
#             print("Failed to start Ollama server.")
#             return None
#     except Exception as e:
#         print(f"Error starting Ollama server: {e}")
#         return None
#
# def stop_ollama_server(process):
#     try:
#         process.terminate()  # Gracefully stop the process
#         process.wait(timeout=5)  # Wait for the process to terminate
#         print("Ollama server stopped.")
#     except Exception as e:
#         print(f"Error stopping Ollama server: {e}")
# # Start Ollama server when the Flask app starts
#
# ollama_process = start_ollama_server()
#
# # Stop Ollama server when the Flask app exits
# @atexit.register
# def shutdown():
#     if ollama_process:
#         stop_ollama_server(ollama_process)



from flask import Flask, request, jsonify
import os
import ollama

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
    message = data.get('message')
    settings = data.get('settings', {})

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Prepare the conversation history and settings
    conversation_history = data.get("messages", [])
    conversation_history.append({"role": "user", "content": message})

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
            model='llama3.1',  # Specify the model
            messages=conversation_history,  # Pass the conversation history
            options=options,  # Pass the model settings
            stream=False  # Set to True if you want streaming responses
        )

        # Extract the model's response
        model_response = response['message']['content']

        # Append the model's response to the conversation history
        conversation_history.append({"role": "assistant", "content": model_response})

        # Return the response to the client
        return jsonify({
            "response": model_response
        })

    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return jsonify({"error": "Failed to get response from LLM service"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # Run on a different port (e.g., 5001)