
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
import requests
import json
import os
app = Flask(__name__)

# Ollama configuration
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")



@app.route('/generate', methods=['POST'])
def generate_response():
    """Handle incoming messages, process with Llama3.1 via Ollama, and return the response"""
    data = request.json
    message = data.get('message')
    settings = data.get('settings', {})

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Prepare request to Ollama API
    llm_request_data = {
        "prompt": message,
        "model": "llama3.1",
        "system": settings.get("systemPrompt", "You're a helpful assistant."),
        "options": {
            "temperature": settings.get("temperature", 0.5),
            "top_p": settings.get("topP", 1.0),
            "top_k": settings.get("topK", 5),
            "max_tokens": settings.get("maxTokens", 2000)
        }
    }

    # Call Ollama API
    try:
        # Use a session to stream the response
        with requests.Session() as session:
            response = session.post(OLLAMA_API_URL, json=llm_request_data, stream=True)
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if line:
                    # Decode the line and parse it as JSON
                    chunk = line.decode('utf-8')
                    chunk_data = json.loads(chunk)

                    # Append the response chunk to the full response
                    full_response += chunk_data.get("response", "")

                    # Check if the response is done
                    if chunk_data.get("done", False):
                        break

            # Return the full response to the client
            return jsonify({
                "response": full_response
            })

    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return jsonify({"error": "Failed to get response from LLM service"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # Run on a different port (e.g., 5001)