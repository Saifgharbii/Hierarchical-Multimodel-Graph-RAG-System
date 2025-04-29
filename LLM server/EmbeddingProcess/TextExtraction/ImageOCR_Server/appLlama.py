from flask import Flask, request, jsonify
from base64 import b64encode
import requests
import os
import logging
from PIL import Image
from io import BytesIO
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2-vision")
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "5m")

# Resource management settings
MAX_VRAM_USAGE_MB = int(os.environ.get("MAX_VRAM_USAGE_MB", 4000))  # 4GB VRAM limit
MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", 1024))  # Resize large images
MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", 1))

# Semaphore to limit concurrent requests
request_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)


def optimize_image(image_bytes, max_size=MAX_IMAGE_SIZE):
    """Optimize image to reduce memory usage."""
    try:
        image = Image.open(BytesIO(image_bytes))

        # Resize if image is too large
        if max(image.width, image.height) > max_size:
            if image.width > image.height:
                new_width = max_size
                new_height = int(image.height * (max_size / image.width))
            else:
                new_height = max_size
                new_width = int(image.width * (max_size / image.height))

            image = image.resize((new_width, new_height), Image.LANCZOS)

        # Convert to RGB if needed (removing alpha channel)
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
            image = background

        # Save optimized image to bytes
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        return buffer.getvalue()

    except Exception as e:
        logger.error(f"Image optimization failed: {e}")
        # Return original image if optimization fails
        return image_bytes


def query_vision_model(image_bytes, prompt) -> str:
    """
    Query the Ollama vision model with optimized resource usage.

    Args:
        image_bytes (bytes): Raw image data
        prompt (str): Text prompt for the vision model

    Returns:
        str: Model response text
    """
    # Acquire semaphore to limit concurrent requests
    with request_semaphore:
        try:
            # Optimize image to reduce memory usage
            optimized_image = optimize_image(image_bytes)
            base64_image = b64encode(optimized_image).decode("utf-8")

            # Prepare request payload with resource management options
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [base64_image]
                    }
                ],
                "stream": False,
                "keep_alive": KEEP_ALIVE,
                "options": {
                    "num_gpu": 1,  # Use GPU
                    "num_thread": os.cpu_count() or 4,  # Optimal CPU threads
                    "num_batch": 1,  # Process one request at a time
                    "temperature": 0.7,
                    "top_k": 40,
                    "top_p": 0.9,
                    "mirostat": 0,  # Disable mirostat to reduce complexity
                    # GPU memory management
                    "f16_kv": True,  # Use half-precision for key/value cache
                    "gpu_layers": -1,  # Let Ollama decide optimal layers
                    "rope_frequency_base": 10000,  # Use standard RoPE config
                    "rope_frequency_scale": 1,
                    "main_gpu": 1
                }
            }

            # Set max VRAM usage if defined
            if MAX_VRAM_USAGE_MB > 0:
                payload["options"]["vram_limit_mb"] = MAX_VRAM_USAGE_MB

            logger.info(f"Sending request to Ollama with model: {OLLAMA_MODEL}")
            response = requests.post(f"{OLLAMA_API_URL}/api/chat", json=payload, timeout=500)
            response.raise_for_status()

            result = response.json().get("message", {}).get("content", "")
            logger.info(f"Successfully received response: {len(result)} chars")
            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            return f"Error connecting to Ollama service: {e}"
        except Exception as e:
            logger.error(f"Unexpected error in vision model query: {e}")
            return f"Unexpected error: {e}"


# Initialize Flask application
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint to check API health status"""
    try:
        # Simple ping to Ollama API
        response = requests.get(f"{OLLAMA_API_URL}/api/health")
        if response.status_code == 200:
            return jsonify({
                'status': 'healthy',
                'ollama': 'connected',
                'model': OLLAMA_MODEL
            })
        else:
            return jsonify({
                'status': 'degraded',
                'ollama': 'connected but not ready',
                'model': OLLAMA_MODEL
            }), 503
    except requests.exceptions.RequestException:
        return jsonify({
            'status': 'unhealthy',
            'ollama': 'disconnected',
            'model': OLLAMA_MODEL
        }), 503


@app.route("/process", methods=["POST"])
def process_image():
    """
    Process an image with the vision model.

    Requires:
        - 'image' file in request files
        - 'message' field in form data

    Returns:
        JSON response with model output or error
    """
    logger.info("Received image processing request")

    # Validate input
    if 'image' not in request.files:
        logger.warning("No image file provided")
        return jsonify({'error': 'Image file is required'}), 400

    if 'message' not in request.form:
        logger.warning("No message provided")
        return jsonify({'error': 'Message prompt is required'}), 400

    try:
        # Read input data
        image_file = request.files['image']
        image_bytes = image_file.read()
        message = request.form['message']

        # Validate image size
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            return jsonify({'error': 'Image file too large (max 10MB)'}), 413

        # Process the request
        response = query_vision_model(image_bytes, message)

        # Return results
        return jsonify({'response': response})

    except Exception as e:
        logger.exception("Error processing request")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "False").lower() == "true"

    logger.info(f"Starting Ollama Vision API on {host}:{port} with model {OLLAMA_MODEL}")
    logger.info(f"VRAM limit: {MAX_VRAM_USAGE_MB}MB, Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")

    app.run(host=host, port=port, debug=debug)
