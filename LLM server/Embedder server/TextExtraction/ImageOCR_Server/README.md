# Ollama Vision API

A Flask-based REST API wrapper for using Ollama's vision models with optimized resource management.

## Overview

This service provides an API endpoint that accepts images and text prompts, processes them through the Ollama API, and returns the vision model's response. It includes optimizations for balancing CPU and GPU usage while managing memory constraints.

## Features

- **Resource Optimization**: Balances between CPU and GPU usage
- **Memory Management**: Configurable VRAM limits to prevent OOM errors
- **Image Optimization**: Automatically resizes and optimizes large images
- **Concurrency Control**: Limits parallel requests to prevent resource exhaustion
- **Health Monitoring**: Endpoint to check service status

## Requirements

- Python 3.8+
- Flask
- Requests
- Pillow (PIL)
- Ollama running locally or on a remote server
- CUDA-compatible GPU (optional but recommended)

## Installation

1. Clone the repository or download the script
2. Install dependencies:

```bash
pip install flask requests pillow
```

3. Ensure Ollama is installed and running:

```bash
# Check if Ollama is running
curl http://localhost:11434/api/health
```

4. Pull the vision model:

```bash
ollama pull llama3.2-vision
```

## Configuration

The service can be configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_API_URL` | URL of the Ollama API | `http://localhost:11434` |
| `OLLAMA_MODEL` | Vision model to use | `llama3.2-vision` |
| `OLLAMA_KEEP_ALIVE` | Keep model loaded for this duration | `5m` |
| `MAX_VRAM_USAGE_MB` | Maximum VRAM usage in MB | `4000` (4GB) |
| `MAX_IMAGE_SIZE` | Max dimension for image resizing | `1024` |
| `MAX_CONCURRENT_REQUESTS` | Max parallel requests | `1` |
| `PORT` | Server port | `5003` |
| `HOST` | Server host | `0.0.0.0` |
| `DEBUG` | Enable Flask debug mode | `False` |

## Usage

### Starting the Server

```bash
# Set environment variables if needed
export MAX_VRAM_USAGE_MB=4000  # Adjust based on your GPU
export MAX_CONCURRENT_REQUESTS=1

# Run the server
python app.py
```

### API Endpoints

#### Process Image

```
POST /process
```

**Request:**
- `multipart/form-data` with:
  - `image`: Image file (JPG, PNG, etc.)
  - `message`: Text prompt for the vision model

**Response:**
```json
{
  "response": "Description or analysis from the vision model"
}
```

#### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "ollama": "connected",
  "model": "llama3.2-vision"
}
```

## Resource Optimization Details

### GPU Memory Management

- **VRAM Limiting**: Sets maximum VRAM usage to prevent out-of-memory errors
- **Half-precision Caching**: Uses FP16 for key/value cache to reduce memory usage
- **Adaptive GPU Usage**: Dynamically allocates GPU layers based on available resources

### CPU Optimization

- **Multi-threading**: Utilizes optimal CPU thread count based on system resources
- **Workload Balancing**: Automatically offloads work to CPU when GPU is saturated

### Image Optimization

- **Automatic Resizing**: Reduces large images to limit memory usage
- **Format Optimization**: Converts to memory-efficient formats
- **Quality Adjustment**: Balances visual quality and memory consumption

## Performance Tips

1. **GPU Configuration**:
   - With 6GB VRAM, set `MAX_VRAM_USAGE_MB=4000` to leave headroom for system
   - For integrated GPUs, consider reducing to `MAX_VRAM_USAGE_MB=2000`

2. **Processing Large Batches**:
   - For high-volume processing, keep `MAX_CONCURRENT_REQUESTS=1`
   - Process sequentially to avoid memory issues

3. **Image Size Considerations**:
   - Very large images (>10MB) may still cause issues despite optimization
   - Consider pre-processing images before sending to the API

4. **Model Selection**:
   - Smaller models use less VRAM but may have reduced quality
   - Consider using more efficient models like `phi3-vision-mini` if available

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce `MAX_VRAM_USAGE_MB`
   - Decrease `MAX_IMAGE_SIZE`
   - Ensure no other GPU-intensive tasks are running

2. **Slow Response Times**:
   - Check CPU utilization and adjust `num_thread` if needed
   - Verify network connectivity to Ollama service
   - Consider using a smaller/faster model

3. **Connection Errors**:
   - Ensure Ollama is running and accessible
   - Verify the `OLLAMA_API_URL` is correct
   - Check firewall settings if using a remote Ollama instance

## Development and Extension

The code is designed to be modular and extensible. Key components:

- `optimize_image()`: Image preprocessing and optimization
- `query_vision_model()`: Handles Ollama API communication with resource management
- Flask route handlers: API endpoints and request validation

To add features, consider extending these components while maintaining the resource optimization principles.
