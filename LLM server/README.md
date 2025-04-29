# LLM Server (Ollama) Documentation

## Overview
The Flask-based service provides two core functionalities:
1. Main LLM response generation
2. Conversation title generation
3. Secure API access via Bearer token authentication

## Features
- Integration with Ollama's local LLM models
- Dynamic model parameter configuration
- Automatic title generation for conversations
- API key security middleware
- Error handling for LLM interactions

## Service Setup

### Prerequisites
- Python 3.9+
- Ollama server running locally (on localhost:)
- Required models downloaded via Ollama:
  ```bash
  ollama pull llama3.1
  ollama pull llama3.2:1b
  ```

### Installation
1. Clone your RAG system repository
2. Install dependencies:
   ```bash
   pip install requirements.txt
   ```
3. Configure environment variables (create `.env` file):
   ```env
   API_KEY=your_secure_key_here
   OLLAMA_HOST=http://localhost:11434
   ```

### Running the Service
```bash
python llm_server.py
```
Service runs on `http://localhost:5001`

## API Endpoints

### 1. Generate Response
- **Endpoint**: `/generate`
- **Method**: POST
- **Headers**:
  - `Authorization: Bearer <API_KEY>`
- **Request Body**:
  ```json
  {
    "messages": [
      {"role": "user", "content": "What is RAG?"}
    ],
    "settings": {
      "temperature": 0.7,
      "topP": 0.9,
      "topK": 50,
      "maxTokens": 1000
    }
  }
  ```
- **Success Response**:
  ```json
  {"response": "RAG (Retrieval-Augmented Generation) is..."}
  ```

### 2. Generate Title
- **Endpoint**: `/generate-title`
- **Method**: POST
- **Headers**:
  - `Authorization: Bearer <API_KEY>`
- **Request Body**:
  ```json
  {"message": "Can you explain quantum computing?"}
  ```
- **Success Response**:
  ```json
  {"response": "Quantum computing explained"}
  ```

## Models Configuration
| Endpoint         | Model           | Parameters              |
|------------------|-----------------|-------------------------|
| /generate        | llama3.1        | temperature: 0.5        |
|                  |                 | top_p: 1.0              |
|                  |                 | maxTokens: 2000         |
|------------------|-----------------|-------------------------|
| /generate-title  | llama3.2:1b     | temperature: 0.5        |
|                  |                 | top_p: 0.8              |
|                  |                 | maxTokens: 10           |

## Security
- API key authentication via Bearer tokens
- Middleware implementation:
  ```python
  def require_api_key(func):
      def wrapper(*args, **kwargs):
          key = request.headers.get("Authorization")
          if key != f"Bearer {API_KEY}":
              return jsonify({"error": "Unauthorized"}), 401
          return func(*args, **kwargs)
      return wrapper
  ```

## Integration with RAG System
1. Add the service to your architecture diagram:
   ```
   [Client] → [API Gateway] → [LLM Server (Ollama)] ↔ [Ollama Service]
                         ↘       ↗
                       [RAG Pipeline]
   ```

2. Sample integration code:
```python
def query_llm(prompt: str) -> str:
    headers = {"Authorization": "Bearer Secret"}
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "settings": {"temperature": 0.7}
    }
    response = requests.post(
        "http://localhost:5001/generate",
        json=data,
        headers=headers
    )
    return response.json()["response"]
```

## Example Usage

### Generate Response
```bash
curl -X POST http://localhost:5001/generate \
  -H "Authorization: Bearer Secret" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain neural networks"}
    ],
    "settings": {
      "temperature": 0.7
    }
  }'
```

### Generate Title
```bash
curl -X POST http://localhost:5001/generate-title \
  -H "Authorization: Bearer Secret" \
  -H "Content-Type: application/json" \
  -d '{"message": "How does photosynthesis work?"}'
```

## Error Handling
| Code | Status          | Description                     |
|------|-----------------|---------------------------------|
| 401  | Unauthorized    | Missing/invalid API key         |
| 400  | Bad Request     | Missing required parameters     |
| 500  | Server Error    | Ollama API communication failed |

## Performance Considerations
- Keep `num_predict` values conservative for latency control
- Monitor Ollama server resource usage (CPU/GPU/RAM)
- Consider model quantization for better performance
- Implement caching for frequent similar requests

## Maintenance
1. Model updates:
   ```bash
   ollama pull llama3.1
   ollama pull llama3.2:1b
   ```
2. Service monitoring:
   - API response times
   - Error rates
   - Model performance metrics

## Troubleshooting
Common issues:
- Ollama connection errors: Verify Ollama service is running
- Model not found: Check installed models with `ollama list`
- API key mismatches: Validate `.env` file and request headers
