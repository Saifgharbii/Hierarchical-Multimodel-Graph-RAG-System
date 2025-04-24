# Ollama requirements
import requests
from .Image_manipulation import encode_image_base64
from .FigureProcessingModule import FiguresProcessing


class LLaMaImageProcessing(FiguresProcessing, model_name="llama"):
    """
    Processes image input using the LLaMa 3.2 Vision model via Ollama.
    """

    def __init__(self, images_bytes: list[bytes], prompt: str, keep_alive: str = "0m") -> None:
        super().__init__(images_bytes=images_bytes, prompt=prompt, keep_alive=keep_alive)

    def query_vision_model(self) -> str:
        base64_images = [encode_image_base64(image_bytes) for image_bytes in self.images_bytes]
        payload = {
            "model": "llama3.2-vision",
            "messages": [
                {
                    "role": "user",
                    "content": self.prompt,
                    "images": base64_images
                }
            ],
            "stream": False,
            "keep_alive": self.keep_alive
        }

        try:
            response = requests.post("http://localhost:11434/api/chat", json=payload)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return ""