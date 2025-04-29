import torch
from transformers import AutoProcessor, MultiModalityCausalLM
from PIL import Image
import io
from typing import List, Dict, Tuple, Optional
import base64
from tqdm import tqdm

class DeepSeekVLProcessor:
    """
    A class for processing batches of images with DeepSeek VL model to extract meaningful text.
    """

    def __init__(self, model_name: str = "deepseek-ai/deepseek-vl-7b-chat", device: str = "cuda", batch_size: int = 4):
        """
        Initialize the DeepSeek VL processor.

        Args:
            model_name: The name of the DeepSeek VL model to use
            device: The device to run the model on ('cuda' or 'cpu')
            batch_size: Default batch size for processing
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.batch_size = batch_size

        print(f"Loading DeepSeek VL model on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MultiModalityCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        print("Model loaded successfully!")

    def _bytes_to_image(self, image_bytes: bytes) -> Image.Image:
        """Convert image bytes to PIL Image."""
        return Image.open(io.BytesIO(image_bytes))

    def _process_batch(self, batch_images: List[Image.Image], batch_prompts: List[str]) -> List[str]:
        """Process a batch of images with their corresponding prompts."""
        inputs = self.processor(
            text=batch_prompts,
            images=batch_images,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.1
            )

        # Decode the generated text
        decoded_outputs = []
        for i, output in enumerate(outputs):
            # Find the right offset to decode from
            input_length = inputs.input_ids[i].shape[0]
            if hasattr(self.processor, "decode"):
                text = self.processor.decode(output[input_length:], skip_special_tokens=True)
            else:
                text = self.processor.tokenizer.decode(output[input_length:], skip_special_tokens=True)
            decoded_outputs.append(text.strip())

        return decoded_outputs

    def process(self,
                image_bytes_list: List[bytes],
                prompts: List[str],
                batch_size: Optional[int] = None) -> List[str]:
        """
        Process a list of image bytes with their corresponding prompts in batches.

        Args:
            image_bytes_list: List of image bytes
            prompts: List of prompts corresponding to the images
            batch_size: Batch size to use (defaults to self.batch_size if None)

        Returns:
            List of extracted text from the images
        """
        if len(image_bytes_list) != len(prompts):
            raise ValueError("Number of images and prompts must match")

        batch_size = batch_size or self.batch_size
        all_results = []

        # Convert all bytes to PIL Images
        images = [self._bytes_to_image(img_bytes) for img_bytes in image_bytes_list]

        # Process in batches
        for i in tqdm(range(0, len(images), batch_size), desc="Processing image batches"):
            batch_images = images[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]

            batch_results = self._process_batch(batch_images, batch_prompts)
            all_results.extend(batch_results)

        return all_results


def process_image_batch(
        image_bytes_list: List[bytes],
        prompts: List[str],
        model_name: str = "deepseek-ai/deepseek-vl-7b-chat",
        device: str = "cuda",
        batch_size: int = 4
) -> List[str]:
    """
    Convenience function to process a batch of images without creating a processor class.

    Args:
        image_bytes_list: List of image bytes in PNG format
        prompts: List of prompts to use for each image
        model_name: The name of the DeepSeek VL model to use
        device: The device to run the model on ('cuda' or 'cpu')
        batch_size: Batch size for processing

    Returns:
        List of extracted text from the images
    """
    processor = DeepSeekVLProcessor(model_name=model_name, device=device, batch_size=batch_size)
    return processor.process(image_bytes_list, prompts, batch_size)



if __name__ == "__main__":
    # Example usage
    def read_image_as_bytes(image_path):
        with open(image_path, "rb") as f:
            return f.read()


    # Example with a local image
    image_bytes = read_image_as_bytes("path/to/image.png")

    # Process a single image
    processor = DeepSeekVLProcessor()
    results = processor.process(
        [image_bytes],
        ["Extract and summarize the text content from this image."]
    )
    print(results[0])

    # Process multiple images
    image_bytes_list = [read_image_as_bytes(f"path/to/image{i}.png") for i in range(1, 4)]
    prompts = [
        "Extract the main paragraph from this image.",
        "Summarize the text content in this image.",
        "What text is shown in this image? Provide it as a complete paragraph."
    ]

    results = processor.process(image_bytes_list, prompts)
    for i, result in enumerate(results):
        print(f"Image {i + 1} result: {result}")

    # Alternative: use the convenience function
    results = process_image_batch(
        image_bytes_list,
        prompts,
        batch_size=2  # Process 2 images at a time
    )