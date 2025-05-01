import os
import json
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModel


class DocumentEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32, max_length: int = 2048):
        """Initialize the embedder with model and parameters."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.devices = []
        self.models = {}

        # Check available GPUs
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                device = f"cuda:{i}"
                self.devices.append(device)

                # Load model on each GPU
                self.models[device] = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                self.models[device].to(device)
        else:
            self.devices = ["cpu"]
            self.models["cpu"] = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    def _get_embeddings_batch(self, texts: List[str], device: str) -> List[List[float]]:
        """Get embeddings for a batch of texts on a specific device."""
        embeddings_list = []
        if not texts:
            return embeddings_list

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True,
                                           max_length=2048, return_tensors='pt').to(device)
            local_model = self.models[device]
            # Compute token embeddings
            with torch.no_grad():
                outputs = local_model(**encoded_input)

            # Use CLS token embedding or mean pooling as needed
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token

            # Move to CPU to free GPU memory
            embeddings = embeddings.cpu().numpy()
            embeddings_list.extend(embeddings.tolist())

        return embeddings_list

    def process_texts_with_device(self, texts_with_ids: List[Tuple[int, str]], device: str) -> List[
        Tuple[int, List[float]]]:
        """Process a batch of texts on a specific device."""
        texts = [text for _, text in texts_with_ids]
        ids = [id_ for id_, _ in texts_with_ids]

        embeddings = self._get_embeddings_batch(texts, device)
        return list(zip(ids, embeddings))

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts using all available devices."""
        if not texts:
            return []

        # Assign IDs to texts for tracking
        texts_with_ids = list(enumerate(texts))

        # Distribute texts across devices
        n_devices = len(self.devices)
        chunks = [[] for _ in range(n_devices)]
        for i, item in enumerate(texts_with_ids):
            chunks[i % n_devices].append(item)

        # Process on each device
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_devices) as executor:
            future_to_device = {
                executor.submit(self.process_texts_with_device, chunk, device): device
                for device, chunk in zip(self.devices, chunks) if chunk
            }

            for future in concurrent.futures.as_completed(future_to_device):
                device = future_to_device[future]
                try:
                    result = future.result()
                    results.extend(result)
                except Exception as e:
                    print(f"error in processing on {device} : {str(e)}")

        # Sort results back by original ID
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]