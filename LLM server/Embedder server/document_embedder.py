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


def extract_content_for_embedding(document: Dict[Any, Any]) -> Tuple[List[str], List[str]]:
    """Extract all texts (chunks, titles, descriptions) that need embedding."""
    chunks = []
    chunk_sources = []

    def process_section(section, path=""):
        # Add section title
        if section.get("title"):
            title = section["title"]
            chunks.append(title)
            chunk_sources.append(f"{path}/title")

        # Add section description
        if section.get("description"):
            description = section["description"]
            chunks.append(description)
            chunk_sources.append(f"{path}/description")

        # Add section text content
        if section.get("text_content"):
            text_content = section["text_content"]
            chunks.append(text_content)
            chunk_sources.append(f"{path}/text_content")

        # Add individual chunks
        if "chunks" in section and section["chunks"]:
            for i, chunk in enumerate(section["chunks"]):
                if chunk and chunk != "...":
                    chunks.append(chunk)
                    chunk_sources.append(f"{path}/chunks/{i}")

        # Process subsections
        if "subsections" in section and section["subsections"]:
            for i, subsection in enumerate(section["subsections"]):
                subsection_path = f"{path}/subsections/{i}"
                process_section(subsection, subsection_path)

        # Process subsubsections
        if "subsubsections" in section and section["subsubsections"]:
            for i, subsubsection in enumerate(section["subsubsections"]):
                subsubsection_path = f"{path}/subsubsections/{i}"
                process_section(subsubsection, subsubsection_path)

    # Process each content section in the document
    if "content" in document and document["content"]:
        for i, section in enumerate(document["content"]):
            section_path = f"content/{i}"
            process_section(section, section_path)

    return chunks, chunk_sources


def update_document_with_embeddings(document: Dict[Any, Any], chunk_sources: List[str],
                                    embeddings: List[List[float]]) -> Dict[Any, Any]:
    """
    Args:
        document: The original document dictionary.
        chunk_sources: A list of source paths pointing to fields or list elements to embed.
        embeddings: A list of embeddings corresponding to the sources.

    Returns:
        The updated document with embeddings inserted.

    Goes through zip(embeddings, chunk_sources) and injects each embedding into
    the updated document based on the source path.

    - The keys 'title' and 'description' are transformed into dictionaries containing the original text and the embedding.
    - The list elements under 'chunks' are transformed into dictionaries containing the chunk text and the embedding.
    """
    updated_doc = document.copy()

    for embedding, source in zip(embeddings, chunk_sources):
        parts = source.split('/')  # Example: ["content", "0", "subsections", "2", "chunks", "1"]
        node = updated_doc

        # Navigate down to the parent of the field to be modified
        for p in parts[:-1]:
            if p.isdigit():
                node = node[int(p)]
            else:
                node = node[p]

        last = parts[-1]

        if last.isdigit():
            # It is an index inside a list (case of chunks)
            idx = int(last)
            original = node[idx]
            node[idx] = {
                "chunk": original,
                "embedding": embedding
            }
        else:
            # It is a dictionary key
            original = node[last]
            if last == "title":
                node[last] = {
                    "title": original,
                    "embedding": embedding
                }
            elif last == "description":
                node[last] = {
                    "description": original,
                    "embedding": embedding
                }

    return updated_doc


def process_json_file(file_path: str, embedder: DocumentEmbedder, output_dir="EmbeddedDocuments") -> None:
    """Process a single JSON file."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Load JSON data
        with open(file_path, 'r', encoding='utf-8') as f:
            document = json.load(f)

        # Extract content for embedding
        chunks, chunk_sources = extract_content_for_embedding(document)
        if not chunks:
            return

        # Get embeddings
        embeddings = embedder.embed_texts(chunks)

        # Update document with embeddings
        updated_document = update_document_with_embeddings(document, chunk_sources, embeddings)

        # output_dir = os.path.join("/kaggle/working", output_dir) #On Kaggle

        # Save updated document
        file_name = file_path.split("/")[-1].replace('.json', '_embedded.json')
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated_document, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")


def main(input_path, model_name, batch_size, max_length):
    """Main function to process all JSON files."""
    # Initialize embedder
    embedder = DocumentEmbedder(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length
    )
    print(f"Initializing embedder with model: {model_name}")

    # Find all JSON files
    input_path = Path(input_path)
    if input_path.is_file() and input_path.suffix == '.json':
        json_files = [input_path]
    else:
        json_files = list(input_path.glob('**/*.json'))

    # Process files
    for file_path in tqdm(json_files, desc="Processing files"):
        print(f"processing the file {file_path}")
        process_json_file(str(file_path), embedder)


if __name__ == "__main__":
    input_path = {"Kaggle": "/kaggle/input/chunked-documents-p2m/chunked_documents",
                  "Local": "./DocumentsChunking/ChunkedDocuments"}
    model_name = "dunzhang/stella_en_400M_v5"
    batch_size = 64
    max_length = 2048
    main(input_path["Local"], model_name, batch_size, max_length)
    # main(input_path["Kaggle"], model_name, batch_size, max_length) #On Kaggle

