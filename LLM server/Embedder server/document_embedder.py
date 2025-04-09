import os
import json
import logging
import time
import glob
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import docx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embedding_process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RAG-Embedder")


class RAGEmbedder:
    def __init__(
            self,
            input_dir: str,
            chroma_host: str = "localhost",
            chroma_port: int = 8000,
            collection_name: str = "rag_documents",
            model_name: str = "sentence-transformers/all-distilroberta-v1",
            segment_size_mb: int = 128,
            batch_size: int = 32,
            use_gpu: bool = True,
            checkpoint_path: str = "embedding_checkpoint.json"
    ):
        self.input_dir = input_dir
        self.collection_name = collection_name
        self.segment_size_mb = segment_size_mb
        self.batch_size = batch_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.checkpoint_path = checkpoint_path

        # Setup device for embedding
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize model
        model_cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(model_cache_dir, exist_ok=True)
        self.model = SentenceTransformer(model_name, cache_folder=model_cache_dir)
        self.model.to(self.device)

        # Initialize Chroma client
        self.client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",
                chroma_client_auth_credentials="admin:admin"
            )
        )

        # Get or create collection with HNSW index
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Connected to existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name),
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")

        # Load checkpoint if exists
        self.processed_files = set()
        self.current_segment = 0
        self.load_checkpoint()

    def preprocess_doc(self, file_path: str) -> Dict[str, str]:
        """
        Placeholder for document preprocessing function.
        This will be implemented by the user.
        """
        # This is just a placeholder - user will replace with their implementation
        return {"placeholder": "This will be replaced by user's implementation"}

    def load_checkpoint(self) -> None:
        """Load processing checkpoint if it exists"""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                self.processed_files = set(checkpoint.get('processed_files', []))
                self.current_segment = checkpoint.get('current_segment', 0)
                logger.info(
                    f"Loaded checkpoint: Segment {self.current_segment}, {len(self.processed_files)} files processed")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                logger.info("Starting from beginning")
        else:
            logger.info("No checkpoint found. Starting from beginning")

    def save_checkpoint(self) -> None:
        """Save current processing state to checkpoint file"""
        try:
            with open(self.checkpoint_path, 'w') as f:
                json.dump({
                    'processed_files': list(self.processed_files),
                    'current_segment': self.current_segment,
                    'timestamp': time.time()
                }, f)
            logger.info(
                f"Saved checkpoint: Segment {self.current_segment}, {len(self.processed_files)} files processed")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def get_document_paths(self) -> List[str]:
        """Get all document paths in the input directory"""
        return glob.glob(os.path.join(self.input_dir, "**/*.docx"), recursive=True)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts into embeddings"""
        # Use model directly for batched encoding
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                device=self.device
            )
        return embeddings

    def estimate_memory_size(self, texts: List[Tuple[str, str, str]]) -> float:
        """Estimate memory size of a batch in MB"""
        # Rough estimation based on text lengths and embedding dimensions
        text_memory = sum(len(text) * 2 for _, _, text in texts) / (1024 * 1024)  # Text in UTF-16
        # Assuming embedding dimension from model
        embedding_dimension = self.model.get_sentence_embedding_dimension()
        embedding_memory = len(texts) * embedding_dimension * 4 / (1024 * 1024)  # 4 bytes per float
        return text_memory + embedding_memory

    def process_segment(self, document_paths: List[str]) -> Tuple[int, bool]:
        """
        Process a segment of documents up to the specified memory limit
        Returns:
            - Number of documents processed
            - Whether processing should continue
        """
        documents_processed = 0
        segment_size = 0
        segment_data = []

        for file_path in tqdm(document_paths, desc=f"Processing Segment {self.current_segment}"):
            if file_path in self.processed_files:
                continue

            try:
                # Process document - this will be replaced by user's implementation
                doc_parts = self.preprocess_doc(file_path)

                for title, content in doc_parts.items():
                    if not content.strip():
                        continue

                    doc_id = f"{os.path.basename(file_path)}_{title}_{hash(content) & 0xffffffff}"
                    segment_data.append((doc_id, title, content))

                    # Check if we've reached segment size limit
                    current_batch_size = self.estimate_memory_size(segment_data)
                    if current_batch_size >= self.segment_size_mb:
                        logger.info(f"Reached segment size limit: {current_batch_size:.2f}MB")
                        self.processed_files.add(file_path)
                        documents_processed += 1
                        break

                # If we've reached the segment size, stop processing more files
                if segment_size >= self.segment_size_mb:
                    break

                self.processed_files.add(file_path)
                documents_processed += 1

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        # If we have data to embed
        if segment_data:
            self.embed_and_store(segment_data)

        return documents_processed, len(segment_data) > 0

    def embed_and_store(self, segment_data: List[Tuple[str, str, str]]) -> None:
        """Embed and store a segment of documents in Chroma DB"""
        if not segment_data:
            return

        ids = [item[0] for item in segment_data]
        titles = [item[1] for item in segment_data]
        texts = [item[2] for item in segment_data]

        logger.info(f"Embedding {len(texts)} text segments")

        # Process in batches to optimize memory usage
        for i in range(0, len(texts), self.batch_size):
            batch_end = min(i + self.batch_size, len(texts))
            batch_ids = ids[i:batch_end]
            batch_texts = texts[i:batch_end]
            batch_titles = titles[i:batch_end]

            # Generate embeddings
            batch_embeddings = self.encode_texts(batch_texts)

            # Create metadata
            metadata = [{"title": title, "source": id.split('_')[0]} for id, title in zip(batch_ids, batch_titles)]

            # Add to Chroma
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings.tolist(),
                documents=batch_texts,
                metadatas=metadata
            )

            logger.info(f"Added batch {i // self.batch_size + 1}/{(len(texts) - 1) // self.batch_size + 1} to ChromaDB")

    def process_all(self) -> None:
        """Process all documents in segments"""
        document_paths = self.get_document_paths()
        total_documents = len(document_paths)
        remaining_documents = [doc for doc in document_paths if doc not in self.processed_files]

        logger.info(f"Found {total_documents} documents, {len(remaining_documents)} remaining to process")

        try:
            while remaining_documents:
                logger.info(f"Starting segment {self.current_segment}")
                start_time = time.time()

                docs_processed, should_continue = self.process_segment(remaining_documents)

                elapsed_time = time.time() - start_time
                logger.info(
                    f"Segment {self.current_segment} completed in {elapsed_time:.2f}s, processed {docs_processed} documents")

                # Update remaining documents
                remaining_documents = [doc for doc in document_paths if doc not in self.processed_files]

                # Increment segment counter and save checkpoint
                self.current_segment += 1
                self.save_checkpoint()

                # If the segment didn't process anything useful, or we're done, exit
                if not should_continue or not remaining_documents:
                    break

                logger.info(f"{len(remaining_documents)} documents remaining")

            logger.info("All documents processed successfully")

        except KeyboardInterrupt:
            logger.info("Process interrupted by user")
            self.save_checkpoint()
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            self.save_checkpoint()


if __name__ == "__main__":
    # Example usage
    embedder = RAGEmbedder(
        input_dir="./documents",
        chroma_host="localhost",
        chroma_port=8000,
        collection_name="rag_collection",
        segment_size_mb=128,
        use_gpu=True
    )

    embedder.process_all()