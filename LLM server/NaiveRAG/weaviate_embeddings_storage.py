import os
import json
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import weaviate
import GPUtil
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system performance during processing"""

    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.gpu_available = len(GPUtil.getGPUs()) > 0

    def get_current_stats(self):
        """Get current system statistics"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = current_memory - self.initial_memory
        cpu_percent = self.process.cpu_percent()

        stats = {
            'memory_usage_mb': memory_usage,
            'total_memory_mb': current_memory,
            'cpu_percent': cpu_percent,
            'elapsed_time': time.time() - self.start_time
        }

        if self.gpu_available:
            try:
                gpu = GPUtil.getGPUs()[0]  # RTX 4060
                stats['gpu_memory_used_mb'] = gpu.memoryUsed
                stats['gpu_memory_total_mb'] = gpu.memoryTotal
                stats['gpu_utilization'] = gpu.load * 100
            except:
                pass

        return stats

    def log_stats(self, stage_name: str):
        """Log current statistics for a processing stage"""
        stats = self.get_current_stats()
        logger.info(f"[{stage_name}] Memory: {stats['memory_usage_mb']:.1f}MB, "
                    f"CPU: {stats.get('cpu_percent', 0):.1f}%, "
                    f"Time: {stats['elapsed_time']:.2f}s")

        if 'gpu_memory_used_mb' in stats:
            logger.info(f"[{stage_name}] GPU Memory: {stats['gpu_memory_used_mb']}/{stats['gpu_memory_total_mb']}MB, "
                        f"GPU Util: {stats['gpu_utilization']:.1f}%")


class HierarchicalClusterer:
    """Hierarchical clustering for embeddings"""

    def __init__(self, n_clusters_l1=50, n_clusters_l2=10, n_clusters_l3=10, random_state=42):
        self.n_clusters_l1 = n_clusters_l1
        self.n_clusters_l2 = n_clusters_l2
        self.n_clusters_l3 = n_clusters_l3
        self.random_state = random_state
        self.cluster_hierarchy = {}

    def fit_predict(self, embeddings: np.ndarray, monitor: PerformanceMonitor) -> List[Tuple[int, int, int]]:
        """Perform hierarchical clustering and return cluster assignments"""
        monitor.log_stats("Starting L1 Clustering")

        # Level 1 clustering (50 clusters)
        kmeans_l1 = KMeans(n_clusters=self.n_clusters_l1, random_state=self.random_state, n_init=10)
        l1_labels = kmeans_l1.fit_predict(embeddings)

        monitor.log_stats("Completed L1 Clustering")

        # Initialize cluster hierarchy
        cluster_assignments = []

        # Level 2 and 3 clustering for each L1 cluster
        for l1_cluster in range(self.n_clusters_l1):
            l1_mask = l1_labels == l1_cluster
            l1_embeddings = embeddings[l1_mask]
            l1_indices = np.where(l1_mask)[0]

            if len(l1_embeddings) <= self.n_clusters_l2:
                # Not enough points for L2 clustering
                for idx in l1_indices:
                    cluster_assignments.append((l1_cluster, 0, 0))
                continue

            # Level 2 clustering
            kmeans_l2 = KMeans(n_clusters=min(self.n_clusters_l2, len(l1_embeddings)),
                               random_state=self.random_state, n_init=10)
            l2_labels = kmeans_l2.fit_predict(l1_embeddings)

            for l2_cluster in range(min(self.n_clusters_l2, len(l1_embeddings))):
                l2_mask = l2_labels == l2_cluster
                l2_embeddings = l1_embeddings[l2_mask]
                l2_indices = l1_indices[l2_mask]

                if len(l2_embeddings) <= self.n_clusters_l3:
                    # Not enough points for L3 clustering
                    for idx in l2_indices:
                        cluster_assignments.append((l1_cluster, l2_cluster, 0))
                    continue

                # Level 3 clustering
                kmeans_l3 = KMeans(n_clusters=min(self.n_clusters_l3, len(l2_embeddings)),
                                   random_state=self.random_state, n_init=10)
                l3_labels = kmeans_l3.fit_predict(l2_embeddings)

                for i, l3_cluster in enumerate(l3_labels):
                    original_idx = l2_indices[i]
                    cluster_assignments.append((l1_cluster, l2_cluster, l3_cluster))

        monitor.log_stats("Completed Hierarchical Clustering")
        return cluster_assignments


class WeaviateManager:
    """Manage Weaviate connection and operations"""

    def __init__(self, host="localhost", port="8080"):
        self.host = host
        self.port = port
        self.client: weaviate.client.Client = None
        self.connect()

    def connect(self):
        """Connect to Weaviate instance"""
        try:
            self.client = weaviate.Client(url=f"http://{self.host}:{self.port}")
            logger.info("Connected to Weaviate successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise

    def create_schema(self):
        """Create Weaviate schema for document chunks"""
        schema = {
            "class": "DocumentChunk",
            "description": "A chunk of text from a document with hierarchical clustering",
            "vectorizer": "none",  # We'll provide embeddings
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The text content of the chunk"
                },
                {
                    "name": "source_file",
                    "dataType": ["string"],
                    "description": "Name of the source JSON file"
                },
                {
                    "name": "chunk_number",
                    "dataType": ["int"],
                    "description": "Sequential number of the chunk"
                },
                {
                    "name": "cluster_l1",
                    "dataType": ["int"],
                    "description": "Level 1 cluster assignment"
                },
                {
                    "name": "cluster_l2",
                    "dataType": ["int"],
                    "description": "Level 2 cluster assignment"
                },
                {
                    "name": "cluster_l3",
                    "dataType": ["int"],
                    "description": "Level 3 cluster assignment"
                },
                {
                    "name": "embedding_dimension",
                    "dataType": ["int"],
                    "description": "Dimension of the embedding vector"
                }
            ]
        }

        # Delete existing schema if it exists
        try:
            self.client.schema.delete_class("DocumentChunk")
            logger.info("Deleted existing DocumentChunk class")
        except:
            pass

        # Create new schema
        self.client.schema.create_class(schema)
        logger.info("Created DocumentChunk schema in Weaviate")

    def batch_insert(self, chunks_data: List[Dict], batch_size: int = 100):
        """Insert chunks in batches"""
        total_inserted = 0

        with self.client.batch as batch:
            batch.batch_size = batch_size

            for chunk_data in chunks_data:
                # Prepare the data object
                data_object = {
                    "content": chunk_data["content"],
                    "source_file": chunk_data["source_file"],
                    "chunk_number": chunk_data["chunk_number"],
                    "cluster_l1": chunk_data["cluster_l1"],
                    "cluster_l2": chunk_data["cluster_l2"],
                    "cluster_l3": chunk_data["cluster_l3"],
                    "embedding_dimension": len(chunk_data["embedding"])
                }

                # Add to batch with embedding
                batch.add_data_object(
                    data_object=data_object,
                    class_name="DocumentChunk",
                    vector=chunk_data["embedding"]
                )

                total_inserted += 1

                if total_inserted % 1000 == 0:
                    logger.info(f"Inserted {total_inserted} chunks so far...")

        logger.info(f"Successfully inserted {total_inserted} chunks into Weaviate")
        return total_inserted


class EmbeddingProcessor:
    """Process JSON files and extract embeddings"""

    def __init__(self, directory_path: str):
        self.directory_path = directory_path
        self.monitor = PerformanceMonitor()

    def load_json_file(self, file_path: str) -> Tuple[str, List[Dict]]:
        """Load a single JSON file and return filename and chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            filename = os.path.basename(file_path)
            chunks = data.get('file_content', [])

            return filename, chunks
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None, []

    def process_files_parallel(self, max_workers: int = 4) -> Tuple[List[np.ndarray], List[Dict]]:
        """Process all JSON files in parallel"""
        json_files = [f for f in os.listdir(self.directory_path) if f.endswith('.json')][:50]
        logger.info(f"Found {len(json_files)} JSON files to process")

        all_embeddings = []
        all_metadata = []
        chunk_counter = 0

        self.monitor.log_stats("Starting File Processing")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file loading tasks
            future_to_file = {
                executor.submit(self.load_json_file, os.path.join(self.directory_path, filename)): filename
                for filename in json_files
            }

            # Process completed tasks
            for future in as_completed(future_to_file):
                filename, chunks = future.result()

                if not chunks:
                    continue

                for chunk_data in chunks:
                    if 'chunk' in chunk_data and 'embedding' in chunk_data:
                        all_embeddings.append(np.array(chunk_data['embedding']))
                        all_metadata.append({
                            'content': chunk_data['chunk'],
                            'source_file': filename,
                            'chunk_number': chunk_counter,
                            'embedding': chunk_data['embedding']
                        })
                        chunk_counter += 1

                logger.info(f"Processed {filename} - Total chunks so far: {len(all_embeddings)}")

        self.monitor.log_stats("Completed File Processing")
        logger.info(f"Loaded {len(all_embeddings)} total chunks from {len(json_files)} files")

        return all_embeddings, all_metadata


def main():
    # Configuration
    JSON_DIRECTORY = "./NaiveRAG_EmbeddedDocuments"  # Change this to your directory path
    WEAVIATE_HOST = "localhost"
    WEAVIATE_PORT = "8080"
    MAX_WORKERS = os.cpu_count()

    # Initialize components
    logger.info("Starting Weaviate Embeddings Storage Process")
    monitor = PerformanceMonitor()

    # Initialize processor
    processor = EmbeddingProcessor(JSON_DIRECTORY)

    # Load and process files
    embeddings, metadata = processor.process_files_parallel(max_workers=MAX_WORKERS)

    if not embeddings:
        logger.error("No embeddings found. Please check your JSON files.")
        return

    # Convert to numpy array for clustering
    embeddings_array = np.vstack(embeddings)
    monitor.log_stats("Converted embeddings to numpy array")

    # Perform hierarchical clustering
    logger.info("Starting hierarchical clustering...")
    clusterer = HierarchicalClusterer()
    cluster_assignments = clusterer.fit_predict(embeddings_array, monitor)

    # Add cluster information to metadata
    for i, (l1, l2, l3) in enumerate(cluster_assignments):
        metadata[i]['cluster_l1'] = l1
        metadata[i]['cluster_l2'] = l2
        metadata[i]['cluster_l3'] = l3

    monitor.log_stats("Completed clustering assignment")

    # Connect to Weaviate and create schema
    logger.info("Connecting to Weaviate...")
    weaviate_manager = WeaviateManager(WEAVIATE_HOST, WEAVIATE_PORT)
    weaviate_manager.create_schema()

    monitor.log_stats("Created Weaviate schema")

    # Insert data into Weaviate
    logger.info("Inserting data into Weaviate...")
    total_inserted = weaviate_manager.batch_insert(metadata)

    monitor.log_stats("Completed Weaviate insertion")

    # Final statistics
    final_stats = monitor.get_current_stats()
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Total chunks processed: {len(metadata)}")
    logger.info(f"Total chunks inserted: {total_inserted}")
    logger.info(f"Total processing time: {final_stats['elapsed_time']:.2f} seconds")
    logger.info(f"Peak memory usage: {final_stats['total_memory_mb']:.1f} MB")

    if 'gpu_memory_used_mb' in final_stats:
        logger.info(f"GPU memory used: {final_stats['gpu_memory_used_mb']}/{final_stats['gpu_memory_total_mb']} MB")

    # Cluster distribution statistics
    cluster_l1_counts = {}
    for item in metadata:
        l1 = item['cluster_l1']
        cluster_l1_counts[l1] = cluster_l1_counts.get(l1, 0) + 1

    logger.info(f"Level 1 clusters created: {len(cluster_l1_counts)}")
    logger.info(f"Average chunks per L1 cluster: {np.mean(list(cluster_l1_counts.values())):.1f}")

    logger.info("Process completed successfully!")


if __name__ == "__main__":
    main()
