import weaviate
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeaviateManager:
    """Manage Weaviate connection and operations"""

    def __init__(self, host="localhost", port="8080"):
        self.host = host
        self.port = port
        self.client = None
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

if __name__ == "__main__":
    weaviate_obj = WeaviateManager()
    weaviate_obj.create_schema()