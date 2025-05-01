import weaviate
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans
import uuid
import os
from tqdm import tqdm


class HierarchicalGraphRAG:
    def __init__(self, weaviate_url: str, optimize_disk: bool = True, num_clusters: int = 10):
        """
        Initialize the Hierarchical Graph RAG system.

        Args:
            weaviate_url (str): URL for Weaviate instance
            optimize_disk (bool): Enable disk optimization features
            num_clusters (int): Number of clusters for K-means clustering
        """
        self.client = weaviate.Client(
            url=weaviate_url,
            additional_headers={
                "X-OpenAI-Api-Key": os.environ.get("OPENAI_API_KEY")  # For OpenAI embeddings if used
            }
        )
        self.optimize_disk = optimize_disk
        self.num_clusters = num_clusters
        self._create_schema()

    def _create_schema(self):
        """Create the Weaviate schema for hierarchical document representation."""

        # Delete existing schema if it exists (for clean initialization)
        try:
            self.client.schema.delete_all()
            print("Cleared existing schema")
        except:
            print("No existing schema to clear")

        # Document class
        document_class = {
            "class": "Document",
            "description": "A document in the collection",
            "vectorizer": "none",  # We'll supply our own vectors
            "properties": [
                {
                    "name": "document_name",
                    "dataType": ["string"],
                    "description": "Name of the document"
                },
                {
                    "name": "scope",
                    "dataType": ["text"],
                    "description": "Section description"
                },
            ]
        }

        # Section class
        section_class = {
            "class": "Section",
            "description": "A major section in a document",
            "vectorizer": "none",
            "properties": [
                {
                    "name": "title",
                    "dataType": ["string"],
                    "description": "Section title"
                },
                {
                    "name": "description",
                    "dataType": ["text"],
                    "description": "Section description"
                },
                {
                    "name": "summary",
                    "dataType": ["text"],
                    "description": "Section summary"
                },
                {
                    "name": "cluster_id",
                    "dataType": ["int"],
                    "description": "Cluster ID for this section"
                }
            ]
        }

        # Subsection class
        subsection_class = {
            "class": "Subsection",
            "description": "A subsection within a document section",
            "vectorizer": "none",
            "properties": [
                {
                    "name": "title",
                    "dataType": ["string"],
                    "description": "Subsection title"
                },
                {
                    "name": "description",
                    "dataType": ["text"],
                    "description": "Subsection description"
                },
                {
                    "name": "summary",
                    "dataType": ["text"],
                    "description": "Subsection summary"
                },
                {
                    "name": "text_content",
                    "dataType": ["text"],
                    "description": "Complete text of the subsection"
                },
                {
                    "name": "cluster_id",
                    "dataType": ["int"],
                    "description": "Cluster ID for this subsection"
                }
            ]
        }

        # Subsubsection class
        subsubsection_class = {
            "class": "Subsubsection",
            "description": "A subsubsection within a subsection",
            "vectorizer": "none",
            "properties": [
                {
                    "name": "title",
                    "dataType": ["string"],
                    "description": "Subsubsection title"
                },
                {
                    "name": "text_content",
                    "dataType": ["text"],
                    "description": "Complete text of the subsubsection"
                },
                {
                    "name": "cluster_id",
                    "dataType": ["int"],
                    "description": "Cluster ID for this subsubsection"
                }
            ]
        }

        # Chunk class
        chunk_class = {
            "class": "Chunk",
            "description": "A text chunk from a document",
            "vectorizer": "none",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Chunk content"
                },
                {
                    "name": "cluster_id",
                    "dataType": ["int"],
                    "description": "Cluster ID for this chunk"
                }
            ]
        }


        # Add all classes to schema
        for class_def in [document_class, section_class, subsection_class,
                          subsubsection_class, chunk_class]:
            try:
                self.client.schema.create_class(class_def)
                print(f"Created class: {class_def['class']}")
            except Exception as e:
                print(f"Error creating class {class_def['class']}: {e}")

    def _cluster_embeddings(self, embeddings: List[List[float]]) -> List[int]:
        """
        Cluster embeddings using K-means.

        Args:
            embeddings: List of embedding vectors

        Returns:
            List of cluster IDs
        """
        if not embeddings or len(embeddings) < self.num_clusters:
            # Return 0 as cluster ID if not enough data for clustering
            return [0] * len(embeddings)

        # Convert to numpy array for K-means
        embeddings_np = np.array(embeddings)

        # Determine optimal number of clusters (capped by self.num_clusters)
        n_clusters = min(self.num_clusters, len(embeddings))

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_ids = kmeans.fit_predict(embeddings_np)

        return cluster_ids.tolist()

    def ingest_documents(self, json_data: List[Dict[str, Any]]):
        """
        Ingest documents from the JSON structure into Weaviate.

        Args:
            json_data: List of document data in the specified JSON structure
        """
        # Process all documents
        for doc_data in tqdm(json_data, desc="Processing documents"):
            doc_id = self._add_document(doc_data)

            # Collect all section embeddings for clustering
            section_embeddings = []
            for section in doc_data["content"]:
                if section.get("description",""):
                    section_embeddings.append(section["description"]["embedding"])

            # Cluster sections if we have enough data
            section_cluster_ids = self._cluster_embeddings(section_embeddings)

            # Process sections
            for i, section in enumerate(tqdm(doc_data["content"], desc="Processing sections")):
                cluster_id = section_cluster_ids[i] if i < len(section_cluster_ids) else 0
                section_id = self._add_section(section, doc_id, cluster_id)


                # Collect subsection embeddings for clustering
                subsection_embeddings = []
                for subsection in section.get("subsections", []):
                    if subsection.get("title",""):
                        subsection_embeddings.append(subsection["title"]["embedding"])

                # Cluster subsections
                subsection_cluster_ids = self._cluster_embeddings(subsection_embeddings)

                # Process subsections
                for j, subsection in enumerate(section.get("subsections", [])):
                    cluster_id = subsection_cluster_ids[j] if j < len(subsection_cluster_ids) else 0
                    subsection_id = self._add_subsection(subsection, section_id, cluster_id)

                    # Collect chunk embeddings for clustering
                    chunk_embeddings = []
                    for chunk in subsection.get("chunks", []):
                        if "embedding" in chunk:
                            chunk_embeddings.append(chunk["embedding"])

                    # Cluster chunks
                    chunk_cluster_ids = self._cluster_embeddings(chunk_embeddings)

                    # Process chunks
                    for k, chunk in enumerate(subsection.get("chunks", [])):
                        cluster_id = chunk_cluster_ids[k] if k < len(chunk_cluster_ids) else 0
                        self._add_chunk(chunk, subsection_id, cluster_id)

                    # Process subsubsections
                    subsubsection_embeddings = []
                    for subsubsection in subsection.get("subsubsections", []):
                        # Create fake embeddings for clustering if not available
                        # In real application, you'd generate embeddings for titles
                        subsubsection_embeddings.append([0] * 10)  # placeholder

                    # Cluster subsubsections
                    subsubsection_cluster_ids = self._cluster_embeddings(subsubsection_embeddings)

                    for l, subsubsection in enumerate(subsection.get("subsubsections", [])):
                        cluster_id = subsubsection_cluster_ids[l] if l < len(subsubsection_cluster_ids) else 0
                        subsubsection_id = self._add_subsubsection(subsubsection, subsection_id, cluster_id)

                        # Process chunks in subsubsection
                        chunk_embeddings = []
                        for chunk in subsubsection.get("chunks", []):
                            if "embedding" in chunk:
                                chunk_embeddings.append(chunk["embedding"])

                        # Cluster chunks
                        chunk_cluster_ids = self._cluster_embeddings(chunk_embeddings)

                        for m, chunk in enumerate(subsubsection.get("chunks", [])):
                            cluster_id = chunk_cluster_ids[m] if m < len(chunk_cluster_ids) else 0
                            self._add_chunk(chunk, subsubsection_id, cluster_id)

    def _add_document(self, doc_data: Dict[str, Any]) -> str:
        """Add a document to Weaviate and return its UUID."""
        doc_id = str(uuid.uuid4())
        for  section in doc_data.get("sections", []):
            if section.get("title").get("title") == "\tScope" :
                scope = {
                    "text" : section.get("description",{}).get("description",""),
                    "vector" : section.get("description",{}).get("embedding",None)
                }
        self.client.data_object.create(
            class_name="Document",
            data_object={
                "document_name": doc_data["document_name"],
                "scope": scope["text"]
            },
            uuid=doc_id,
            vector = scope["vector"]
        )

        return doc_id

    def _add_section(self, section: Dict[str, Any], doc_id: str, cluster_id: int = 0) -> str:
        """Add a section to Weaviate and link it to its document."""
        section_id = str(uuid.uuid4())

        # Create section object
        self.client.data_object.create(
            class_name="Section",
            data_object={
                "title": section["title"]["title"],
                "description": section["description"].get("description","") if section["description"] else "",
                "summary": section.get("summary", ""),
                "cluster_id": cluster_id
            },
            uuid=section_id,
            vector=section["description"].get("embedding") if section["description"].get("embedding","") else section["title"].get("embedding",None)
        )

        # Create reference from document to section
        self.client.data_object.reference.add(
            from_class_name="Document",
            from_uuid=doc_id,
            from_property_name="hasSections",
            to_class_name="Section",
            to_uuid=section_id
        )

        return section_id

    def _add_subsection(self, subsection: Dict[str, Any], section_id: str, cluster_id: int = 0) -> str:
        """Add a subsection to Weaviate and link it to its section."""
        subsection_id = str(uuid.uuid4())

        # Create subsection object
        self.client.data_object.create(
            class_name="Subsection",
            data_object={
                "title": subsection["title"]["title"] if isinstance(subsection["title"], dict) else subsection["title"],
                "description": subsection["description"]["description"] if "description" in subsection and isinstance(
                    subsection["description"], dict) else "",
                "summary": subsection.get("summary", ""),
                "text_content": subsection.get("text_content", ""),
                "cluster_id": cluster_id
            },
            uuid=subsection_id,
            vector=subsection["title"].get("embedding") if isinstance(subsection["title"], dict) and "embedding" in
                                                           subsection["title"] else None
        )

        # Create reference from section to subsection
        self.client.data_object.reference.add(
            from_class_name="Section",
            from_uuid=section_id,
            from_property_name="hasSubsections",
            to_class_name="Subsection",
            to_uuid=subsection_id
        )

        return subsection_id

    def _add_subsubsection(self, subsubsection: Dict[str, Any], subsection_id: str, cluster_id: int = 0) -> str:
        """Add a subsubsection to Weaviate and link it to its subsection."""
        subsubsection_id = str(uuid.uuid4())

        title = subsubsection.get("title", "")
        if isinstance(title, dict) and "title" in title:
            title = title["title"]

        # Create subsubsection object
        self.client.data_object.create(
            class_name="Subsubsection",
            data_object={
                "title": title,
                "text_content": subsubsection.get("text_content", ""),
                "cluster_id": cluster_id
            },
            uuid=subsubsection_id
        )

        # Create reference from subsection to subsubsection
        self.client.data_object.reference.add(
            from_class_name="Subsection",
            from_uuid=subsection_id,
            from_property_name="hasSubsubsections",
            to_class_name="Subsubsection",
            to_uuid=subsubsection_id
        )

        return subsubsection_id

    def _add_chunk(self, chunk: Dict[str, Any], parent_id: str, cluster_id: int = 0) -> str:
        """Add a text chunk to Weaviate and link it to its parent."""
        chunk_id = str(uuid.uuid4())

        # Get the chunk content
        if "chunk" in chunk:
            content = chunk["chunk"]
        else:
            content = chunk.get("content", "")

        # Create chunk object
        self.client.data_object.create(
            class_name="Chunk",
            data_object={
                "content": content,
                "cluster_id": cluster_id
            },
            uuid=chunk_id,
            vector=chunk.get("embedding")
        )

        # Try to determine the parent class by querying Weaviate
        parent_classes = ["Subsection", "Subsubsection"]
        parent_class = None

        for cls in parent_classes:
            try:
                # Check if this ID exists in this class
                result = self.client.data_object.get_by_id(
                    class_name=cls,
                    uuid=parent_id
                )
                if result:
                    parent_class = cls
                    break
            except:
                continue

        if parent_class:
            # Create reference from parent to chunk
            self.client.data_object.reference.add(
                from_class_name=parent_class,
                from_uuid=parent_id,
                from_property_name="hasChunks",
                to_class_name="Chunk",
                to_uuid=chunk_id
            )

        return chunk_id

    def search_document_by_scope(self, query_vector: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """
        Search for documents by querying sections with 'Scope' in the title.

        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return

        Returns:
            List of matched documents with their scope information
        """
        # Find sections with "Scope" in title
        result = self.client.query.get(
            "Section", ["title", "description", "_additional {id}"]
        ).with_where({
            "path": ["title"],
            "operator": "Like",
            "valueString": "*\tScope*"
        }).with_near_vector({
            "vector": query_vector,
        }).with_limit(limit).do()

        # Get document information for each matching section
        documents = []
        if "data" in result and "Get" in result["data"] and "Section" in result["data"]["Get"]:
            for section in result["data"]["Get"]["Section"]:
                # Find the document that contains this section
                doc_result = self.client.query.get(
                    "Document", ["document_name"]
                ).with_where({
                    "path": ["hasSections", "Section", "id"],
                    "operator": "Equal",
                    "valueString": section["_additional"]["id"]
                }).do()

                if ("data" in doc_result and "Get" in doc_result["data"] and
                        "Document" in doc_result["data"]["Get"] and
                        len(doc_result["data"]["Get"]["Document"]) > 0):
                    doc = doc_result["data"]["Get"]["Document"][0]
                    documents.append({
                        "document_name": doc["document_name"],
                        "scope_title": section["title"],
                        "scope_description": section["description"]
                    })

        return documents

    def search_sections_in_document(self, doc_name: str, query_vector: List[float], limit: int = 5) -> List[
        Dict[str, Any]]:
        """
        Search for sections within a specific document.

        Args:
            doc_name: Name of the document to search within
            query_vector: The query embedding vector
            limit: Maximum number of sections to return

        Returns:
            List of matched sections
        """
        # First find the document
        doc_result = self.client.query.get(
            "Document", ["document_name", "_additional {id}"]
        ).with_where({
            "path": ["document_name"],
            "operator": "Equal",
            "valueString": doc_name
        }).do()

        if ("data" not in doc_result or "Get" not in doc_result["data"] or
                "Document" not in doc_result["data"]["Get"] or
                len(doc_result["data"]["Get"]["Document"]) == 0):
            return []

        doc_id = doc_result["data"]["Get"]["Document"][0]["_additional"]["id"]

        # Then search for sections in this document
        result = self.client.query.get(
            "Section", ["title", "description", "summary", "_additional {id}"]
        ).with_where({
            "path": ["id"],
            "operator": "ContainsAny",
            "valueString": [doc_id]  # This should be adjusted based on how references work
        }).with_near_vector({
            "vector": query_vector,
        }).with_limit(limit).do()

        sections = []
        if "data" in result and "Get" in result["data"] and "Section" in result["data"]["Get"]:
            sections = result["data"]["Get"]["Section"]

        return sections

    def search_subsections_by_section(self, section_id: str, query_vector: List[float], limit: int = 5) -> List[
        Dict[str, Any]]:
        """
        Search for subsections within a specific section.

        Args:
            section_id: ID of the section to search within
            query_vector: The query embedding vector
            limit: Maximum number of subsections to return

        Returns:
            List of matched subsections
        """
        result = self.client.query.get(
            "Subsection", ["title", "description", "summary", "text_content", "_additional {id}"]
        ).with_where({
            "path": ["id"],
            "operator": "ContainsAny",
            "valueString": [section_id]  # This should be adjusted based on how references work
        }).with_near_vector({
            "vector": query_vector,
        }).with_limit(limit).do()

        subsections = []
        if "data" in result and "Get" in result["data"] and "Subsection" in result["data"]["Get"]:
            subsections = result["data"]["Get"]["Subsection"]

        return subsections

    def search_chunks_by_subsection(self, subsection_id: str, query_vector: List[float], limit: int = 10) -> List[
        Dict[str, Any]]:
        """
        Search for text chunks within a specific subsection.

        Args:
            subsection_id: ID of the subsection to search within
            query_vector: The query embedding vector
            limit: Maximum number of chunks to return

        Returns:
            List of matched chunks
        """
        result = self.client.query.get(
            "Chunk", ["content", "_additional {id}"]
        ).with_where({
            "path": ["id"],
            "operator": "ContainsAny",
            "valueString": [subsection_id]  # This should be adjusted based on how references work
        }).with_near_vector({
            "vector": query_vector,
        }).with_limit(limit).do()

        chunks = []
        if "data" in result and "Get" in result["data"] and "Chunk" in result["data"]["Get"]:
            chunks = result["data"]["Get"]["Chunk"]

        return chunks

    def optimize_storage(self):
        """Apply disk space optimization techniques."""
        if not self.optimize_disk:
            return

        # This would implement storage optimization techniques:
        # 1. Vector compression strategies
        # 2. Cluster-based storage (already implemented with clustering)
        # 3. Shard management
        # 4. Index optimization

        print("Running storage optimization...")

        # In a real implementation, you might add:
        # - Vector quantization
        # - Sparse vector representations
        # - Dynamic index rebuilding

        print("Storage optimization complete")

def find_json_files(dir_path):
    """Find all JSON embedding files in the given directory"""
    json_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
            json_files.append(data)
    return json_files


# Example usage
if __name__ == "__main__":
    jsons_dir = "../EmbeddingProcess/EmbeddedDocuments/embedded_files"
    json_files = find_json_files(jsons_dir)

    # Initialize the system
    rag_system = HierarchicalGraphRAG(
        weaviate_url="http://localhost:8080",  # Use your actual Weaviate instance URL
        optimize_disk=True,
        num_clusters=10
    )

    # Ingest the documents
    rag_system.ingest_documents(json_files)

    # Apply storage optimization
    rag_system.optimize_storage()

    # Example search query
    query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]  # Sample query embedding

    # Search for documents by scope
    docs = rag_system.search_document_by_scope(query_embedding)
    print(f"Found {len(docs)} documents with matching scope")

    # Search for sections in a document
    if docs:
        sections = rag_system.search_sections_in_document(docs[0]["document_name"], query_embedding)
        print(f"Found {len(sections)} relevant sections in document {docs[0]['document_name']}")

        # Search for subsections in first section
        if sections:
            subsections = rag_system.search_subsections_by_section(sections[0]["_additional"]["id"], query_embedding)
            print(f"Found {len(subsections)} relevant subsections in section {sections[0