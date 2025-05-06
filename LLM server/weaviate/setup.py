import weaviate
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans
import uuid
import os
from tqdm import tqdm
 
 
class HierarchicalGraphRAG:
    def __init__(self, weaviate_url: str, num_clusters: int = 10):
        """
        Initialize the Hierarchical Graph RAG system.
 
        Args:
            weaviate_url (str): URL for Weaviate instance
            num_clusters (int): Number of clusters for K-means clustering
        """
        self.client = weaviate.Client(
            url=weaviate_url,
 
        )
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
                {
                    "name": "hasSections",  # Add this reference property
                    "dataType": ["Section"],
                    "description": "Sections in this document"
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
                },
                {
                    "name": "hasSubsections",
                    "dataType": ["Subsection"],
                    "description": "References to subsections in this section"
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
                },
                {
                    "name": "hasSubsubsections",
                    "dataType": ["Subsubsection"],
                    "description": "References to subsubsections in this subsection"
                },
                {
                    "name": "hasChunks",
                    "dataType": ["Chunk"],
                    "description": "References to chunks in this subsection"
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
                },
                {
                    "name": "hasChunks",
                    "dataType": ["Chunk"],
                    "description": "References to chunks in this subsubsection"
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
 
        classes_in_order = [chunk_class, subsubsection_class, subsection_class, section_class, document_class]
        # Add all classes to schema
        for class_def in classes_in_order:
            try:
                self.client.schema.create_class(class_def)
                print(f"Created class: {class_def['class']}")
            except Exception as e:
                print(f"Error creating class {class_def['class']}: {e}")
        
        # Validate the schema after creation
        schema = self.client.schema.get()
        print("Schema validation complete")
        
    def _cluster_embeddings(self, embeddings: List[List[float]]) -> List[int]:
        """
        Cluster embeddings using K-means with improved handling of edge cases.

        Args:
            embeddings: List of embedding vectors

        Returns:
            List of cluster IDs
        """
        if not embeddings:
            return []
            
        # Convert to numpy array for K-means
        embeddings_np = np.array(embeddings)
        
        # Check for minimum data points before clustering
        if len(embeddings) <= 1:
            return [0] * len(embeddings)
            
        # Check for duplicate vectors
        unique_vectors = np.unique(embeddings_np, axis=0)
        actual_unique_count = len(unique_vectors)
        
        # Determine optimal number of clusters based on unique vectors
        n_clusters = min(self.num_clusters, actual_unique_count)
        
        # If we have very few unique vectors, just assign them to clusters directly
        if actual_unique_count <= 3:
            # Map each vector to its nearest unique vector
            labels = np.zeros(len(embeddings), dtype=int)
            for i, vec in enumerate(embeddings_np):
                # Find the index of the closest unique vector
                distances = np.sum((unique_vectors - vec)**2, axis=1)
                nearest_unique = np.argmin(distances)
                labels[i] = nearest_unique
            return labels.tolist()
            
        # Apply K-means clustering with reduced clusters if needed
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_ids = kmeans.fit_predict(embeddings_np)
            return cluster_ids.tolist()
        except Exception as e:
            print(f"Clustering error: {e}. Falling back to simple assignment.")
            # Simple fallback: assign equal clusters
            return [i % max(1, n_clusters) for i in range(len(embeddings))]
    
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
        
        # Initialize scope BEFORE the loop - this is the critical part!
        scope = {
            "text": "",
            "vector": None
        }
        
        # Now look for a section with title "\tScope"
        for section in doc_data.get("content", []):
            # Handle case where title is a string
            if isinstance(section.get("title", ""), str):
                title_text = section.get("title", "")
            # Handle case where title is a dictionary
            elif isinstance(section.get("title", {}), dict):
                title_text = section.get("title", {}).get("title", "")
            else:
                title_text = ""
                
            if "\tScope" in title_text:
                # Also handle the case where description can be a string or dict
                if isinstance(section.get("description", ""), str):
                    description_text = section.get("description", "")
                    description_vector = None
                elif isinstance(section.get("description", {}), dict):
                    description_text = section.get("description", {}).get("description", "")
                    description_vector = section.get("description", {}).get("embedding", None)
                else:
                    description_text = ""
                    description_vector = None
                    
                scope = {
                    "text": description_text,
                    "vector": description_vector
                }
                break 
                
        self.client.data_object.create(
            class_name="Document",
            data_object={
                "document_name": doc_data["document_name"],
                "scope": scope["text"]
            },
            uuid=doc_id,
            vector=scope["vector"]
        )
        
        return doc_id
    def _add_section(self, section: Dict[str, Any], doc_id: str, cluster_id: int = 0) -> str:
        """Add a subsection to Weaviate and link it to its section."""
        subsection_id = str(uuid.uuid4())

        """Add a section to Weaviate and link it to its document."""
        section_id = str(uuid.uuid4())
        if isinstance(section.get("title", ""), str):
            title = section.get("title", "")
            title_embedding = None
        elif isinstance(section.get("title", {}), dict):
            title = section.get("title", {}).get("title", "")
            title_embedding = section.get("title", {}).get("embedding")
        else:
            title = ""
            title_embedding = None

        if isinstance(section.get("description", ""), str):
            description = section.get("description", "")
            description_embedding = None
        elif isinstance(section.get("description", {}), dict):
            description = section.get("description", {}).get("description", "")
            description_embedding = section.get("description", {}).get("embedding")
        else:
            description = ""
            description_embedding = None
    
        # Get embedding from description or fallback to title
        vector = None
        if description_embedding:
            vector = description_embedding
        elif title_embedding:
            vector = title_embedding

        # Create section object
        self.client.data_object.create(
            class_name="Section",
            data_object={
                "title": title,
                "description": description,
                "summary": section.get("summary", ""),
                "cluster_id": cluster_id
            },
            uuid=section_id,
            vector=vector
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
        # Handle the case where title can be either a string or a dictionary
        if isinstance(subsection.get("title", ""), str):
            title = subsection.get("title", "")
            title_embedding = None
        elif isinstance(subsection.get("title", {}), dict):
            title = subsection.get("title", {}).get("title", "")
            title_embedding = subsection.get("title", {}).get("embedding")
        else:
            title = ""
            title_embedding = None
            
        # Handle the case where description can be either a string or a dictionary
        if isinstance(subsection.get("description", ""), str):
            description = subsection.get("description", "")
            description_embedding = None
        elif isinstance(subsection.get("description", {}), dict):
            description = subsection.get("description", {}).get("description", "")
            description_embedding = subsection.get("description", {}).get("embedding")
        else:
            description = ""
            description_embedding = None
        
        # Choose the appropriate vector
        vector = None
        if title_embedding:
            vector = title_embedding
        elif description_embedding:
            vector = description_embedding

        # Create subsection object
        self.client.data_object.create(
            class_name="Subsection",
            data_object={
                "title": title,
                "description": description,
                "summary": subsection.get("summary", ""),
                "text_content": subsection.get("text_content", ""),
                "cluster_id": cluster_id
            },
            uuid=subsection_id,
            vector=vector
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
        
        # Handle the case where title can be either a string or a dictionary
        if isinstance(subsubsection.get("title", ""), str):
            title = subsubsection.get("title", "")
            title_embedding = None
        elif isinstance(subsubsection.get("title", {}), dict):
            title = subsubsection.get("title", {}).get("title", "")
            title_embedding = subsubsection.get("title", {}).get("embedding")
        else:
            title = ""
            title_embedding = None
        
        # Create subsubsection object
        self.client.data_object.create(
            class_name="Subsubsection",
            data_object={
                "title": title,
                "text_content": subsubsection.get("text_content", ""),
                "cluster_id": cluster_id
            },
            uuid=subsubsection_id,
            vector=title_embedding
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
        print(doc_result)
 
        if ("data" not in doc_result or "Get" not in doc_result["data"] or
                "Document" not in doc_result["data"]["Get"] or
                len(doc_result["data"]["Get"]["Document"]) == 0):
            return []
 
        doc_id = doc_result["data"]["Get"]["Document"][0]["_additional"]["id"]
 
        # Then search for sections in this document
        result = self.client.query.get(
            "subsections", ["title", "description", "summary", "_additional {id}"]
        ).with_where({
            "path": ["id"],
            "operator": "ContainsAny",
            "valueString": [doc_id]  # This should be adjusted based on how references work
        }).with_near_vector({
            "vector": query_vector,
        }).with_limit(limit).do()
        print(f"those are the section for the {doc_name} \n {result}")
 
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
    
    def hierarchical_search(self, query_vector: List[float], max_documents: int = 3, 
                   max_sections_per_doc: int = 2, max_subsections_per_section: int = 2,
                   max_chunks_per_subsection: int = 3) -> Dict[str, Any]:
        '''
        Perform a complete hierarchical search using the query vector.
        
        Args:
            query_vector: The query embedding vector
            max_documents: Maximum number of documents to retrieve
            max_sections_per_doc: Maximum number of sections to retrieve per document
            max_subsections_per_section: Maximum number of subsections to retrieve per section
            max_chunks_per_subsection: Maximum number of chunks to retrieve per subsection
            
        Returns:
            Dictionary co  ntaining results at each hierarchical level with parent-child relationships
        '''
        results = {"documents": [], "sections": [], "subsections": [], "chunks": []}
        
        # Step 1: Find relevant documents using existing search_document_by_scope method
        docs = self.search_document_by_scope(query_vector, limit=max_documents)
        results["documents"] = docs
        
        # Step 2: For each document, find relevant sections
        for doc in docs:
            doc_name = doc["document_name"]
            sections = self.search_sections_in_document(doc_name, query_vector, limit=max_sections_per_doc)
            
            for section in sections:
                # Add parent document info to section
                section_with_context = section.copy()
                section_with_context["parent_document"] = doc_name
                results["sections"].append(section_with_context)
                
                # Step 3: For each section, find relevant subsections
                section_id = section["_additional"]["id"]
                subsections = self.search_subsections_by_section(section_id, query_vector, 
                                                            limit=max_subsections_per_section)
                
                for subsection in subsections:
                    # Add parent info to subsection
                    subsection_with_context = subsection.copy()
                    subsection_with_context["parent_section"] = section["title"]
                    subsection_with_context["parent_document"] = doc_name
                    results["subsections"].append(subsection_with_context)
                    
                    # Step 4: For each subsection, find relevant chunks
                    subsection_id = subsection["_additional"]["id"]
                    chunks = self.search_chunks_by_subsection(subsection_id, query_vector, 
                                                        limit=max_chunks_per_subsection)
                    
                    for chunk in chunks:
                        # Add parent info to chunk
                        chunk_with_context = chunk.copy()
                        chunk_with_context["parent_subsection"] = subsection["title"]
                        chunk_with_context["parent_section"] = section["title"]
                        chunk_with_context["parent_document"] = doc_name
                        results["chunks"].append(chunk_with_context)
        
        return results  # Just
    
 
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
    jsons_dir = r"..\EmbeddingProcess\EmbeddedDocuments\embedded_files"
    json_files = find_json_files(jsons_dir)
 
    # Initialize the system
    rag_system = HierarchicalGraphRAG(
        weaviate_url="http://localhost:8080",  # Use your actual Weaviate instance URL
        num_clusters=10
    )
 
    # Ingest the documents
    rag_system.ingest_documents(json_files)