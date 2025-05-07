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
 
        try:
            print("Cleared existing schema")
        except:
            print("No existing schema to clear")
 
        document_class = {
            "class": "Document",
            "description": "A document in the collection",
            "vectorizer": "none",  
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
                    "name": "hasSections",  
                    "dataType": ["Section"],
                    "description": "Sections in this document"
                },
            ]
        }
 
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
     
        for class_def in classes_in_order:
            try:
                self.client.schema.create_class(class_def)
                print(f"Created class: {class_def['class']}")
            except Exception as e:
                print(f"Error creating class {class_def['class']}: {e}")
        
   
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
            
     
        embeddings_np = np.array(embeddings)
        
      
        if len(embeddings) <= 1:
            return [0] * len(embeddings)
            
    
        unique_vectors = np.unique(embeddings_np, axis=0)
        actual_unique_count = len(unique_vectors)
        
   
        n_clusters = min(self.num_clusters, actual_unique_count)
        
  
        if actual_unique_count <= 3:

            labels = np.zeros(len(embeddings), dtype=int)
            for i, vec in enumerate(embeddings_np):
                
                distances = np.sum((unique_vectors - vec)**2, axis=1)
                nearest_unique = np.argmin(distances)
                labels[i] = nearest_unique
            return labels.tolist()
            
    
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_ids = kmeans.fit_predict(embeddings_np)
            return cluster_ids.tolist()
        except Exception as e:
            print(f"Clustering error: {e}. Falling back to simple assignment.")
           
            return [i % max(1, n_clusters) for i in range(len(embeddings))]
    
    def ingest_documents(self, json_data: List[Dict[str, Any]]):
        """
        Ingest documents from the JSON structure into Weaviate.
 
        Args:
            json_data: List of document data in the specified JSON structure
        """
        
        for doc_data in tqdm(json_data, desc="Processing documents"):
            doc_id = self._add_document(doc_data)
 
            
            section_embeddings = []
            for section in doc_data["content"]:
                if section.get("description",""):
                    section_embeddings.append(section["description"]["embedding"])
 
           
            section_cluster_ids = self._cluster_embeddings(section_embeddings)
 
            
            for i, section in enumerate(tqdm(doc_data["content"], desc="Processing sections")):
                cluster_id = section_cluster_ids[i] if i < len(section_cluster_ids) else 0
                section_id = self._add_section(section, doc_id, cluster_id)
                subsection_embeddings = []
                for subsection in section.get("subsections", []):
                    if subsection.get("title",""):
                        subsection_embeddings.append(subsection["title"]["embedding"])
 
                subsection_cluster_ids = self._cluster_embeddings(subsection_embeddings)

                for j, subsection in enumerate(section.get("subsections", [])):
                    cluster_id = subsection_cluster_ids[j] if j < len(subsection_cluster_ids) else 0
                    subsection_id = self._add_subsection(subsection, section_id, cluster_id)
 
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
            "Section", ["title","description", "_additional {id}"]
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
 
    def search_sections_in_document(self, doc_name: str, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for sections within a specific document using vector similarity search.

        Args:
            doc_name: Name of the document to search within
            query_vector: The query embedding vector
            limit: Maximum number of sections to return

        Returns:
            List of matched sections with their metadata
        """
        # First find the document by name
        doc_result = self.client.query.get(
            "Document", ["document_name", "_additional {id}", "hasSections {... on Section {title description summary _additional {id}}}"]
        ).with_where({
            "path": ["document_name"],
            "operator": "Equal",
            "valueString": doc_name
        }).do()

        if ("data" not in doc_result or "Get" not in doc_result["data"] or
                "Document" not in doc_result["data"]["Get"] or
                len(doc_result["data"]["Get"]["Document"]) == 0):
            print(f"Document not found: {doc_name}")
            return []

        doc_id = doc_result["data"]["Get"]["Document"][0]["_additional"]["id"]
        print(f"Found document ID: {doc_id}")

        # Extract all sections from the result
        all_sections = doc_result["data"]["Get"]["Document"][0].get("hasSections", [])
        
        if not all_sections:
            print(f"Document {doc_name} has no sections")
            return []
        
        print(f"Found {len(all_sections)} sections in document {doc_name}")
        
        # Now perform vector search directly on these sections
        # We'll collect the IDs of all sections to filter
        section_ids = [section["_additional"]["id"] for section in all_sections]
        
        # Perform the vector search on Section class, but only within the sections of this document
        result = self.client.query.get(
            "Section", ["title", "description", "summary", "_additional {id}"]
        ).with_where({
            "path": ["_id"],  # Use _id to filter by object IDs
            "operator": "ContainsAny",
            "valueString": section_ids  # Filter only the sections from this document
        }).with_near_vector({
            "vector": query_vector,
        }).with_limit(limit).do()
        
        sections = []
        if "data" in result and "Get" in result["data"] and "Section" in result["data"]["Get"]:
            sections = result["data"]["Get"]["Section"]
            # Add document name to each section for context
            for section in sections:
                section["parent_document"] = doc_name
        
        print(f"Returning {len(sections)} sections that match the vector query")
        return sections
    
    def get_relevant_subsections(self, query_vector: List[float], doc_name: str, section_title: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        First function: Get only the subsection IDs and descriptions.
        
        Args:
            query_vector: The query embedding vector
            doc_name: Name of the document containing the section
            section_title: Title of the section to search within
            limit: Maximum number of subsections to retrieve
            
        Returns:
            List of matched subsections with only ID and description information
        """
        # Step 1: Find the document by name
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
            print(f"Document not found: {doc_name}")
            return []
        
        doc_id = doc_result["data"]["Get"]["Document"][0]["_additional"]["id"]
        
        # Step 2: Find the section by title within this document
        section_result = self.client.query.get(
            "Section", ["title", "_additional {id}"]
        ).with_where({
            "operator": "And",
            "operands": [
                {
                    "path": ["title"],
                    "operator": "Equal",
                    "valueString": section_title
                }
            ]
        }).do()
        
        if ("data" not in section_result or "Get" not in section_result["data"] or
                "Section" not in section_result["data"]["Get"] or
                len(section_result["data"]["Get"]["Section"]) == 0):
            print(f"Section not found: {section_title} in document {doc_name}")
            return []
        
        # Filter sections that belong to our document
        matching_sections = []
        for section in section_result["data"]["Get"]["Section"]:
            # Get document reference for this section
            doc_ref = self.client.query.get(
                "Document", ["document_name"]
            ).with_where({
                "path": ["hasSections", "Section", "id"],
                "operator": "Equal",
                "valueString": section["_additional"]["id"]
            }).do()
            
            if ("data" in doc_ref and "Get" in doc_ref["data"] and
                    "Document" in doc_ref["data"]["Get"] and
                    len(doc_ref["data"]["Get"]["Document"]) > 0):
                if doc_ref["data"]["Get"]["Document"][0]["document_name"] == doc_name:
                    matching_sections.append(section)

        if not matching_sections:
            print(f"No sections with title '{section_title}' found in document '{doc_name}'")
            return []
        
        section_id = matching_sections[0]["_additional"]["id"]
        
        # Step 3: Get ONLY the subsections IDs and descriptions for this section
        subsection_result = self.client.query.get(
            "Subsection", 
            [
                "title", 
                "description", 
                "_additional {id}"
            ]
        ).with_where({
            "operator": "Or",
            "operands": [
                {
                    "path": ["id"],
                    "operator": "Equal",
                    "valueString": section_id
                }
            ]
        }).with_near_vector({
            "vector": query_vector,
        }).with_limit(limit).do()
        
        # If the first query doesn't work, try with reference-based query
        if ("data" not in subsection_result or "Get" not in subsection_result["data"] or
                "Subsection" not in subsection_result["data"]["Get"] or
                len(subsection_result["data"]["Get"]["Subsection"]) == 0):
            
            # Try alternative query using section reference
            subsection_result = self.client.query.get(
                "Subsection", 
                [
                    "title", 
                    "description", 
                    "_additional {id}"
                ]
            ).do()
            
            # Manual filtering after getting all subsections - fallback method
            all_subsections = []
            if "data" in subsection_result and "Get" in subsection_result["data"] and "Subsection" in subsection_result["data"]["Get"]:
                all_subsections = subsection_result["data"]["Get"]["Subsection"]
            
            # Filter and rank subsections based on vector similarity
            relevant_subsections = []
            for subsection in all_subsections:
                relevant_subsections.append(subsection)
            
            # Sort by some criteria (ideally vector similarity)
            relevant_subsections = relevant_subsections[:limit]
            subsection_result = {"data": {"Get": {"Subsection": relevant_subsections}}}
        
        subsections = []
        if "data" in subsection_result and "Get" in subsection_result["data"] and "Subsection" in subsection_result["data"]["Get"]:
            subsections = subsection_result["data"]["Get"]["Subsection"]
            
            # Add parent section information and structure information to each subsection
            for subsection in subsections:
                subsection["parent_section_id"] = section_id
                subsection["parent_section_title"] = section_title
                subsection["parent_document_name"] = doc_name
        
        print(f"Found {len(subsections)} subsections")
        return subsections


    def get_best_chunks_from_subsection(self, subsection_id: str, query_vector: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """
        Second function: Get the best similar chunks from a subsection.
        
        Args:
            subsection_id: ID of the subsection to search within
            query_vector: The query embedding vector
            limit: Maximum number of best chunks to return
            
        Returns:
            List of best matching chunks with their IDs and content
        """
        # First verify that this subsection has direct chunks
        subsection_result = self.client.query.get(
            "Subsection", 
            [
                "title",
                "_additional {id}",
                "hasChunks {... on Chunk {_additional {id}}}"
            ]
        ).with_where({
            "path": ["_id"],
            "operator": "Equal",
            "valueString": subsection_id
        }).do()
        
        if ("data" not in subsection_result or "Get" not in subsection_result["data"] or
                "Subsection" not in subsection_result["data"]["Get"] or
                len(subsection_result["data"]["Get"]["Subsection"]) == 0):
            
            print(f"Subsection not found with ID: {subsection_id}")
        
        subsection_data = subsection_result["data"]["Get"]["Subsection"][0]
        has_chunks = subsection_data.get("hasChunks", [])
        
        if not has_chunks:
            print(f"Subsection '{subsection_id}' has no direct chunks")
            
        
        # Get chunk IDs from the subsection
        chunk_ids = [chunk["_additional"]["id"] for chunk in has_chunks]
        
        # Search for best matching chunks using vector similarity
        result = self.client.query.get(
            "Chunk", 
            [
                "content", 
                "_additional {id}"
            ]
        ).with_where({
            "path": ["_id"],
            "operator": "ContainsAny",
            "valueString": chunk_ids
        }).with_near_vector({
            "vector": query_vector,
        }).with_limit(limit).do()
        
        chunks = []
        if "data" in result and "Get" in result["data"] and "Chunk" in result["data"]["Get"]:
            chunks = result["data"]["Get"]["Chunk"]
            
            # Add parent subsection ID to each chunk
            for chunk in chunks:
                chunk["parent_subsection_id"] = subsection_id
        
        print(f"Found {len(chunks)} best matching chunks in subsection {subsection_id}")
        return chunks


    def get_best_subsubsections(self, subsection_id: str, query_vector: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """
        Third function: Get the best similar subsubsections from a subsection.
        
        Args:
            subsection_id: ID of the subsection to search within
            query_vector: The query embedding vector
            limit: Maximum number of best subsubsections to return
            
        Returns:
            List of best matching subsubsections with their metadata
        """
        # First verify that this subsection has subsubsections
        subsection_result = self.client.query.get(
            "Subsection", 
            [
                "title",
                "_additional {id}",
                "hasSubsubsections {... on Subsubsection {_additional {id}}}"
            ]
        ).with_where({
            "path": ["_id"],
            "operator": "Equal",
            "valueString": subsection_id
        }).do()
        
        if ("data" not in subsection_result or "Get" not in subsection_result["data"] or
                "Subsection" not in subsection_result["data"]["Get"] or
                len(subsection_result["data"]["Get"]["Subsection"]) == 0):
            print(f"Subsection not found with ID: {subsection_id}")

        
        subsection_data = subsection_result["data"]["Get"]["Subsection"][0]
        subsection_title = subsection_data.get("title", "")
        has_subsubsections = subsection_data.get("hasSubsubsections", [])
        
        if not has_subsubsections:
            print(f"Subsection '{subsection_id}' has no subsubsections")
        
        # Get subsubsection IDs from the subsection
        subsubsection_ids = [subsubsection["_additional"]["id"] for subsubsection in has_subsubsections]
        
        # Search for best matching subsubsections using vector similarity
        result = self.client.query.get(
            "Subsubsection", 
            [
                "title", 
                "text_content", 
                "_additional {id}",
                "hasChunks {... on Chunk {content _additional {id}}}"
            ]
        ).with_where({
            "path": ["_id"],
            "operator": "ContainsAny",
            "valueString": subsubsection_ids
        }).with_near_vector({
            "vector": query_vector,
        }).with_limit(limit).do()
        
        subsubsections = []
        if "data" in result and "Get" in result["data"] and "Subsubsection" in result["data"]["Get"]:
            subsubsections = result["data"]["Get"]["Subsubsection"]
            
            # Add parent subsection info to each subsubsection
            for subsubsection in subsubsections:
                subsubsection["parent_subsection_id"] = subsection_id
                subsubsection["parent_subsection_title"] = subsection_title
                
                # Count chunks for each subsubsection
                subsubsection["chunk_count"] = len(subsubsection.get("hasChunks", []))
        
        print(f"Found {len(subsubsections)} best matching subsubsections")
        return subsubsections


    def hierarchical_search(self, query_vector: List[float], max_documents: int = 3, 
                                max_sections_per_doc: int = 2, max_subsections_per_section: int = 2,
                                max_subsubsections_per_subsection: int = 2, max_chunks_per_container: int = 3) -> Dict[str, Any]:
        '''
        Enhanced hierarchical search using the three specialized functions.
        
        Args:
            query_vector: The query embedding vector
            max_documents: Maximum number of documents to retrieve
            max_sections_per_doc: Maximum number of sections to retrieve per document
            max_subsections_per_section: Maximum number of subsections to retrieve per section
            max_subsubsections_per_subsection: Maximum number of subsubsections to retrieve per subsection
            max_chunks_per_container: Maximum number of chunks to retrieve per container
                
        Returns:
            Dictionary containing results at each hierarchical level with parent-child relationships
        '''
        results = {"documents": [], "sections": [], "subsections": [], "subsubsections": [], "chunks": []}
        
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
                
                # Step 3: For each section, use the first function to get subsections
                section_title = section["title"]
                subsections = self.get_relevant_subsections(
                    query_vector, doc_name, section_title, limit=max_subsections_per_section
                )
                
                # Process each subsection
                for subsection in subsections:
                    # Add parent info to subsection
                    subsection_with_context = subsection.copy()
                    subsection_with_context["parent_section"] = section_title
                    subsection_with_context["parent_document"] = doc_name
                    results["subsections"].append(subsection_with_context)
                    
                    subsection_id = subsection["_additional"]["id"]
                    
                    # Step 4: Check for direct chunks using the second function
                    chunks = self.get_best_chunks_from_subsection(
                        subsection_id, query_vector, limit=max_chunks_per_container
                    )
                    
                    for chunk in chunks:
                        # Add parent info to chunk
                        chunk_with_context = chunk.copy()
                        chunk_with_context["parent_subsection_title"] = subsection["title"]
                        chunk_with_context["parent_section"] = section_title
                        chunk_with_context["parent_document"] = doc_name
                        results["chunks"].append(chunk_with_context)
                    
                    # Step 5: Check for subsubsections using the third function
                    subsubsections = self.get_best_subsubsections(
                        subsection_id, query_vector, limit=max_subsubsections_per_subsection
                    )
                    
                    for subsubsection in subsubsections:
                        # Add parent info to subsubsection
                        subsubsection_with_context = subsubsection.copy()
                        subsubsection_with_context["parent_subsection"] = subsection["title"]
                        subsubsection_with_context["parent_section"] = section_title
                        subsubsection_with_context["parent_document"] = doc_name
                        results["subsubsections"].append(subsubsection_with_context)
                        
                        # Get chunks from subsubsection
                        subsubsection_id = subsubsection["_additional"]["id"]
                        chunks = self.get_best_chunks_from_subsection(
                            subsubsection_id, query_vector, limit=max_chunks_per_container
                        )
                        
                        for chunk in chunks:
                            # Add parent info to chunk
                            chunk_with_context = chunk.copy()
                            chunk_with_context["parent_subsubsection"] = subsubsection["title"]
                            chunk_with_context["parent_subsection"] = subsection["title"]
                            chunk_with_context["parent_section"] = section_title
                            chunk_with_context["parent_document"] = doc_name
                            results["chunks"].append(chunk_with_context)
    
        return results  
    
 
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