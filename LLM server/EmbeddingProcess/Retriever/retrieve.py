import argparse
import json
from typing import List, Dict, Any
import sys
import traceback
import importlib.util

import requests

# from setup import HierarchicalGraphRAG

# Check if xformers is installed
XFORMERS_AVAILABLE = importlib.util.find_spec("xformers") is not None

import weaviate
import json
from typing import List, Dict, Any
import os


class HierarchicalGraphRAG:
    def __init__(self, weaviate_url: str, num_clusters: int = 5):
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
            "Document", ["document_name", "_additional {id}",
                         "hasSections {... on Section {title description summary _additional {id}}}"]
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

        # Keywords to exclude (case-insensitive)
        excluded_titles = ["\tScope", "\tReferences", "Introduction", "Foreword"]

        # Filter sections that do NOT match the excluded titles
        filtered_sections = [
            section for section in all_sections
            if
            section.get("title") and all(keyword not in section["title"] for keyword in excluded_titles)
        ]

        if not filtered_sections:
            print(f"Document {doc_name} has no sections")
            return []

        print(f"Found {len(filtered_sections)} sections in document {doc_name}")

        # Now perform vector search directly on these sections
        # We'll collect the IDs of all sections to filter
        section_ids = [section["_additional"]["id"] for section in filtered_sections]

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

    def get_relevant_subsections(self, query_vector: List[float], doc_name: str, section_title: str, limit: int = 5) -> \
            List[Dict[str, Any]]:
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
            if "data" in subsection_result and "Get" in subsection_result["data"] and "Subsection" in \
                    subsection_result["data"]["Get"]:
                all_subsections = subsection_result["data"]["Get"]["Subsection"]

            # Filter and rank subsections based on vector similarity
            relevant_subsections = []
            for subsection in all_subsections:
                relevant_subsections.append(subsection)

            # Sort by some criteria (ideally vector similarity)
            relevant_subsections = relevant_subsections[:limit]
            subsection_result = {"data": {"Get": {"Subsection": relevant_subsections}}}

        subsections = []
        if "data" in subsection_result and "Get" in subsection_result["data"] and "Subsection" in \
                subsection_result["data"]["Get"]:
            subsections = subsection_result["data"]["Get"]["Subsection"]

            # Add parent section information and structure information to each subsection
            for subsection in subsections:
                subsection["parent_section_id"] = section_id
                subsection["parent_section_title"] = section_title
                subsection["parent_document_name"] = doc_name

        print(f"Found {len(subsections)} subsections")
        return subsections

    def get_best_chunks_from_subsection(self, subsection_id: str, query_vector: List[float], limit: int = 3) -> List[
        Dict[str, Any]]:
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
        # print(f"subsection results are : {subsection_result}")
        subsection_result_example = {
            'data':
                {'Get':
                    {'Subsection':
                        [
                            {'_additional':
                                 {'id': '0004098f-a8f4-49bb-bb26-ac1f56493a40'},
                             'hasChunks':
                                 [
                                     {'_additional':
                                          {'id': '42ebeef8-7759-4188-b0a7-e9233bbed1f5'}
                                      }
                                 ],
                             'title': '6.3.2\tTotal power dynamic range'
                             }
                        ]
                    }
                }
        }

        if ("data" not in subsection_result or "Get" not in subsection_result["data"] or
                "Subsection" not in subsection_result["data"]["Get"] or
                len(subsection_result["data"]["Get"]["Subsection"]) == 0):
            print(f"Subsection not found with ID: {subsection_id}")
            return [{}]

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

    def get_best_subsubsections(self, subsection_id: str, query_vector: List[float], limit: int = 3) -> List[
        Dict[str, Any]]:
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
            return [{}]

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

    from typing import List, Dict, Any

    def hierarchical_search(self, query_vector: List[float], max_documents: int = 3,
                            max_sections_per_doc: int = 2, max_subsections_per_section: int = 2,
                            max_subsubsections_per_subsection: int = 2, max_chunks_per_container: int = 3) -> Dict[
        str, Any]:
        """
        Performs a hierarchical search across documents, sections, subsections, subsubsections, and chunks,
        maintaining parent-child relationships at each level. Results are limited by configurable maximums per level.

        Args:
            query_vector: Embedding vector representing the search query.
            max_documents: Maximum number of top-level documents to retrieve (default: 3).
            max_sections_per_doc: Maximum sections to retrieve per document (default: 2).
            max_subsections_per_section: Maximum subsections to retrieve per section (default: 2).
            max_subsubsections_per_subsection: Maximum subsubsections to retrieve per subsection (default: 2).
            max_chunks_per_container: Maximum text chunks to retrieve per subsection/subsubsection (default: 3).

        Returns:
            A dictionary containing lists of results at each hierarchical level. Each entry includes metadata
            and parent references for reconstructing hierarchical relationships. Structure:
            {
                "documents": List[dict],       # Top-level documents
                "sections": List[dict],        # Sections with 'parent_document' field
                "subsections": List[dict],     # Subsections with 'parent_section' and 'parent_document'
                "subsubsections": List[dict],  # Subsubsections with 'parent_subsection', 'parent_section', etc.
                "chunks": List[dict]           # Chunks with full parent hierarchy up to document level
            }
        """
        results = {
            "documents": [],
            "sections": [],
            "subsections": [],
            "subsubsections": [],
            "chunks": []
        }

        # Retrieve top-level documents
        documents = self.search_document_by_scope(query_vector, limit=max_documents)
        results["documents"] = documents

        for document in documents:
            doc_name = document["document_name"]

            # Retrieve sections within the document
            sections = self.search_sections_in_document(doc_name, query_vector, limit=max_sections_per_doc)
            for section in sections:
                section_title = section["title"]
                # Annotate section with parent document
                results["sections"].append({
                    **section.copy(),
                    "parent_document": doc_name
                })

                # Retrieve subsections within the section
                subsections = self.get_relevant_subsections(
                    query_vector, doc_name, section_title, limit=max_subsections_per_section
                )
                for subsection in subsections:
                    subsection_title = subsection["title"]
                    subsection_id = subsection["_additional"]["id"]
                    # Annotate subsection with parent hierarchy
                    enriched_subsection = {
                        **subsection.copy(),
                        "parent_section": section_title,
                        "parent_document": doc_name
                    }
                    results["subsections"].append(enriched_subsection)

                    # Retrieve chunks directly under the subsection
                    chunks = self.get_best_chunks_from_subsection(
                        subsection_id, query_vector, limit=max_chunks_per_container
                    )
                    for chunk in chunks:
                        # Annotate chunk with full parent hierarchy
                        results["chunks"].append({
                            **chunk.copy(),
                            "parent_subsection": subsection_title,
                            "parent_section": section_title,
                            "parent_document": doc_name
                        })

                    # Retrieve subsubsections within the subsection
                    subsubsections = self.get_best_subsubsections(
                        subsection_id, query_vector, limit=max_subsubsections_per_subsection
                    )
                    if subsubsections == [{}]:
                        continue
                    # print(f"Retrieving subsection from those subsubsections :\n{subsubsections}")
                    for subsubsection in subsubsections:
                        subsubsection_title = subsubsection["title"]
                        subsubsection_id = subsubsection["_additional"]["id"]
                        # Annotate subsubsection with parent hierarchy
                        enriched_subsubsection = {
                            **subsubsection.copy(),
                            "parent_subsection": subsection_title,
                            "parent_section": section_title,
                            "parent_document": doc_name
                        }
                        results["subsubsections"].append(enriched_subsubsection)

                        # Retrieve chunks under the subsubsection
                        chunks = self.get_best_chunks_from_subsection(
                            subsubsection_id, query_vector, limit=max_chunks_per_container
                        )
                        for chunk in chunks:
                            # Annotate chunk with full parent hierarchy
                            results["chunks"].append({
                                **chunk.copy(),
                                "parent_subsubsection": subsubsection_title,
                                "parent_subsection": subsection_title,
                                "parent_section": section_title,
                                "parent_document": doc_name
                            })

        return results


class HierarchicalSearchEngine:
    def __init__(
            self,
            weaviate_url: str = "http://localhost:8080",
            num_clusters: int = 5,
    ):
        """
        Initialize the search engine with embedding model and RAG system.

        Args:
            weaviate_url: URL of the Weaviate instance
            model_name: Name of the sentence transformer model
            num_clusters: Number of clusters for the RAG system

        """
        # Initialize the RAG system
        try:
            print(f"Connecting to Weaviate at: {weaviate_url}")
            self.rag_system = HierarchicalGraphRAG(
                weaviate_url=weaviate_url,
                num_clusters=num_clusters
            )
            print("Search engine initialized successfully!")
        except Exception as e:
            print(f"Error connecting to Weaviate: {e}")
            raise

    def search(
            self,
            query_embedding: list[float],
            max_documents: int = 2,
            max_sections_per_doc: int = 2,
            max_subsections_per_section: int = 2,
            max_subsubsections_per_subsection: int = 2,
            max_chunks_per_container: int = 3,
            verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a hierarchical search with a given query using the initialized components.

        Args:
            query_embedding: The search vector query
            max_documents: Maximum number of documents to retrieve
            max_sections_per_doc: Maximum number of sections per document
            max_subsections_per_section: Maximum number of subsections per section
            max_chunks_per_subsection: Maximum number of chunks per subsection
            verbose: Whether to print additional information

        Returns:
            Dictionary containing search results
        """
        try:
            # Get query embedding with the s2p_query prompt

            # Perform hierarchical search
            print("Performing hierarchical search...")
            results = self.rag_system.hierarchical_search(
                query_vector=query_embedding,
                max_documents=max_documents,
                max_sections_per_doc=max_sections_per_doc,
                max_subsections_per_section=max_subsections_per_section,
                max_subsubsections_per_subsection=max_subsubsections_per_subsection,
                max_chunks_per_container=max_chunks_per_container,
            )

            # Print summary statistics
            print("\n----- Search Results Summary -----")
            print(f"Found {len(results['documents'])} documents")
            print(f"Found {len(results['sections'])} sections")
            print(f"Found {len(results['subsections'])} subsections")
            print(f"Found {len(results['chunks'])} chunks")
            print(f"Found {len(results['subsubsections'])} subbsubsections")
            print(f"Found {len(results['chunks'])} chunks")

            # Display the results in a hierarchical manner
            if verbose:
                self.print_hierarchical_results(results)
            else:
                # Just print the chunks with their context
                self.print_chunks_with_context(results["chunks"])

            return results

        except Exception as e:
            print(f"Error during search: {e}")
            traceback.print_exc()
            return {"error": str(e)}

    @staticmethod
    def print_hierarchical_results(results: Dict[str, Any]) -> None:
        """Print the results in a hierarchical format"""
        print("\n----- Documents -----")
        for i, doc in enumerate(results["documents"], 1):
            print(f"{i}. Document: {doc.get('document_name', 'Unknown')}")
            print(f"   Scope: {doc.get('scope_title', 'Unknown')}")
            print(f"   Description: {doc.get('scope_description', 'Unknown')[:100]}...")

        print("\n----- Sections -----")
        for i, section in enumerate(results["sections"], 1):
            print(f"{i}. Section: {section.get('title', 'Unknown')}")
            print(f"   From Document: {section.get('parent_document', 'Unknown')}")
            if 'description' in section:
                print(f"   Description: {section['description'][:100]}...")
            if 'summary' in section:
                print(f"   Summary: {section['summary'][:100]}...")

        print("\n----- Subsections -----")
        for i, subsection in enumerate(results["subsections"], 1):
            print(f"{i}. Subsection: {subsection.get('title', 'Unknown')}")
            print(f"   From Section: {subsection.get('parent_section', 'Unknown')}")
            print(f"   From Document: {subsection.get('parent_document', 'Unknown')}")
            if 'description' in subsection:
                print(f"   Description: {subsection['description'][:100]}...")
            if 'summary' in subsection:
                print(f"   Summary: {subsection['summary'][:100]}...")

        HierarchicalSearchEngine.print_chunks_with_context(results["chunks"])

    @staticmethod
    def print_chunks_with_context(chunks: List[Dict[str, Any]]) -> None:
        """Print chunks with their hierarchical context"""
        print("\n----- Most Relevant Chunks -----")
        for i, chunk in enumerate(chunks, 1):
            print(f"{i}. Chunk from:")
            print(f"   Document: {chunk.get('parent_document', 'Unknown')}")
            print(f"   Section: {chunk.get('parent_section', 'Unknown')}")
            print(f"   Subsection: {chunk.get('parent_subsection', 'Unknown')}")
            print(f"   Content: {chunk.get('content', 'Unknown')[:300]}...")
            print("-" * 80)


def main(EMBEDDER_URL="http://localhost:5003/embed"):
    parser = argparse.ArgumentParser(description="Run hierarchical search with embeddings")
    parser.add_argument("--weaviate-url", default="http://localhost:8080", help="Weaviate URL")
    parser.add_argument("--model", default="dunzhang/stella_en_400M_v5", help="Sentence transformer model name")
    parser.add_argument("--fallback-model", default="all-MiniLM-L6-v2", help="Fallback model if primary fails")
    parser.add_argument("--max-docs", type=int, default=3, help="Maximum number of documents")
    parser.add_argument("--max-sections", type=int, default=3, help="Maximum sections per document")
    parser.add_argument("--max-subsections", type=int, default=2, help="Maximum subsections per section")
    parser.add_argument("--max-subsubsections", type=int, default=2, help="Maximum subsubsections per subsection")
    parser.add_argument("--max-chunks", type=int, default=3, help="Maximum chunks")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    parser.add_argument("--output", default="results.json", help="Save results to JSON file")

    args = parser.parse_args()

    # Check if we need to warn about xformers
    if "stella" in args.model.lower() and not XFORMERS_AVAILABLE:
        print("\n" + "=" * 80)
        print("WARNING: You are trying to use a Stella model but xformers is not installed.")
        print("This will cause the model to fall back to the default model.")
        print("To install xformers, run: pip install xformers")
        print("=" * 80 + "\n")

    # Initialize the search engine once
    try:
        search_engine = HierarchicalSearchEngine(
            weaviate_url=args.weaviate_url,
            num_clusters=5
        )
    except Exception as e:
        print(f"Failed to initialize search engine: {e}")
        sys.exit(1)

    # Interactive query loop
    while True:
        query = input("\nEnter your search query (or 'exit' to quit): ")
        if query.lower() in ('exit', 'quit', 'q'):
            break
        data = requests.post(EMBEDDER_URL, json={"texts": [query]})
        if data.status_code != 200:
            print("Something went wrong. Please try again.")
            continue
        data = data.json()
        query_embedding = data["embeddings"][0]
        results = search_engine.search(
            query_embedding=query_embedding,
            max_documents=args.max_docs,
            max_sections_per_doc=args.max_sections,
            max_subsections_per_section=args.max_subsections,
            max_subsubsections_per_subsection=args.max_subsubsections,
            max_chunks_per_container=args.max_chunks,
            verbose=args.verbose
        )

        # Save results to JSON file
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
