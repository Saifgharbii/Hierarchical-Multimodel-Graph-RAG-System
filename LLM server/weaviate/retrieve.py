import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from typing import List, Dict, Any, Optional
import os
import sys
import traceback
import importlib.util

# Import the HierarchicalGraphRAG class
# If it's in another file, make sure to import it properly
from setup import HierarchicalGraphRAG

# Check if xformers is installed
XFORMERS_AVAILABLE = importlib.util.find_spec("xformers") is not None

class HierarchicalSearchEngine:
    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        model_name: str = "dunzhang/stella_en_400M_v5",
        optimize_disk: bool = True,
        num_clusters: int = 10,
    ):
        """
        Initialize the search engine with embedding model and RAG system.
        
        Args:
            weaviate_url: URL of the Weaviate instance
            model_name: Name of the sentence transformer model
            optimize_disk: Whether to optimize for disk usage
            num_clusters: Number of clusters for the RAG system

        """
        # Try to load the embedding model
        print(f"Loading embedding model: {model_name}")
        try:
            # Check if we're using Stella and need xformers
            if "stella" in model_name.lower() and not XFORMERS_AVAILABLE:
                print("WARNING: Stella models require xformers which is not installed.")
                print("To install xformers, run: pip install xformers")
            else:
                self.model = SentenceTransformer(model_name, trust_remote_code=True)
                print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
        
        
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
        query: str,
        max_documents: int = 2,
        max_sections_per_doc: int = 2,
        max_subsections_per_section: int = 2,
        max_subsubsections_per_subsection: int=2,
        max_chunks_per_container: int = 3,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a hierarchical search with a given query using the initialized components.
        
        Args:
            query: The search query
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
            print(f"Generating embedding for query: '{query}'")
            query_embedding = self.model.encode([query], prompt_name="s2p_query")[0].tolist()
            
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
            print(f"Found {results['documents']} documents")
            print(f"Found {len(results['sections'])} sections")
            print(f"Found {results['sections']} sections")
            print(f"Found {len(results['subsections'])} subsections")
            print(f"Found {results['subsections']} subsections")
            print(f"Found {len(results['chunks'])} chunks")
            print(f"Found {results['chunks']} chunks")
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


def main():
    parser = argparse.ArgumentParser(description="Run hierarchical search with embeddings")
    parser.add_argument("--weaviate-url", default="http://localhost:8080", help="Weaviate URL")
    parser.add_argument("--model", default="dunzhang/stella_en_400M_v5", help="Sentence transformer model name")
    parser.add_argument("--fallback-model", default="all-MiniLM-L6-v2", help="Fallback model if primary fails")
    parser.add_argument("--max-docs", type=int, default=3, help="Maximum number of documents")
    parser.add_argument("--max-sections", type=int, default=2, help="Maximum sections per document")
    parser.add_argument("--max-subsections", type=int, default=2, help="Maximum subsections per section")
    parser.add_argument("--max-subsubsections", type=int, default=2, help="Maximum subsubsections per subsection")
    parser.add_argument("--max-chunks", type=int, default=3, help="Maximum chunks")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    parser.add_argument("--output", default="results.json", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Check if we need to warn about xformers
    if "stella" in args.model.lower() and not XFORMERS_AVAILABLE:
        print("\n" + "="*80)
        print("WARNING: You are trying to use a Stella model but xformers is not installed.")
        print("This will cause the model to fall back to the default model.")
        print("To install xformers, run: pip install xformers")
        print("="*80 + "\n")
    
    # Initialize the search engine once
    try:
        search_engine = HierarchicalSearchEngine(
            weaviate_url=args.weaviate_url,
            model_name=args.model,
        )
    except Exception as e:
        print(f"Failed to initialize search engine: {e}")
        sys.exit(1)
    
    # Interactive query loop
    while True:
        query = input("\nEnter your search query (or 'exit' to quit): ")
        if query.lower() in ('exit', 'quit', 'q'):
            break
            
        results = search_engine.search(
            query=query,
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