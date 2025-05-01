import os
import json
import re

import torch
import chromadb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration parameters
CONFIG = {
    # Directory settings
    "PROCESSED_DATA_DIR": "../EmbeddingProcess/EmbeddedDocuments/embedded_files",
    "CHROMA_DB_PATH": "./chroma_db",

    # File patterns - corrected to match your actual naming patterns
    "CHUNK_FILE_PREFIX": "chunk_",
    "CHUNK_FILE_SUFFIX": ".txt",
    "EMBED_FILE_PREFIX": "embed_",
    "EMBED_FILE_SUFFIX": ".json",

    # Search settings
    "DEFAULT_SIMILARITY_THRESHOLD": 0.6,
    "DEFAULT_TOP_K": 5,

    # LLM settings
    "LLM_MODEL_NAME": "meta-llama/Llama-3.2-8B-Instruct",
    "MAX_NEW_TOKENS": 512,
    "TEMPERATURE": 0.7,

    # Batch processing settings
    "BATCH_SIZE": 8,

    # Logging verbosity
    "VERBOSE": True
}


def setup_chromadb():
    """Initialize ChromaDB client"""
    client = chromadb.PersistentClient(path=CONFIG["CHROMA_DB_PATH"])
    return client


def initialize_llm():
    """Initialize Llama 3.2 model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["LLM_MODEL_NAME"])
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["LLM_MODEL_NAME"],
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return model, tokenizer
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("Continuing without LLM capabilities.")
        return None, None


def sanitize_collection_name(name):
    """Sanitize collection name to be compatible with ChromaDB requirements"""
    # Replace invalid characters with underscores and truncate if necessary
    sanitized = (name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                 .replace(".", "_").replace("\t","").replace("\n","").replace(",","").replace("–","-"))
    sanitized = sanitized.strip("_")
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', sanitized)
    return sanitized[:63]  # ChromaDB collection name length limit


def find_json_files(dir_path):
    """Find all JSON embedding files in the given directory"""
    json_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.startswith(CONFIG["EMBED_FILE_PREFIX"]) and file.endswith(CONFIG["EMBED_FILE_SUFFIX"]):
                json_files.append(os.path.join(root, file))
    return json_files


def load_embedding_from_json(json_path):
    """Load both chunk text and embedding from your JSON file format"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if your JSON structure has chunk and embedding fields
        if "chunk" in data and "embedding" in data:
            return data["chunk"], data["embedding"]

        # Or if the embedding is stored directly
        elif "embedding" in data:
            # Try to find corresponding chunk file
            chunk_file = json_path.replace(CONFIG["EMBED_FILE_PREFIX"], CONFIG["CHUNK_FILE_PREFIX"]).replace(
                CONFIG["EMBED_FILE_SUFFIX"], CONFIG["CHUNK_FILE_SUFFIX"])
            if os.path.exists(chunk_file):
                with open(chunk_file, 'r', encoding='utf-8') as cf:
                    chunk_text = cf.read()
                return chunk_text, data["embedding"]
            else:
                if CONFIG["VERBOSE"]:
                    print(f"No corresponding chunk file found for {json_path}")
                return None, data["embedding"]
        else:
            if CONFIG["VERBOSE"]:
                print(f"Unexpected JSON format in {json_path}")
            return None, None
    except Exception as e:
        if CONFIG["VERBOSE"]:
            print(f"Error loading data from {json_path}: {e}")
        return None, None


def upload_local_embeddings_to_chromadb(root_dir=None, chroma_client=None):
    """Upload locally stored embeddings to ChromaDB using the JSON files"""
    # Use default from CONFIG if not provided
    root_dir = root_dir or CONFIG["PROCESSED_DATA_DIR"]
    chroma_client = chroma_client or setup_chromadb()

    # Find all JSON embedding files
    json_files = find_json_files(root_dir)

    if CONFIG["VERBOSE"]:
        print(f"Found {len(json_files)} JSON embedding files")

    # Group JSON files by directory to maintain hierarchy
    directory_files = {}
    for json_file in json_files:
        dir_path = os.path.dirname(json_file)
        if dir_path not in directory_files:
            directory_files[dir_path] = []
        directory_files[dir_path].append(json_file)

    # Process each directory separately to create hierarchical collections
    for dir_path, files in tqdm(directory_files.items(), desc="Processing directories"):
        # Create a collection for this directory
        collection_name = sanitize_collection_name(dir_path)

        # Store directory hierarchy information for later use in hierarchical search
        parts = dir_path.split(os.sep)

        # Determine level in hierarchy - root is level 1
        root_dir_name = os.path.basename(os.path.normpath(root_dir))
        try:
            index_of_root = parts.index(root_dir_name)
            level = len(parts) - index_of_root
        except ValueError:
            level = len(parts)  # Fallback if root not found

        # Determine parent directory
        parent_dir = os.path.dirname(dir_path) if level > 1 else None
        parent_collection = sanitize_collection_name(parent_dir) if parent_dir else None

        # Get or create collection
        try:
            collection = chroma_client.get_collection(name=collection_name)
            if CONFIG["VERBOSE"]:
                print(f"Found existing collection: {collection_name}")
        except:
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={
                    "directory_path": dir_path,
                    "hierarchy_level": level,
                    "parent_collection": parent_collection
                }
            )
            if CONFIG["VERBOSE"]:
                print(f"Created new collection: {collection_name} (Level: {level}, Parent: {parent_collection})")

        # Prepare batch processing data
        batch_embeddings = []
        batch_texts = []
        batch_ids = []
        batch_metadatas = []

        # Process all JSON files in this directory
        for json_file in tqdm(files, desc=f"Processing files in {dir_path}", leave=False):
            # Extract a unique ID from the filename
            base_name = os.path.basename(json_file)
            file_id = base_name.replace(CONFIG["EMBED_FILE_PREFIX"], "").replace(CONFIG["EMBED_FILE_SUFFIX"], "")

            # Create a unique document ID
            doc_id = f"{collection_name}_{file_id}"

            # Check if this document is already in the collection
            try:
                collection.get(ids=[doc_id])
                if CONFIG["VERBOSE"]:
                    print(f"Document {doc_id} already exists in collection, skipping")
                continue
            except:
                pass

            # Load chunk text and embedding from the JSON file
            chunk_text, embedding = load_embedding_from_json(json_file)
            if embedding is None:
                continue

            # Prepare metadata
            metadata = {
                "original_file_path": json_file,
                "directory": dir_path,
                "file_id": file_id
            }

            # Add to batch (even if chunk_text is None, we'll store the embedding)
            batch_embeddings.append(embedding)
            batch_texts.append(chunk_text if chunk_text else "")  # Empty string if no text available
            batch_ids.append(doc_id)
            batch_metadatas.append(metadata)

            # Process batch if it reaches BATCH_SIZE
            if len(batch_embeddings) >= CONFIG["BATCH_SIZE"]:
                try:
                    collection.add(
                        embeddings=batch_embeddings,
                        documents=batch_texts,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                except Exception as e:
                    print(f"Error adding batch to collection: {e}")
                batch_embeddings, batch_texts, batch_ids, batch_metadatas = [], [], [], []

        # Process any remaining items in the batch
        if batch_embeddings:
            try:
                collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            except Exception as e:
                print(f"Error adding final batch to collection: {e}")

    if CONFIG["VERBOSE"]:
        print("All local embeddings uploaded to ChromaDB")


def get_collection_hierarchy(chroma_client):
    """
    Build a hierarchical map of collections based on their metadata
    """
    collections = chroma_client.list_collections()
    hierarchy_map = {}

    for collection in collections:
        collection_obj = chroma_client.get_collection(name=collection.name)
        metadata = collection_obj.metadata

        if not metadata or "hierarchy_level" not in metadata:
            # Handle collections without hierarchy metadata
            hierarchy_level = 1  # Default to top level
            parent = None
        else:
            hierarchy_level = metadata["hierarchy_level"]
            parent = metadata.get("parent_collection")

        if hierarchy_level not in hierarchy_map:
            hierarchy_map[hierarchy_level] = []

        hierarchy_map[hierarchy_level].append({
            "name": collection.name,
            "parent": parent
        })

    return hierarchy_map


def hierarchical_search(query_embedding, chroma_client=None, similarity_threshold=None, top_k=None):
    """
    Perform hierarchical search starting from top-level collections
    """
    # Use defaults from CONFIG if not provided
    chroma_client = chroma_client or setup_chromadb()
    similarity_threshold = similarity_threshold or CONFIG["DEFAULT_SIMILARITY_THRESHOLD"]
    top_k = top_k or CONFIG["DEFAULT_TOP_K"]

    # Get collection hierarchy
    hierarchy = get_collection_hierarchy(chroma_client)
    max_level = max(hierarchy.keys()) if hierarchy else 0

    # Store results for each level
    all_results = []
    matched_paths = set()

    if CONFIG["VERBOSE"]:
        print(f"Searching through {len(hierarchy.get(1, []))} top-level collections...")

    # Search through each level
    for level in range(1, max_level + 1):
        level_collections = hierarchy.get(level, [])
        level_results = []

        for collection_info in level_collections:
            collection_name = collection_info["name"]
            parent = collection_info["parent"]

            # Only search in this collection if:
            # 1. It's a top-level collection (level 1), or
            # 2. Its parent was a match in the previous level
            if level == 1 or parent in matched_paths:
                try:
                    collection = chroma_client.get_collection(name=collection_name)
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k
                    )

                    # Process results
                    for i, (doc_id, doc_text, distance) in enumerate(zip(
                            results['ids'][0],
                            results['documents'][0],
                            results['distances'][0])):

                        # Convert distance to similarity score
                        similarity = distance

                        if similarity >= similarity_threshold:
                            result = {
                                "collection": collection_name,
                                "document_id": doc_id,
                                "text": doc_text,
                                "similarity": similarity,
                                "hierarchy_level": level,
                                "rank_in_collection": i + 1
                            }
                            level_results.append(result)
                            matched_paths.add(collection_name)

                except Exception as e:
                    if CONFIG["VERBOSE"]:
                        print(f"Error searching collection {collection_name}: {e}")

        # Add this level's results to all results
        if level_results:
            if CONFIG["VERBOSE"]:
                print(f"Found {len(level_results)} matches at level {level}")
            all_results.extend(level_results)
        elif CONFIG["VERBOSE"]:
            print(f"No matches found at level {level}")

        # If we didn't find any matches at this level, we can't go deeper in those branches
        if level < max_level and not matched_paths:
            if CONFIG["VERBOSE"]:
                print(f"No matching paths to continue search to level {level + 1}")
            break

    # Sort all results by similarity score
    all_results.sort(key=lambda x: x["similarity"], reverse=True)

    return all_results[:top_k]


def perform_search(query, query_embedding, chroma_client=None, similarity_threshold=None, top_k=None):
    """
    Perform search and display results in a user-friendly way
    """
    chroma_client = chroma_client or setup_chromadb()
    similarity_threshold = similarity_threshold or CONFIG["DEFAULT_SIMILARITY_THRESHOLD"]
    top_k = top_k or CONFIG["DEFAULT_TOP_K"]

    print(f"\nPerforming hierarchical search for: '{query}'")
    results = hierarchical_search(
        query_embedding=query_embedding,
        chroma_client=chroma_client,
        similarity_threshold=similarity_threshold,
        top_k=top_k
    )

    if not results:
        print("No matching results found")
        return []

    print("\n=== Search Results ===")
    for i, result in enumerate(results):
        print(f"\n[{i + 1}] Score: {result['similarity']:.4f} (Level: {result['hierarchy_level']})")
        print(f"Collection: {result['collection']}")
        print(f"Text: {result['text'][:150]}..." if len(result['text']) > 150 else f"Text: {result['text']}")
        print("-" * 50)

    return results


def find_sample_embedding(root_dir=None):
    """Find a sample embedding for search testing"""
    root_dir = root_dir or CONFIG["PROCESSED_DATA_DIR"]
    json_files = find_json_files(root_dir)

    if not json_files:
        print("No embedding files found!")
        return None

    _, embedding = load_embedding_from_json(json_files[0])
    return embedding


def generate_llm_response(query, search_results, model=None, tokenizer=None):
    """Generate an LLM response using Llama 3.2 based on search results"""
    if model is None or tokenizer is None:
        print("LLM model not available. Showing only search results.")
        return None

    # Construct context from search results
    context = ""
    for i, result in enumerate(search_results):
        context += f"Document {i + 1} (Similarity: {result['similarity']:.4f}):\n{result['text']}\n\n"

    # Construct the prompt
    prompt = f"""<|begin_of_text|><|system|>
                You are a helpful assistant that answers questions based on provided document context.
                If the documents don't contain relevant information to answer the question, admit that you don't know.
                Don't make up information that's not in the documents.
                Always cite which document(s) you used to generate your answer.
                </s>
                <|user|>
                Context information:
                {context}
                
                Based on this context, please answer the following question:
                {query}
                </s>
                <|assistant|>
            """

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=CONFIG["MAX_NEW_TOKENS"],
        temperature=CONFIG["TEMPERATURE"],
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract just the assistant's response (remove the prompt)
    response_parts = response.split("<|assistant|>")
    if len(response_parts) > 1:
        return response_parts[1].strip()
    return response


def main():
    # Initialize ChromaDB client
    chroma_client = setup_chromadb()

    # Initialize LLM (if possible)
    llm_model, llm_tokenizer = initialize_llm()
    llm_available = llm_model is not None and llm_tokenizer is not None

    # Choose whether to process data or search
    print("\n=== Document Embedding and Retrieval System with LLM ===")
    print("1: Upload Local Embeddings to ChromaDB")
    print("2: Search Documents and Generate Response")
    print("3: Configure Parameters")
    print("4: Exit")

    action = input("\nChoose action (1-4): ").strip()

    if action == "1":
        # Upload local embeddings to ChromaDB
        upload_local_embeddings_to_chromadb(CONFIG["PROCESSED_DATA_DIR"], chroma_client)
        print("Local embeddings uploaded to ChromaDB successfully!")
        main()  # Return to main menu

    elif action == "2":
        # Find a sample embedding for search
        sample_embedding = find_sample_embedding(CONFIG["PROCESSED_DATA_DIR"])
        if sample_embedding is None:
            print("Cannot perform search without embeddings.")
            main()
            return

        # Perform search
        while True:
            query = input("\nEnter search query (or 'q' to quit): ").strip()
            if query.lower() == 'q':
                main()  # Return to main menu
                break

            # For demonstration, we'll use a sample embedding from our files
            # In a real system, you'd generate an embedding for the query text
            print("Using sample embedding for search (since we don't have the original embedding model loaded)")

            # Allow custom parameters per search if needed
            custom_params = input("Use custom parameters? (y/n): ").strip().lower()
            similarity_threshold = CONFIG["DEFAULT_SIMILARITY_THRESHOLD"]
            top_k = CONFIG["DEFAULT_TOP_K"]

            if custom_params == 'y':
                try:
                    similarity_threshold = float(
                        input(f"Similarity threshold (default: {similarity_threshold}): ") or similarity_threshold)
                    top_k = int(input(f"Number of results (default: {top_k}): ") or top_k)
                except ValueError:
                    print("Invalid input, using default parameters")

            # Perform the search
            search_results = perform_search(query, sample_embedding, chroma_client, similarity_threshold, top_k)

            # Generate LLM response if available
            if search_results and llm_available:
                print("\n=== Generating LLM Response ===")
                llm_response = generate_llm_response(query, search_results, llm_model, llm_tokenizer)
                print("\n=== LLM Response ===")
                print(llm_response)
            elif search_results and not llm_available:
                print(
                    "\nLLM model not available. Install the required packages and ensure you have access to Llama 3.2.")

    elif action == "3":
        # Configure parameters
        print("\nCurrent Configuration:")
        for key, value in CONFIG.items():
            print(f"{key}: {value}")

        print("\nEnter new values (leave blank to keep current value):")
        for key in CONFIG:
            if key in ["VERBOSE"]:
                new_value = input(f"New value for {key} (True/False): ").strip()
                if new_value.lower() in ['true', 'false']:
                    CONFIG[key] = new_value.lower() == 'true'
            elif key in ["BATCH_SIZE", "DEFAULT_TOP_K", "MAX_NEW_TOKENS"]:
                new_value = input(f"New value for {key}: ").strip()
                if new_value.isdigit():
                    CONFIG[key] = int(new_value)
            elif key in ["DEFAULT_SIMILARITY_THRESHOLD", "TEMPERATURE"]:
                new_value = input(f"New value for {key}: ").strip()
                try:
                    CONFIG[key] = float(new_value)
                except ValueError:
                    pass
            else:
                new_value = input(f"New value for {key}: ").strip()
                if new_value:
                    CONFIG[key] = new_value

        print("\nUpdated Configuration:")
        for key, value in CONFIG.items():
            print(f"{key}: {value}")

        # Continue to main menu
        main()

    elif action == "4":
        print("Exiting program. Goodbye!")
        return

    else:
        print("Invalid action selected")
        main()


if __name__ == "__main__":
    main()
