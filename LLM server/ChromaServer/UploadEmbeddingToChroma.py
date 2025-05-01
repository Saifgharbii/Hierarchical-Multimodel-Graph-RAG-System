import os
import json
from tqdm import tqdm
import chromadb
import re

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


def process_node(node, level, parent_collection_name, document_id_base, chroma_client, json_file_path):
    """Recursively process a node in the JSON hierarchy and upload its embeddings"""
    # Extract node title and sanitize for collection name
    node_title = node.get("title", {}).get("title", "unnamed_node")
    sanitized_title = sanitize_collection_name(node_title)
    collection_suffix = f"{sanitized_title}_level_{level}"
    collection_name = f"{document_id_base}_{collection_suffix}"

    # Create or get collection
    try:
        collection = chroma_client.get_collection(name=collection_name)
        if CONFIG["VERBOSE"]:
            print(f"Using existing collection: {collection_name}")
    except Exception as e:
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={
                "hierarchy_level": level,
                "parent_collection": parent_collection_name,
                "source_document": document_id_base,
                "node_title": node_title,
                "json_file": json_file_path
            }
        )
        if CONFIG["VERBOSE"]:
            print(f"Created collection: {collection_name} (Level: {level}, Parent: {parent_collection_name})")

    # Collect all potential document IDs for this node
    doc_ids = []
    title_doc_id = f"{collection_name}_title"
    doc_ids.append(title_doc_id)
    description_doc_id = f"{collection_name}_description"
    doc_ids.append(description_doc_id)
    chunk_doc_ids = [f"{collection_name}_chunk_{i}" for i in range(len(node.get("chunks", [])))]
    doc_ids.extend(chunk_doc_ids)

    # Check existing documents
    try:
        existing = collection.get(ids=doc_ids)
        existing_ids = set(existing["ids"])
    except Exception as e:
        existing_ids = set()

    # Prepare batch data
    batch_embeddings = []
    batch_texts = []
    batch_ids = []
    batch_metadatas = []

    # Process title embedding
    title_embedding = node.get("title", {}).get("embedding")
    title_text = node.get("title", {}).get("title", "")
    if title_embedding and title_doc_id not in existing_ids:
        batch_embeddings.append(title_embedding)
        batch_texts.append(title_text)
        batch_ids.append(title_doc_id)
        batch_metadatas.append({
            "type": "title",
            "document": document_id_base,
            "node_title": node_title,
            "hierarchy_level": level,
            "source_file": json_file_path
        })

    # Process description embedding
    desc_embedding = node.get("description", {}).get("embedding")
    desc_text = node.get("description", {}).get("description", "")
    if desc_embedding and description_doc_id not in existing_ids:
        batch_embeddings.append(desc_embedding)
        batch_texts.append(desc_text)
        batch_ids.append(description_doc_id)
        batch_metadatas.append({
            "type": "description",
            "document": document_id_base,
            "node_title": node_title,
            "hierarchy_level": level,
            "source_file": json_file_path
        })

    # Process chunks
    for i, chunk in enumerate(node.get("chunks", [])):
        chunk_doc_id = chunk_doc_ids[i]
        if chunk_doc_id in existing_ids:
            continue

        chunk_embedding = chunk.get("embedding")
        chunk_text = chunk.get("chunk", "")
        if chunk_embedding:
            batch_embeddings.append(chunk_embedding)
            batch_texts.append(chunk_text)
            batch_ids.append(chunk_doc_id)
            batch_metadatas.append({
                "type": "chunk",
                "document": document_id_base,
                "node_title": node_title,
                "hierarchy_level": level,
                "chunk_index": i,
                "source_file": json_file_path
            })

    # Batch processing
    for i in range(0, len(batch_embeddings), CONFIG["BATCH_SIZE"]):
        batch_e = batch_embeddings[i:i + CONFIG["BATCH_SIZE"]]
        batch_t = batch_texts[i:i + CONFIG["BATCH_SIZE"]]
        batch_i = batch_ids[i:i + CONFIG["BATCH_SIZE"]]
        batch_m = batch_metadatas[i:i + CONFIG["BATCH_SIZE"]]

        try:
            collection.add(
                embeddings=batch_e,
                documents=batch_t,
                metadatas=batch_m,
                ids=batch_i
            )
        except Exception as e:
            print(f"Error adding batch to {collection_name}: {e}")

    # Recursively process subsections
    next_level = level + 1
    if level == 1:
        child_key = "subsections"
    elif level == 2:
        child_key = "subsubsections"
    else:
        child_key = None

    if child_key:
        for child_node in node.get(child_key, []):
            process_node(
                node=child_node,
                level=next_level,
                parent_collection_name=collection_name,
                document_id_base=document_id_base,
                chroma_client=chroma_client,
                json_file_path=json_file_path
            )


def sanitize_collection_name(name):
    """Sanitize collection name to be compatible with ChromaDB requirements"""
    # Replace invalid characters with underscores and truncate if necessary
    sanitized = (name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                 .replace(".", "_").replace("\t","").replace("\n","").replace(",","").replace("–","-"))
    sanitized = sanitized.strip("_")
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', sanitized)
    return sanitized[:63]  # ChromaDB collection name length limit


def find_json_files(directory):
    """Find all JSON files in the given directory"""
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def upload_local_embeddings_to_chromadb(root_dir=None, chroma_client=None):
    """Upload embeddings from JSON files with hierarchical structure to ChromaDB"""
    # Use default from CONFIG if not provided
    root_dir = root_dir or CONFIG["PROCESSED_DATA_DIR"]
    chroma_client = chroma_client or setup_chromadb()

    # Find all JSON files in the root directory
    json_files = find_json_files(root_dir)

    if CONFIG["VERBOSE"]:
        print(f"Found {len(json_files)} JSON files")

    # Process each JSON file
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error loading {json_file}: {e}")
                continue

        document_name = str(data.get("document_name", "unnamed_document")).replace(".docx", "")
        document_id_base = document_name  # Base for collection names

        # Process each top-level content section
        for section in data.get("content", []):
            process_node(
                node=section,
                level=1,
                parent_collection_name="",
                document_id_base=document_id_base,
                chroma_client=chroma_client,
                json_file_path=json_file
            )

    if CONFIG["VERBOSE"]:
        print("All local embeddings uploaded to ChromaDB")
