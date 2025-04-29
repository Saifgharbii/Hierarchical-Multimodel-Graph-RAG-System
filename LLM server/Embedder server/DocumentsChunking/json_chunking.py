import os
import json
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_json_files(input_dir, pattern="*.json"):
    """
    Load all JSON files from the input directory matching the pattern.
    Returns a dictionary with file paths as keys and loaded JSON data as values.
    """
    file_pattern = os.path.join(input_dir, pattern)
    json_files = glob.glob(file_pattern)

    if not json_files:
        print(f"No JSON files found matching pattern: {file_pattern}")
        return {}

    print(f"Found {len(json_files)} JSON files to process")

    file_data = {}
    for file_path in tqdm(json_files, desc="Loading JSON files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                file_data[file_path] = data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return file_data


def create_chunks_for_text(text, text_splitter):
    """Split text into chunks using the provided text splitter"""
    if not text or not text.strip():
        return []

    chunks = text_splitter.split_text(text)
    return chunks


def process_subsubsection(subsubsection, text_splitter):
    """Process a subsubsection and create chunks for its text_content"""
    result = {}

    if "text_content" in subsubsection and subsubsection["text_content"]:
        result["chunks"] = create_chunks_for_text(subsubsection["text_content"].replace("\u00a0", " "), text_splitter)
    else:
        result["chunks"] = []

    # Copy other fields
    for key, value in subsubsection.items():
        if key != "chunks":  # Avoid overwriting the chunks we just created
            result[key] = value

    return result


def process_subsection(subsection, text_splitter):
    """Process a subsection and create chunks for its text_content and process its subsubsections"""
    result = {}

    # Create chunks for text_content
    if "text_content" in subsection and subsection["text_content"]:
        result["chunks"] = create_chunks_for_text(subsection["text_content"].replace("\u00a0", " "), text_splitter)
    else:
        result["chunks"] = []

    # Process subsubsections
    if "subsubsections" in subsection and subsection["subsubsections"]:
        result["subsubsections"] = [
            process_subsubsection(subsubsection, text_splitter)
            for subsubsection in subsection["subsubsections"]
        ]
    else:
        result["subsubsections"] = subsection.get("subsubsections", [])

    # Copy other fields
    for key, value in subsection.items():
        if key not in ["chunks", "subsubsections"]:  # Avoid overwriting fields we've already processed
            result[key] = value

    return result


def process_section(section, text_splitter):
    """Process a section and create chunks for its subsections and their subsubsections"""
    result = {}

    # Process subsections
    if "subsections" in section and section["subsections"]:
        result["subsections"] = [
            process_subsection(subsection, text_splitter)
            for subsection in section["subsections"]
        ]
    else:
        result["subsections"] = section.get("subsections", [])

    # Copy other fields
    for key, value in section.items():
        if key != "subsections":  # Avoid overwriting subsections we just processed
            result[key] = value

    return result


def process_document(document_data, text_splitter):
    """Process a document and all its sections, subsections, and subsubsections"""
    result = {}

    # Copy document metadata
    for key, value in document_data.items():
        if key != "content":
            result[key] = value

    # Process content sections
    if "content" in document_data and document_data["content"]:
        result["content"] = [
            process_section(section, text_splitter)
            for section in document_data["content"]
        ]
    else:
        result["content"] = []

    return result


def process_file(file_path, document_data, text_splitter):
    """Process a single file with the text splitter"""
    try:
        processed_data = process_document(document_data, text_splitter)
        return file_path, processed_data
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return file_path, None


def chunk_json_documents(input_dir, model_name="gpt2", max_chunk_size=2048, chunk_overlap=200, workers=None,
                         pattern="*.json"):
    """
    Process JSON documents and chunk text content using LangChain's RecursiveCharacterTextSplitter.

    Args:
        input_dir (str): Directory containing JSON files
        model_name (str): Model name for tokenizer
        max_chunk_size (int): Maximum chunk size in tokens
        chunk_overlap (int): Overlap between chunks in tokens
        workers (int): Number of worker threads (defaults to CPU count)
        pattern (str): File pattern to match

    Returns:
        dict: Dictionary with processed JSON data including chunks
    """
    # Set default number of workers if not specified
    if workers is None:
        workers = os.cpu_count()

    # Load tokenizer
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def token_counter(text):
        return len(tokenizer.encode(text))

    # Create the LangChain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_counter,
        separators=["\n\n", "\n", " ", ""]
    )

    # Load JSON files
    file_data = load_json_files(input_dir, pattern)

    if not file_data:
        return {}

    # Process files in parallel
    processed_data = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_file, file_path, document_data, text_splitter)
            for file_path, document_data in file_data.items()
        ]

        for future in tqdm(futures, total=len(file_data), desc="Processing and chunking files"):
            file_path, result = future.result()
            if result:
                processed_data[file_path] = result

    print(f"Successfully processed {len(processed_data)} files")
    return processed_data


# Example usage
if __name__ == "__main__":
    input_directory = "../TextExtraction/ProcessedDocuments/KaggleResults"
    output_dir = "./ChunkedDocuments"
    model = "dunzhang/stella_en_400M_v5"
    chunk_size = 2048  # Maximum chunk size in tokens
    overlap = 200  # Overlap between chunks in tokens
    num_workers = os.cpu_count()
    file_pattern = "*.json"

    # Run the chunking process
    chunked_documents = chunk_json_documents(
        input_dir=input_directory,
        model_name=model,
        max_chunk_size=chunk_size,
        chunk_overlap=overlap,
        workers=num_workers,
        pattern=file_pattern
    )

    # If you want to save the processed data
    if chunked_documents:
        output_dir = "chunked_documents"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Saving processed documents to {output_dir}")
        for i, (file_path, data) in enumerate(chunked_documents.items()):
            output_file = os.path.join(output_dir, f"chunked_{i}_{os.path.basename(file_path)}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        print(f"Saved {len(chunked_documents)} processed documents")