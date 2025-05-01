import os
import json
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_markdwon_files(input_dir, pattern="*.md"):
    """
    Load all markdwon files from the input directory matching the pattern.
    Returns a dictionary with file paths as keys and loaded markdwon data as values.
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
                data = f.read()
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


def process_file(file_path, markdown_text, text_splitter):
    """Process a single file with the text splitter"""
    try:
        processed_data = create_chunks_for_text(markdown_text, text_splitter)
        return file_path, processed_data
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return file_path, None


def chunk_markdwon_documents(input_dir, model_name="gpt2", max_chunk_size=1024, chunk_overlap=200, workers=None,
                         pattern="*.md"):
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
    file_data = load_markdwon_files(input_dir, pattern)

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
    os.makedirs("./ChunkedDocuments/", exist_ok=True)

    # Run the chunking process
    chunked_documents = chunk_markdwon_documents(
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