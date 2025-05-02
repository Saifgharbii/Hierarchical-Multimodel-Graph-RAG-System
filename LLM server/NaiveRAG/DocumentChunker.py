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


def chunk_markdwon_documents(input_dir, output_dir, model_name="gpt2", max_chunk_size=1024, chunk_overlap=200,
                             workers=None, pattern="*.md"):
    """
    Process Markdown documents, chunk text content, and save chunks to separate files.

    Added functionality: Save chunks to output directory preserving folder structure
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # (Existing processing code remains the same until after processed_data is created)
    if workers is None:
        workers = os.cpu_count()

    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def token_counter(text):
        return len(tokenizer.encode(text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_counter,
        separators=["\n\n", "\n", " ", ""]
    )

    file_data = load_markdwon_files(input_dir, pattern)

    if not file_data:
        return {}

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

    print(f"\nSaving chunks to output directory: {output_dir}")
    for file_path, chunks in processed_data.items():
        try:
            relative_path = os.path.relpath(file_path, input_dir)
            new_relative = os.path.splitext(relative_path)[0] + '_chunks.json'
            output_path = os.path.join(output_dir, new_relative)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save chunks as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            print(f"Saved {len(chunks)} chunks to {output_path}")
        except Exception as e:
            print(f"Error saving chunks for {file_path}: {e}")

    print(f"\nFinished saving all chunks to {output_dir}")
    return processed_data


if __name__ == "__main__":
    input_directory = "./MarkdownFiles"
    output_dir = "./ChunkedDocuments"
    model_name = "dunzhang/stella_en_400M_v5"
    chunk_size = 1024  # Maximum chunk size in tokens
    overlap = 100  # Overlap between chunks in tokens
    num_workers = os.cpu_count()
    file_pattern = "*.md"
    batch_size = 64
    max_length = 1024

    # Run the chunking process
    chunked_documents = chunk_markdwon_documents(
        input_dir=input_directory,
        output_dir=output_dir,
        model_name=model_name,
        max_chunk_size=chunk_size,
        chunk_overlap=overlap,
        workers=num_workers,
        pattern=file_pattern
    )
