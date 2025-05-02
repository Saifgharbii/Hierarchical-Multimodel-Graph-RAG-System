import os
import nltk
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from nltk.tokenize import sent_tokenize
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.schema import Document


CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
DEFAULT_WINDOW_SIZE = 5
DEFAULT_STEP_SIZE = 2
SEPARATORS = ["\n\n", "\n"]
MAX_WORKERS = 2


def download_nltk_dependencies():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt_tab')


def load_single_document(file_path):
    """
    Load a single document file with optimized extraction
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        list: The loaded documents
    """
    
    filename = os.path.basename(file_path)
    
    
    loaders = [
        
        JSONLoader(
            file_path=file_path,
            jq_schema='.. | select(.text_content? != null and .text_content != "") | {content: .text_content, path: .title?}',
            content_key="content",
            metadata_func=lambda meta, content: {
                "source": filename,
                "path": meta.get("path", ""),
                "type": "text_content"
            }
        ),
        
        JSONLoader(
            file_path=file_path,
            jq_schema='.. | select(.description? != null and .description != "") | {content: .description, path: .title?}',
            content_key="content",
            metadata_func=lambda meta, content: {
                "source": filename,
                "path": meta.get("path", ""),
                "type": "description"
            }
        )
    ]
    
    
    documents = []
    for loader in loaders:
        try:
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Error in extraction: {str(e)}")
    
    return documents


def create_text_splitter():
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        disallowed_special=(),
        separators=SEPARATORS
    )
    return splitter


def apply_sentence_windows(doc, window_size, step_size):
    windows = []
    sentences = sent_tokenize(doc.page_content)
    
    if len(sentences) <= window_size:
        new_doc = doc.copy()
        new_doc.page_content = " ".join(sentences)
        windows.append(new_doc)
    else:
        for i in range(0, len(sentences) - window_size + 1, step_size):
            window = sentences[i:i + window_size]
            new_doc = doc.copy()
            new_doc.page_content = " ".join(window)
            windows.append(new_doc)
    
    return windows


def save_documents_to_folder(documents, output_folder, prefix="doc", include_metadata=True):
    """
    Save processed documents to a folder for later use in embedding or other steps.
    
    Args:
        documents (list): List of Document objects (from LangChain) to save
        output_folder (str): Path to the folder where documents will be saved
        prefix (str): Prefix for the document filenames
        include_metadata (bool): Whether to include metadata in the saved files
        
    Returns:
        tuple: (Path to the output folder, success status)
    """
    
    os.makedirs(output_folder, exist_ok=True)
    
    saved_count = 0
    for i, doc in enumerate(documents):
        try:
            filename = f"{prefix}_{i+1}.txt"
            file_path = os.path.join(output_folder, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                if hasattr(doc, 'page_content'):
                    f.write(doc.page_content)
                else:
                    f.write(str(doc))
                if include_metadata and hasattr(doc, 'metadata') and doc.metadata:
                    f.write("\n\n--- METADATA ---\n")
                    for key, value in doc.metadata.items():
                        f.write(f"{key}: {value}\n")
            
            saved_count += 1
        except Exception as e:
            print(f"Error saving document {i+1}: {str(e)}")
    
    print(f"Saved {saved_count} documents to {output_folder}")
    return output_folder, saved_count == len(documents)


def prepare_and_split_docs_by_sentence(file_path, output_folder="processed_docs", window_size=DEFAULT_WINDOW_SIZE, 
                                       step_size=DEFAULT_STEP_SIZE, prefix="chunk", save=True):
    """
    Process a single document file, split it into sentence-level chunks with sliding window,
    and optionally save them to disk.
    
    Args:
        file_path (str): Path to the document file
        output_folder (str): Path where processed documents will be saved
        window_size (int): Number of sentences per window
        step_size (int): Number of sentences to slide the window for each chunk
        prefix (str): Prefix for saved document filenames
        save (bool): Whether to save the documents to disk
        
    Returns:
        tuple: (list of Document objects or output directory path, success status)
    """
    try:
        
        download_nltk_dependencies()
        documents = load_single_document(file_path)
        
        if not documents:
            print(f"Document could not be loaded: {file_path}")
            return None, False
        
        pre_splitter = create_text_splitter()
        pre_split_docs = pre_splitter.split_documents(documents)
        sentence_windows = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(apply_sentence_windows, doc, window_size, step_size) for doc in pre_split_docs]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    sentence_windows.extend(result)
                except Exception as e:
                    print(f"Error in thread: {str(e)}")
        
        print(f"Document is split into {len(sentence_windows)} sentence windows")
        
        # Save documents if requested
        if save and sentence_windows:
            output_path, success = save_documents_to_folder(
                sentence_windows, 
                output_folder, 
                prefix=prefix
            )
            return output_path, success
        return sentence_windows, True
        
    except Exception as e:
        print(f"Error in document processing: {str(e)}")
        return None, False


def process_file_task(file_path, output_folder, file_index, window_size, step_size, prefix):
    """Process a single file in a worker thread"""
    file_name = os.path.basename(file_path)
    file_base_name = os.path.splitext(file_name)[0]
    file_folder = os.path.join(output_folder, f"file{file_index}")
    file_content_folder = os.path.join(file_folder, f"file{file_index}_content")
    os.makedirs(file_folder, exist_ok=True)
    os.makedirs(file_content_folder, exist_ok=True)
    
    print(f"Processing file {file_index}: {file_name}")
    
    result, success = prepare_and_split_docs_by_sentence(
        file_path=file_path,
        output_folder=file_content_folder, 
        window_size=window_size,
        step_size=step_size,
        prefix=prefix,
        save=True,
    )
    
    if success:
        print(f"Processing completed successfully for file {file_index}. Results in: {file_content_folder}")
    else:
        print(f"Document processing failed for file {file_index}.")
    
    return file_index, success, file_folder


def process_files_parallel(files_paths, output_folder, window_size=DEFAULT_WINDOW_SIZE, 
                          step_size=DEFAULT_STEP_SIZE, prefix="chunk", max_workers=MAX_WORKERS):
    """
    Process multiple files in parallel
    
    Args:
        files_paths (list): List of file paths to process
        output_folder (str): Base output folder
        window_size (int): Window size for sentence chunking
        step_size (int): Step size for sentence window sliding
        prefix (str): Prefix for output filenames
        max_workers (int): Maximum number of parallel workers
        
    Returns:
        dict: Results of processing each file
    """
    results = {}
    json_files = [f for f in files_paths if f.lower().endswith('.json')]
    
    if not json_files:
        print("No JSON files found in the input directory")
        return results
    actual_workers = min(max_workers, len(json_files))
    
    print(f"Processing {len(json_files)} JSON files with {actual_workers} parallel workers")
    
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = {}
        for i, file_path in enumerate(json_files):
            future = executor.submit(
                process_file_task, 
                file_path, 
                output_folder, 
                i, 
                window_size, 
                step_size, 
                prefix
            )
            futures[future] = i
        for future in as_completed(futures):
            file_index, success, output_path = future.result()
            results[file_index] = {
                'success': success,
                'output_path': output_path
            }
    
    return results


def main():
    input_path = "first_process/ProcessedDocuments/KaggleResults/results1-88"  
    output_folder = "second_process"  
    window_size = 5 
    step_size = 2 
    max_workers = 2  
    prefix = "chunk"  
    file_index = None  
    jq_schema = '.content[].text_content'  
    print(f"Starting the JSON chunking script with window size {window_size} and step size {step_size}") 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(base_dir, output_folder)
    os.makedirs(output_folder, exist_ok=True)
    input_path = os.path.join(base_dir, input_path) if not os.path.isabs(input_path) else input_path
    if os.path.isfile(input_path):
        if not input_path.lower().endswith('.json'):
            print(f"Error: Input file must be a JSON file, but got {input_path}")
            return
        file_folder = os.path.join(output_folder, "file0")
        file_content_folder = os.path.join(file_folder, "file0_content")
        os.makedirs(file_folder, exist_ok=True)
        os.makedirs(file_content_folder, exist_ok=True)
        
        print(f"Processing single JSON file: {os.path.basename(input_path)}")
        result, success = prepare_and_split_docs_by_sentence(
            file_path=input_path,
            output_folder=file_content_folder, 
            window_size=window_size,
            step_size=step_size,
            prefix=prefix,
            save=True,
        )   
        if success:
            print(f"Processing completed successfully. Results in: {file_content_folder}")
        else:
            print("Document processing failed.")  
    elif os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                if os.path.isfile(os.path.join(input_path, f)) and f.lower().endswith('.json')]
        if not files:
            print(f"No JSON files found in directory: {input_path}")
            return    
        if file_index is not None and 0 <= file_index < len(files):
            file_path = files[file_index]
            file_folder = os.path.join(output_folder, f"file{file_index}")
            file_content_folder = os.path.join(file_folder, f"file{file_index}_content")
            os.makedirs(file_folder, exist_ok=True)
            os.makedirs(file_content_folder, exist_ok=True)
            print(f"Processing file {file_index}: {os.path.basename(file_path)}")
            result, success = prepare_and_split_docs_by_sentence(
                file_path=file_path,
                output_folder=file_content_folder,  
                window_size=window_size,
                step_size=step_size,
                prefix=prefix,
                save=True,
            )
            if success:
                print(f"Processing completed successfully. Results in: {file_content_folder}")
            else:
                print("Document processing failed.")
        else:
            results = process_files_parallel(
                files_paths=files,
                output_folder=output_folder,
                window_size=window_size,
                step_size=step_size,
                prefix=prefix,
                max_workers=max_workers
            )
            successful = sum(1 for r in results.values() if r['success'])
            print(f"\nProcessing summary: {successful} of {len(files)} files processed successfully")     
    else:
        print(f"Error: Input path '{input_path}' does not exist or is neither a file nor a directory")


if __name__ == "__main__":
    main()