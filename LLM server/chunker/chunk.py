# Import section
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from docx import Document as DocxReader
import nltk
from nltk.tokenize import sent_tokenize
import os

# Constants and configuration
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
DEFAULT_WINDOW_SIZE = 5
DEFAULT_STEP_SIZE = 2
SEPARATORS = ["\n\n", "\n"]

def download_nltk_dependencies():
    nltk.download('punkt_tab')

def load_docx_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".docx"):
            path = os.path.join(directory, filename)
            docx = DocxReader(path)
            text = "\n".join([para.text for para in docx.paragraphs])
            documents.append(Document(page_content=text, metadata={"source": filename}))
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
    text = doc.page_content
    sentences = sent_tokenize(text)
    
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
        str: Path to the output folder
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save each document
    saved_count = 0
    for i, doc in enumerate(documents):
        try:
            # Create a filename
            filename = f"{prefix}_{i+1}.txt"
            file_path = os.path.join(output_folder, filename)
            
            # Save the document content
            with open(file_path, 'w', encoding='utf-8') as f:
                # Ensure we have page_content
                if hasattr(doc, 'page_content'):
                    f.write(doc.page_content)
                else:
                    f.write(str(doc))
                
                # Save metadata if requested and available
                if include_metadata and hasattr(doc, 'metadata') and doc.metadata:
                    f.write("\n\n--- METADATA ---\n")
                    for key, value in doc.metadata.items():
                        f.write(f"{key}: {value}\n")
            
            saved_count += 1
        except Exception as e:
            print(f"Error saving document {i+1}: {str(e)}")
    
    print(f"Saved {saved_count} documents to {output_folder}")
    return output_folder, saved_count == len(documents)

def prepare_and_split_docs_by_sentence(directory, output_folder="processed_docs", window_size=DEFAULT_WINDOW_SIZE, 
                                       step_size=DEFAULT_STEP_SIZE, prefix="window", save=True):
    """
    Process documents from a directory, split them into sentence-level chunks with sliding window,
    and optionally save them to disk.
    
    Args:
        directory (str): Path to the directory containing documents
        output_folder (str): Path where processed documents will be saved
        window_size (int): Number of sentences per window
        step_size (int): Number of sentences to slide the window for each chunk
        prefix (str): Prefix for saved document filenames
        save (bool): Whether to save the documents to disk
        
    Returns:
        tuple: (list of Document objects or output directory path, success status)
    """
    try:
        # Download NLTK dependencies
        download_nltk_dependencies()
        
        # Create and use document loaders
        documents = load_docx_documents(directory)
        
        if not documents:
            print("No documents were found or loaded.")
            return None, False
        
        # Apply initial chunking
        pre_splitter = create_text_splitter()
        pre_split_docs = pre_splitter.split_documents(documents)
        
        # Apply sentence-level sliding window
        sentence_windows = []
        for doc in pre_split_docs:
            windows = apply_sentence_windows(doc, window_size, step_size)
            sentence_windows.extend(windows)
        
        print(f"Documents are split into {len(sentence_windows)} sentence windows")
        
        # Save documents if requested
        if save and sentence_windows:
            output_path, success = save_documents_to_folder(
                sentence_windows, 
                output_folder, 
                prefix=prefix
            )
            return output_path, success
        
        # Return the documents if not saving
        return sentence_windows, True
        
    except Exception as e:
        print(f"Error in document processing: {str(e)}")
        return None, False

# Example usage
if __name__ == "__main__":
    input_dir = "documents"  # Replace with your input directory
    output_dir = "processed_docs"  # Replace with your desired output directory
    
    result, success = prepare_and_split_docs_by_sentence(
        input_dir,
        output_dir,
        window_size=5,
        step_size=2,
        prefix="window"
    )
    
    if success:
        print(f"Document processing completed successfully. Results saved to: {result}")
    else:
        print("Document processing failed.")