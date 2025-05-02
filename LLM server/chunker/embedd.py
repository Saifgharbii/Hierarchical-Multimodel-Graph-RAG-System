import os
import json
import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

# Configuration parameters
CONFIG = {
    # Directory settings
    "PROCESSED_DATA_DIR": "C:/Users/rimba/Desktop/p2m_repo/P2M/LLM server/chunker/second_process_per_batch/1",  # Input directory with chunks
    "OUTPUT_DATA_DIR": "C:/Users/rimba/Desktop/p2m_repo/P2M/LLM server/chunker/new_process/1",  # Will be set by user - where to store embeddings
    
    # Embedding model settings
    "MODEL_NAME": "BAAI/bge-m3",
    "USE_FP16": True,
    "RETURN_DENSE": True,
    "RETURN_SPARSE": True,  # Changed to True to get lexical weights
    "RETURN_COLBERT_VECS": False,

    # File patterns
    "CHUNK_FILE_PREFIX": "chunk_",
    "CHUNK_FILE_SUFFIX": ".txt",
    "EMBED_FILE_PREFIX": "embed_",
    "EMBED_FILE_SUFFIX": ".json",

    # Batch processing settings
    "BATCH_SIZE": 4,  # Slightly increased for efficiency

    # Performance settings
    "USE_CUDA": torch.cuda.is_available(),

    # Logging verbosity
    "VERBOSE": True
}

def convert_float16_to_float32(output):
    for key, value in output.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.float16:
                output[key] = value.astype(np.float32)  # Convert to float32
    return output


def initialize_model():
    """Initialize the embedding model"""
    model = BGEM3FlagModel(
        CONFIG["MODEL_NAME"],
        use_fp16=CONFIG["USE_FP16"]
    )
    return model

def read_chunk(file_path):
    """Read content from a chunk file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def save_embedding(embedding, output_path):
    """Save embedding as JSON file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_embedding = {}

    for key, value in embedding.items():
        if isinstance(value, np.ndarray):
            # Convert float16 to float32 before serialization
            if value.dtype == np.float16:
                value = value.astype(np.float32)
            serializable_embedding[key] = value.tolist()
        elif isinstance(value, torch.Tensor):
            tensor_value = value.cpu()
            # Convert float16 tensors to float32
            if tensor_value.dtype == torch.float16:
                tensor_value = tensor_value.float()
            serializable_embedding[key] = tensor_value.tolist()
        else:
            serializable_embedding[key] = value

    # Save the embedding
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_embedding, f)

def process_embeddings(model, texts):
    """Process a batch of texts to generate embeddings - return only dense vectors"""
    try:
        # Generate embeddings using the model's encode method
        output = model.encode(
            texts,
            return_dense=CONFIG["RETURN_DENSE"],
            return_sparse=CONFIG["RETURN_SPARSE"],
            return_colbert_vecs=CONFIG["RETURN_COLBERT_VECS"]
        )

        if CONFIG["VERBOSE"]:
            print(f"Processed {len(texts)} texts")
            print(f"Output type: {type(output)}")
            print(f"Output keys: {list(output.keys())}")

        # Extract individual embeddings for each text - only dense vectors
        chunk_embeddings = []
        batch_size = len(texts)

        for i in range(batch_size):
            # Create an embedding dict with only dense vectors
            embedding = {}

            # Extract dense vectors if available
            if 'dense_vecs' in output and output['dense_vecs'] is not None:
                embedding['chunk'] = texts[i]
                embedding['embedding'] = output['dense_vecs'][i]

            # No lexical_weights or colbert_vecs added

            chunk_embeddings.append(embedding)

        return chunk_embeddings
    except Exception as e:
        print(f"Error processing embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def get_relative_path(full_path, base_path):
    """Get relative path from base_path to full_path"""
    return os.path.relpath(full_path, base_path)

def process_and_store_data(input_dir=None, output_dir=None, model=None):
    """Process all chunk files and save their embeddings"""
    # Use default from CONFIG if not provided
    input_dir = input_dir or CONFIG["PROCESSED_DATA_DIR"]
    output_dir = output_dir or CONFIG["OUTPUT_DATA_DIR"]
    model = model or initialize_model()
    
    if not output_dir:
        raise ValueError("Output directory must be specified")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all chunk files
    chunk_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if (file.startswith(CONFIG["CHUNK_FILE_PREFIX"]) and
                file.endswith(CONFIG["CHUNK_FILE_SUFFIX"])):
                chunk_files.append(os.path.join(root, file))

    if CONFIG["VERBOSE"]:
        print(f"Found {len(chunk_files)} chunk files")

    # Group chunk files by directory to maintain hierarchy
    directory_chunks = {}
    for chunk_file in chunk_files:
        dir_path = os.path.dirname(chunk_file)
        if dir_path not in directory_chunks:
            directory_chunks[dir_path] = []
        directory_chunks[dir_path].append(chunk_file)

    # Process each directory separately
    for dir_path, chunks in tqdm(directory_chunks.items(), desc="Processing directories"):
        # Calculate level in hierarchy
        parts = dir_path.split(os.sep)
        level = len(parts) - parts.index(os.path.basename(input_dir)) if os.path.basename(input_dir) in parts else len(parts)

        # Calculate corresponding output directory that mirrors the input structure
        rel_path = get_relative_path(dir_path, input_dir)
        output_dir_path = os.path.join(output_dir, rel_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir_path, exist_ok=True)

        if CONFIG["VERBOSE"]:
            print(f"Processing directory: {dir_path} (Level: {level})")
            print(f"Output directory: {output_dir_path}")

        # Process chunks in batches
        for i in tqdm(range(0, len(chunks), CONFIG["BATCH_SIZE"]),
                     desc=f"Processing chunks in {dir_path}", leave=False):
            # Get batch of chunk files
            batch_chunks = chunks[i:i + CONFIG["BATCH_SIZE"]]

            # Prepare batch data
            batch_texts = []
            batch_files = []

            for chunk_file in batch_chunks:
                # Generate embedding file path
                base_name = os.path.basename(chunk_file)
                chunk_id = base_name.replace(CONFIG["CHUNK_FILE_PREFIX"], "").replace(CONFIG["CHUNK_FILE_SUFFIX"], "")
                embed_name = f"{CONFIG['EMBED_FILE_PREFIX']}{chunk_id}{CONFIG['EMBED_FILE_SUFFIX']}"
                embed_path = os.path.join(output_dir_path, embed_name)

                # Check if embedding already exists
                if os.path.exists(embed_path):
                    if CONFIG["VERBOSE"]:
                        print(f"Embedding already exists for {chunk_file}, skipping")
                    continue

                try:
                    # Read chunk text
                    chunk_text = read_chunk(chunk_file)

                    # Add to batch
                    batch_texts.append(chunk_text)
                    batch_files.append((chunk_file, embed_path))
                except Exception as e:
                    print(f"Error reading {chunk_file}: {str(e)}")

            # Process batch if not empty
            if batch_texts:
                try:
                    # Process embeddings for this batch
                    embeddings = process_embeddings(model, batch_texts)

                    # Save embeddings to files
                    if len(embeddings) > 0:
                        for j, (chunk_file, embed_path) in enumerate(batch_files):
                            if j < len(embeddings):
                                if CONFIG["VERBOSE"]:
                                    print(f"Saving embedding for {chunk_file} to {embed_path}")
                                save_embedding(embeddings[j], embed_path)
                            else:
                                print(f"Warning: No embedding generated for {chunk_file}")
                    else:
                        print(f"Warning: No embeddings generated for batch")
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    import traceback
                    traceback.print_exc()

    if CONFIG["VERBOSE"]:
        print("All data processed and embeddings stored")

def test_model_output(model):
    """Test the model output format with a simple example"""
    test_texts = ["This is a test sentence to check embedding format.",
                 "A second test sentence to verify batch processing."]
    print(f"Testing model with {len(test_texts)} texts")

    try:
        # Generate embeddings using the model's encode method with correct parameters
        output = model.encode(
            test_texts,
            return_dense=CONFIG["RETURN_DENSE"],
            return_sparse=CONFIG["RETURN_SPARSE"],
            return_colbert_vecs=CONFIG["RETURN_COLBERT_VECS"]
        )

        print(f"Output type: {type(output)}")
        print(f"Output keys: {list(output.keys())}")

        # Print information about each component
        for key, value in output.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: type={type(value)}, shape={value.shape}, dtype={value.dtype}")
                # Print a sample of the data
                if len(value.shape) == 2:
                    print(f"  Sample (first row, first 5 elements): {value[0, :5]}")
                elif len(value.shape) > 2:
                    print(f"  Sample (first item, first row, first 5 elements): {value[0, 0, :5]}")
            else:
                print(f"{key}: type={type(value)}")

        # Process embeddings using our function
        processed = process_embeddings(model, test_texts)
        print(f"\nProcessed embeddings: {len(processed)}")
        print(f"First embedding keys: {list(processed[0].keys())}")
        for key, value in processed[0].items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: type={type(value)}, shape={value.shape}")
            else:
                print(f"  {key}: type={type(value)}")

        return output, processed
    except Exception as e:
        print(f"Error in test_model_output: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def display_embeddings(root_dir=None):
    """Display the structure of embeddings stored on disk"""
    root_dir = root_dir or CONFIG["OUTPUT_DATA_DIR"] or CONFIG["PROCESSED_DATA_DIR"]

    print("\n=== Embedding Files Structure ===")
    print(f"Root directory: {root_dir}")

    # Check if directory exists
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} does not exist!")
        return

    # Count statistics
    total_chunks = 0
    total_embeddings = 0

    # Walk through directory structure
    for root, dirs, files in os.walk(root_dir):
        # Calculate level in hierarchy
        rel_path = os.path.relpath(root, root_dir)
        level = len(rel_path.split(os.sep)) if rel_path != "." else 1
        indent = "  " * (level - 1)

        # Print directory name
        if root != root_dir:
            print(f"{indent}├── {os.path.basename(root)}/")
        else:
            print(f"{os.path.basename(root)}/")

        # Count and display embedding files
        embed_files = [f for f in files if f.startswith(CONFIG["EMBED_FILE_PREFIX"]) and f.endswith(CONFIG["EMBED_FILE_SUFFIX"])]

        total_embeddings += len(embed_files)

        # Display files with indentation
        if embed_files:
            # Sort the files to display them in order
            embed_files.sort()

            # Print a sample of embedding files
            max_display = 5  # Limit number of files to display
            for i, embed_file in enumerate(embed_files[:max_display]):
                print(f"{indent}│   ├── {embed_file}")

            # Show summary if more files than displayed
            if len(embed_files) > max_display:
                print(f"{indent}│   └── ... and {len(embed_files) - max_display} more embedding files")

    print("\n" + "=" * 50)
    print(f"Summary: {total_embeddings} embedding files")
    print("=" * 50)

def main():
    # Initialize the model
    model = initialize_model()

    # Get input and output directories
    print("\n=== Embedding Processor Configuration ===")
    
    
    
    print(f"\nInput directory: {CONFIG['PROCESSED_DATA_DIR']}")
    print(f"Output directory: {CONFIG['OUTPUT_DATA_DIR']}")

    # Provide options to the user
    print("\nChoose an action:")
    print("1: Process Data (generate and store embeddings)")
    print("2: Configure Parameters")
    print("3: Display Embedding Structure")
    print("4: Test Model Output")

    action = input("\nEnter your choice (1-4): ").strip()

    if action == "1":
        # Process all directories and store embeddings
        print("\nProcessing data and generating embeddings...")
        process_and_store_data(CONFIG["PROCESSED_DATA_DIR"], CONFIG["OUTPUT_DATA_DIR"], model)
        
        # Display the structure after processing
        print("\nEmbedding structure after processing:")
        display_embeddings(CONFIG["OUTPUT_DATA_DIR"])

    elif action == "2":
        # Configure parameters
        print("\nCurrent Configuration:")
        for key, value in CONFIG.items():
            print(f"{key}: {value}")

        print("\nEnter new values (leave blank to keep current value):")
        for key in CONFIG:
            if key in ["USE_CUDA", "VERBOSE", "USE_FP16", "RETURN_DENSE", "RETURN_SPARSE", "RETURN_COLBERT_VECS"]:
                new_value = input(f"New value for {key} (True/False): ").strip()
                if new_value.lower() in ['true', 'false']:
                    CONFIG[key] = new_value.lower() == 'true'
            elif key in ["BATCH_SIZE"]:
                new_value = input(f"New value for {key}: ").strip()
                if new_value.isdigit():
                    CONFIG[key] = int(new_value)
            else:
                new_value = input(f"New value for {key}: ").strip()
                if new_value:
                    CONFIG[key] = new_value

        print("\nUpdated Configuration:")
        for key, value in CONFIG.items():
            print(f"{key}: {value}")

        # Continue to main menu
        main()

    elif action == "3":
        # Display embedding structure
        print("\n=== Detailed Embedding Structure ===")
        display_embeddings(CONFIG["OUTPUT_DATA_DIR"])

    elif action == "4":
        # Test model output
        print("\nTesting model output format...")
        raw_output, processed_output = test_model_output(model)

        # Display a sample embedding
        if processed_output and len(processed_output) > 0:
            print("\nSample processed embedding:")
            embedding = processed_output[0]

            for key, value in embedding.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    print(f"    First 5 values: {value.flatten()[:5]}")
                else:
                    print(f"  {key}: {type(value)}")

    else:
        print("Invalid action selected")

    # Ask if user wants to continue
    continue_choice = input("\nDo you want to continue (y/n)? ").strip().lower()
    if continue_choice == 'y':
        main()
    else:
        print("Exiting program. Goodbye!")

if __name__ == "__main__":
    main()