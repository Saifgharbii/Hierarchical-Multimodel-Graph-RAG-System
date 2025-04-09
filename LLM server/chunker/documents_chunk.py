import os
from chunk import prepare_and_split_docs_by_sentence

def main():
    # Debug: Print starting message
    print("Starting the document processing script...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output paths
    data_dir = os.path.join(base_dir, "data")
    output_folder = os.path.join(base_dir, "processed_data")

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist")
        return

    # Debug: Print data directory path
    print(f"Data directory exists: {data_dir}")

    # Run the document processing
    print("Running the document processing function...")
    result, success = prepare_and_split_docs_by_sentence(
        directory=data_dir,  
        output_folder=output_folder,
        window_size=5,
        step_size=2,
        prefix="chunk",
        save=True
    )

    # Report results
    if success:
        print("Document processing completed successfully!")
        print(f"Processed documents saved to: {result}")
    else:
        print("Document processing failed.")

if __name__ == "__main__":
    main()
