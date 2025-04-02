import os
from sentence_transformers import SentenceTransformer

# Define the project directory where you want to save the model
project_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
model_cache_dir = os.path.join(project_dir, "model_cache")  # Define cache folder

# Ensure the directory exists
os.makedirs(model_cache_dir, exist_ok=True)

# Load the model and specify the cache folder
model = SentenceTransformer('sentence-transformers/all-distilroberta-v1', cache_folder=model_cache_dir)

print(f"Model downloaded to: {model_cache_dir}")


sentences = ["This is an example sentence", "Each sentence is converted"]
embeddings = model.encode(sentences)
print(embeddings)


