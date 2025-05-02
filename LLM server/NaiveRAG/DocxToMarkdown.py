# This script converts all .docx files in the specified directory to .md files using pypandoc.
# It uses the tqdm library to show a progress bar for the conversion process.
# Make sure to install the required libraries:
# pip install pypandoc tqdm
# Note: You may need to install pandoc separately for pypandoc to work.
# The script assumes that the input files are in a folder named "DocxDataset" and the output files will be saved in a folder named "MarkdownFiles".
# The script creates the output folder if it doesn't exist.

import pypandoc
import os
from tqdm import tqdm

dataset_path = "../EmbeddingProcess/DocxDataset"
output_markdown = "./MarkdownFiles"
os.makedirs(output_markdown, exist_ok=True)

for docx_file in tqdm(iterable=os.listdir(dataset_path), desc="Processing files", unit="file"):
    # Check if the file is a DOCX file
    if docx_file.endswith(".docx"):
        docx_path = os.path.join(dataset_path, docx_file)
        output_markdown = os.path.join(output_markdown, f"{docx_file.replace(".docx", "")}.md")
        markdown_text = pypandoc.convert_file(docx_path, 'markdown')
        with open(output_markdown, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        print(f"Converted {docx_file} to Markdown format.")


