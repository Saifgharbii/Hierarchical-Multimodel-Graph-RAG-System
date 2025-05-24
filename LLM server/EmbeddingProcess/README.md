# EmbeddingProcess

This directory contains the complete preprocessing pipeline for a RAG (Retrieval-Augmented Generation) system. It handles document processing from raw DOCX files to embedded, searchable chunks ready for retrieval.

## Overview

The EmbeddingProcess pipeline transforms documents through several stages:

1. **Text Extraction** - Extract text and images from DOCX files
2. **Table Extraction** - Extract and structure tables from documents
3. **Document Chunking** - Split documents into manageable chunks
4. **Embedding Generation** - Create vector embeddings for all content
5. **Retrieval Setup** - Prepare documents for search and retrieval

## Directory Structure

```
EmbeddingProcess/
├── TextExtraction/          # Extract text and process images from documents
│   ├── ImageOCR_Server/     # OCR server for image text extraction
│   │   └── README.md        # 📖 [OCR Server Documentation](./TextExtraction/ImageOCR_Server/README.md)
│   ├── ImageProcessing/     # Image manipulation and processing modules
│   │   └── README.md        # 📖 [Image Processing Documentation](./TextExtraction/ImageProcessing/README.md)
│   ├── README.md           # 📖 [Text Extraction Documentation](./TextExtraction/README.md)
│   └── Text_extraction.py   # Main text extraction pipeline
├── TableExtraction/         # Extract and structure tables from documents
│   ├── Method 1 without infos/  # Basic table extraction
│   ├── Method 2 with infos/     # Enhanced table extraction with metadata
│   └── JsonTables/          # Output JSON files with extracted tables
├── DocumentsChunking/       # Split documents into chunks for processing
│   ├── README.md           # 📖 [Document Chunking Documentation](./DocumentsChunking/README.md)
│   ├── json_chunking.py     # Main chunking logic
│   └── token_length_distribution.png  # Visualization of chunk sizes
├── Retriever/              # Search and retrieval functionality
│   ├── retrieve.py         # Hierarchical search engine
│   └── setup.py           # Retriever setup script
├── INSTALL.md              # 📖 [Installation Guide](./INSTALL.md)
├── document_embedder.py    # Main embedding generation script
├── embedder_server.py      # Flask API server for embedding services
└── requirements.txt        # Python dependencies
```

## Core Components

### 1. Document Embedder (`document_embedder.py`)

The main component that generates vector embeddings for document content using the Stella embedding model.

**Features:**

- Multi-GPU support for faster processing
- Batch processing for efficiency
- Hierarchical document structure preservation
- Concurrent processing across devices

**Key Classes:**

- `DocumentEmbedder`: Main embedding class with multi-device support
- Supports both GPU and CPU processing
- Automatic memory management and cleanup

**Usage:**

```python
from document_embedder import DocumentEmbedder

embedder = DocumentEmbedder(
    model_name="dunzhang/stella_en_400M_v5",
    batch_size=32,
    max_length=2048
)

# Embed texts
embeddings = embedder.embed_texts(["text1", "text2", ...])
embedder.cleanup()  # Important for memory management
```

### 2. Embedding Server (`embedder_server.py`)

Flask API server providing embedding and retrieval services.

**Endpoints:**

- `POST /embed` - Generate embeddings for input texts
- `POST /retrieve` - Retrieve relevant documents based on query

**API Usage:**

```bash
# Embed texts
curl -X POST http://localhost:5003/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Your text here"]}'

# Retrieve documents
curl -X POST http://localhost:5003/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "search query", "user_query": "original user question"}'
```

### 3. Text Extraction Pipeline

📖 **[Detailed Documentation](./TextExtraction/README.md)**

**Components:**

- **Text_extraction.py**: Main pipeline for extracting text from DOCX files
- **ImageOCR_Server**: Handles OCR for images within documents → [📖 OCR Server Guide](./TextExtraction/ImageOCR_Server/README.md)
- **ImageProcessing**: Image manipulation and processing modules → [📖 Image Processing Guide](./TextExtraction/ImageProcessing/README.md)

**Features:**

- DOCX document parsing
- Image extraction and OCR processing
- Figure and table detection
- Structured JSON output

### 4. Table Extraction

**Two Methods Available:**

- **Method 1**: Basic table extraction without metadata
- **Method 2**: Enhanced extraction with additional context information

**Output**: Structured JSON files containing table data and metadata

### 5. Document Chunking

📖 **[Detailed Documentation](./DocumentsChunking/README.md)**

**Features:**

- Intelligent text chunking based on document structure
- Token length optimization
- Hierarchical preservation (sections, subsections, etc.)
- Visual analysis of chunk size distribution

### 6. Retrieval System

**Features:**

- Hierarchical search engine
- Vector similarity search
- Re-ranking capabilities
- Context-aware retrieval

## Installation

📖 **[Complete Installation Guide](./INSTALL.md)**

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. For specific components, check their individual documentation:
   - **General Setup**: [📖 INSTALL.md](./INSTALL.md) - Detailed setup instructions
   - **OCR Server**: [📖 ImageOCR_Server/README.md](./TextExtraction/ImageOCR_Server/README.md) - OCR server setup
   - **Image Processing**: [📖 ImageProcessing/README.md](./TextExtraction/ImageProcessing/README.md) - Image processing setup
   - **Document Chunking**: [📖 DocumentsChunking/README.md](./DocumentsChunking/README.md) - Chunking configuration

## Quick Start Guide

### 🚀 Complete Pipeline Workflow

For detailed instructions on each step, refer to the component-specific documentation:

1. **📄 Text Extraction** → [📖 Guide](./TextExtraction/README.md)

   - Convert DOCX files to structured JSON
   - Extract images and apply OCR → [📖 OCR Setup](./TextExtraction/ImageOCR_Server/README.md)

2. **📊 Table Extraction**

   - Extract and structure tables from documents
   - Choose between Method 1 (basic) or Method 2 (enhanced)

3. **✂️ Document Chunking** → [📖 Guide](./DocumentsChunking/README.md)

   - Split content into manageable, searchable chunks
   - Optimize chunk sizes for your use case

4. **🔢 Generate Embeddings**

   - Create vector embeddings using the main pipeline
   - Multi-GPU processing for faster results

5. **🔍 Setup Retrieval**
   - Configure hierarchical search engine
   - Test retrieval functionality

### Running the Embedding Server

```bash
python embedder_server.py
```

The server will start on port 5003 (configurable via PORT environment variable).

### Processing Local Files

```python
from document_embedder import main

# Process all JSON files in a directory
main(
    input_path="./DocumentsChunking/ChunkedDocuments",
    model_name="dunzhang/stella_en_400M_v5",
    batch_size=64,
    max_length=2048
)
```

## Configuration

### Model Configuration

- **Default Model**: `dunzhang/stella_en_400M_v5`
- **Batch Size**: 32-64 (adjust based on GPU memory)
- **Max Length**: 2048 tokens
- **Multi-GPU**: Automatically detected and utilized

### Environment Variables

- `PORT`: Server port (default: 5003)

## Data Flow and Output Formats

### Input Format (Before Chunking)

Documents start as structured JSON files from the text extraction pipeline:

```json
{
  "document_name": "example.docx",
  "content": [
    {
      "title": "Section Title",
      "description": "Section text content...",
      "summary": "",
      "tables": [
        {
          "description": "Table description",
          "table number": 1,
          "summary": "",
          "name": ""
        }
      ],
      "figures_meta_data": [
        {
          "figure_id": "<<<Figure 1: Example>>>",
          "last_paragraph": "Text preceding the figure..."
        }
      ],
      "subsections": [
        {
          "title": "Subsection Title",
          "description": "",
          "summary": "",
          "text_content": "Subsection content...",
          "tables": [],
          "figures_meta_data": [],
          "subsubsections": [
            {
              "title": "Subsubsection Title",
              "text_content": "Subsubsection content...",
              "figures_meta_data": [],
              "tables": []
            }
          ]
        }
      ]
    }
  ]
}
```

### After Chunking (Intermediate Format)

The document is split into manageable chunks while preserving structure → [📖 Chunking Details](./DocumentsChunking/README.md)

### Final Output Format (After Embedding)

The final embedded documents contain vector embeddings for all textual content:

```json
{
    "document_name": "example.docx",
    "content": [
        {
            "title": {
                "title": "Section Title",
                "embedding": [0.1, 0.2, -0.3, ...]
            },
            "description": {
                "description": "Section text content...",
                "embedding": [0.2, -0.1, 0.4, ...]
            },
            "chunks": [
                {
                    "chunk": "Subsection chunk",
                    "embedding": [0.2, 0.1, ...]
                }
            ],
            "tables": [
                {
                    "description": {
                        "description": "Table description",
                        "embedding": [0.0, 0.1, 0.2, ...]
                    },
                    "table number": 1,
                    "summary": "",
                    "name": ""
                }
            ],
            "subsections": [
                {
                    "title": {
                        "title": "Subsection Title",
                        "embedding": [0.1, 0.2, ...]
                    },
                    "text_content": "Original content",
                    "chunks": [
                        {
                            "chunk": "Subsection chunk",
                            "embedding": [0.2, 0.1, ...]
                        }
                    ],
                    "subsubsections" :[
                      {
                        "title": {
                        "title": "Subsection Title",
                        "embedding": [0.1, 0.2, ...]
                        },
                        "text_content": "Original content",
                        "chunks": [
                            {
                                "chunk": "Subsection chunk",
                                "embedding": [0.2, 0.1, ...]
                            }
                        ]
                      }

                    ]
                }
            ]
        }
    ]
}
```

### Processing Pipeline

1. **Text Extraction** → Structured JSON with document hierarchy
2. **Table Extraction** → Enhanced JSON with table metadata
3. **Document Chunking** → Split into searchable chunks
4. **Embedding Generation** → Add vector embeddings to all text fields
5. **Ready for Retrieval** → Searchable document store

## Performance Considerations

- **GPU Memory**: Monitor GPU usage during batch processing
- **Batch Size**: Adjust based on available memory (32-64 recommended)
- **Concurrent Processing**: Automatically distributes across available GPUs
- **Memory Cleanup**: Always call `embedder.cleanup()` after processing

## Dependencies

Key dependencies include:

- PyTorch (with CUDA support recommended)
- Transformers (Hugging Face)
- Flask (for API server)
- Various document processing libraries

See `requirements.txt` for complete list.

## Detailed Component Documentation

### 📚 Component-Specific Guides

Each component has detailed documentation with specific setup instructions, usage examples, and troubleshooting:

| Component             | Documentation                                                              | Description                                  |
| --------------------- | -------------------------------------------------------------------------- | -------------------------------------------- |
| **General Setup**     | [📖 INSTALL.md](./INSTALL.md)                                              | Complete installation and environment setup  |
| **Text Extraction**   | [📖 TextExtraction/README.md](./TextExtraction/README.md)                  | DOCX processing and text extraction pipeline |
| **OCR Server**        | [📖 ImageOCR_Server/README.md](./TextExtraction/ImageOCR_Server/README.md) | Image OCR processing server setup and API    |
| **Image Processing**  | [📖 ImageProcessing/README.md](./TextExtraction/ImageProcessing/README.md) | Image manipulation and processing modules    |
| **Document Chunking** | [📖 DocumentsChunking/README.md](./DocumentsChunking/README.md)            | Text chunking strategies and configuration   |

### 🛠️ Advanced Configuration

For advanced users who want to customize the pipeline behavior, each component's documentation includes:

- Configuration parameters
- Performance tuning options
- Integration examples
- API specifications
- Troubleshooting guides

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch_size parameter
2. **Model Loading Issues**: Ensure transformers library is up to date
3. **File Not Found**: Check input paths and file permissions
4. **Server Connection**: Verify port availability and firewall settings

### Memory Management

The system includes automatic memory management:

- GPU memory is cleared after processing
- Models are moved to CPU when not in use
- Garbage collection is triggered automatically

## Contributing

When adding new features:

1. Follow the existing code structure
2. Add appropriate error handling
3. Include memory cleanup in new components
4. Update documentation as needed
