```yaml
---
title: Chat with PDF
emoji: 💬
colorFrom: indigo
colorTo: violet
sdk: gradio
sdk_version: "5.35.0"
app_file: app.py
pinned: false
---
```

# Chat with PDF: Semantic Q&A with RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

An interactive semantic search Q&A system that enables natural language queries over PDF documents using Retrieval-Augmented Generation (RAG).

## Overview

This project implements a complete RAG pipeline for document question-answering:
- **Document Processing:** PDF parsing and chunking with semantic overlap
- **Embedding Generation:** Dense vector representations using HuggingFace models
- **Vector Search:** FAISS for fast similarity search across document chunks
- **LLM Integration:** Open-source language models (Flan-T5) for answer generation
- **Conversation Memory:** Multi-turn dialogue with context retention

### Architecture
```
PDF Upload → Text Extraction → Chunking → Embedding Generation 
                                               ↓
User Query → Query Embedding → FAISS Search → Relevant Chunks
                                               ↓
                           Context + Query → LLM → Answer + Sources
```

## Key Features

✅ **End-to-end RAG pipeline** built from scratch  
✅ **Semantic search** using FAISS vector similarity  
✅ **Context-aware answers** with source attribution  
✅ **Conversation memory** for multi-turn interactions  
✅ **Production-ready deployment** on HuggingFace Spaces  
✅ **Open-source models** (no API keys required for demo)

## Technical Stack

- **LangChain:** RAG orchestration and document processing
- **HuggingFace Transformers:** Embeddings and language models
- **FAISS:** Fast approximate nearest neighbor search
- **Gradio:** Interactive web interface
- **Python 3.8+**

## Project Structure
```
chat-with-pdf/
├── app.py               # Entry point and Gradio app initialization
├── chain.py             # RAG pipeline: embeddings, vector store, LLM chain
├── interface.py         # UI components and event handlers
├── utils.py             # Shared state management and reset logic
├── requirements.txt     # Python dependencies
├── README.md            # Documentation (you're here!)
├── CHANGELOG.md         # Version history
└── LICENSE              # MIT license
```

## Local Development

### Installation
```bash
# Clone repository
git clone https://github.com/serasr/Chat-with-PDF.git
cd Chat-with-PDF

# Install dependencies
pip install -r requirements.txt
```

### Run Locally
```bash
python app.py
```

Visit `http://localhost:7860` in your browser.

### Usage

1. Upload a PDF document
2. Wait for processing (embedding generation)
3. Ask questions in natural language
4. View answers with source citations
5. Ask follow-up questions (conversation context maintained)

## Deploy to HuggingFace Spaces

This app is designed for easy deployment to [HuggingFace Spaces](https://huggingface.co/spaces):

1. Create a new Space (SDK: Gradio, SDK version: 5.35.0)
2. Upload all project files
3. Space will automatically build and deploy
4. Access at: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

Deployment config is managed via YAML front matter in this README.

## Technical Details

### RAG Pipeline Components

**1. Document Processing:**
- PDF text extraction using PyPDF2/pdfplumber
- Text chunking with configurable overlap (prevents context loss at boundaries)
- Metadata preservation for source attribution

**2. Embedding Generation:**
- Uses sentence-transformers for dense embeddings
- Supports multiple embedding models (configurable)
- Batch processing for efficiency

**3. Vector Store:**
- FAISS index for fast approximate nearest neighbor search  
- Uses L2 (Euclidean) distance on normalized embeddings from sentence-transformers/all-MiniLM-L6-v2
- Retrieves top-2 most relevant document chunks per query (configurable via search_kwargs)

**4. Answer Generation:**
- Flan-T5-large for open-source generation
- Easily swappable with other HuggingFace models
- Context window management to fit relevant chunks

**5. Conversation Memory:**
- Tracks dialogue history
- Maintains context across multiple questions
- Configurable memory length

## Performance

- **Embedding Generation:** ~2-5 seconds for typical PDFs (10-50 pages)
- **Query Response:** <2 seconds on CPU
- **Memory Usage:** ~2-4 GB RAM (depending on PDF size)

## Roadmap

Future enhancements planned:

- [ ] Multi-PDF support (query across multiple documents)
- [ ] Embedding caching for faster re-uploads
- [ ] Highlighted source text in answers
- [ ] User feedback mechanism (thumbs up/down)
- [ ] Chat history export (Markdown/PDF)
- [ ] Multilingual support
- [ ] Advanced chunking strategies (semantic splitting)
- [ ] Model comparison (A/B testing different LLMs)

## Use Cases

This RAG system is applicable to:
- Research paper analysis
- Legal document review
- Technical documentation Q&A
- Educational material comprehension
- Knowledge base search

## Contributing

Contributions welcome! Areas of interest:
- Performance optimization
- Additional model integrations
- UI/UX improvements
- Documentation enhancements

Please open an issue or PR to discuss.

## License

Released under [MIT License](LICENSE). Free to use, modify, and distribute.

## Acknowledgments

Built with:
- [LangChain](https://www.langchain.com/) for RAG orchestration
- [HuggingFace](https://huggingface.co/) for models and deployment
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Gradio](https://www.gradio.app/) for UI

## Contact

**Sera Singha Roy**  

For questions about this project or my broader research on AI safety and uncertainty quantification, feel free to reach out.

---

*This project demonstrates practical implementation of RAG architecture for semantic document search, a foundational technique for building retrieval-augmented LLM applications.*


