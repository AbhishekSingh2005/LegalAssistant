#RAG-Powered Legal Assistant

A privacy-focused Legal AI assistant that provides accurate, citation-based answers to legal questions using the Indian legal system, including the Constitution of India and other legal texts. Built with Retrieval-Augmented Generation (RAG) for reliable, source-grounded responses.

## 🚀 Features

- **🔍 Semantic Search** - Advanced embeddings for precise legal document retrieval
- **🧠 Local LLM** - Powered by Ollama (supports LLaMA, Mistral, and other models)
- **📚 Multi-format Support** - Process PDFs, Word docs, text files, and web pages
- **🔐 Privacy-First** - 100% private, runs locally on your machine
- **⚖️ Legal-Specific** - Optimized for legal document processing and citation
- **🌐 Web Scraping** - Directly ingest legal documents from the web
- **💬 Interactive UI** - User-friendly Streamlit interface

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/legal-ai-rag.git
   cd legal-ai-rag
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Ollama**:
   - Install from [ollama.ai](https://ollama.ai/)
   - Download a model:
     ```bash
     ollama pull llama3.2
     ```

## Quick Start

1. **Add legal documents** to the `source_documents/` folder

2. **Process documents**:
   ```bash
   python ingest.py
   ```

3. **Launch the assistant**:
   ```bash
   streamlit run privateGPT.py
   ```

4. **Access the web interface** at `http://localhost:8501`

## 📂 Project Structure

```
legal-ai-rag/
├── data/              # Legal documents storage
├── db/                # Vector database
├── source_documents/  # Documents for ingestion
├── constants.py       # Configuration
├── ingest.py          # Document processing
├── privateGPT.py      # Main application
└── requirements.txt   # Dependencies
```

## Important Notes

- This tool is for educational purposes only
- Not a substitute for professional legal advice
- Always verify critical legal information with qualified professionals

## 📄 License

MIT
