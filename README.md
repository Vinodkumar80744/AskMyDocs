# AskMyDocs – AI-Powered PDF Question Answering System

AskMyDocs is an AI-powered document question-answering application built using **Retrieval-Augmented Generation (RAG)**.

Upload PDF documents, ask questions in natural language, retrieve relevant information from the pdfs using semantic search, and generate context-aware answers using a Large Language Model.

<p align="center">

**[🚀 Try AskMyDocs Live](https://askmydocs-diyexx8jehp2jf88hfg7hf.streamlit.app/)**

</p>

## Live Demo

🌐 **Application:** https://askmydocs-diyexx8jehp2jf88hfg7hf.streamlit.app/

The application is deployed using Streamlit Community Cloud.

## Features

- PDF document upload and management
- PDF text extraction using PyMuPDF
- Overlapping text chunking
- Semantic embeddings using Sentence Transformers
- Vector similarity search using FAISS
- AI-powered answer generation using Groq
- Document-based question answering
- Knowledge base overview
- Interactive Streamlit interface

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Web Framework | Streamlit |
| PDF Processing | PyMuPDF |
| Embedding Model | all-MiniLM-L6-v2 |
| Embedding Library | Sentence Transformers |
| Vector Database | FAISS |
| LLM Provider | Groq |
| LLM Model | openai/gpt-oss-20b |
| Environment Management | python-dotenv |

## RAG Architecture

AskMyDocs follows a Retrieval-Augmented Generation workflow:

1. **Document Ingestion:** Load PDF documents and extract text.
2. **Text Chunking:** Divide extracted text into smaller overlapping chunks.
3. **Embedding Generation:** Convert text chunks into numerical vectors.
4. **Vector Storage:** Store embeddings in FAISS.
5. **Query Processing:** Convert the user's question into an embedding.
6. **Semantic Retrieval:** Retrieve relevant document passages.
7. **Answer Generation:** Send the retrieved context and question to Groq.
8. **Response:** Generate an answer grounded in the retrieved document context.

## Project Structure

```text
AskMyDocs/
│
├── data/
│   └── documents/
│
├── src/
│   ├── chunking/
│   ├── embeddings/
│   ├── generation/
│   ├── ingestion/
│   ├── rag/
│   └── retrieval/
│
├── .streamlit/
├── app.py
├── main.py
├── rebuild_index.py
├── check_models.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Vinodkumar80744/AskMyDocs.git
cd AskMyDocs
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your API key from the [Groq Console](https://console.groq.com/keys).

**Never upload your `.env` file or expose your API key publicly.**

### 5. Add PDF documents

Place PDF files inside:

```text
data/documents/
```

### 6. Run the application

```bash
streamlit run app.py
```

## CLI Usage

Run the command-line version:

```bash
python main.py
```

## Rebuilding the Vector Index

After adding or replacing PDFs outside the application, rebuild the index:

```bash
python rebuild_index.py
```

## Deployment

AskMyDocs is deployed using **Streamlit Community Cloud**.

The deployment uses:

- GitHub repository: [Vinodkumar80744/AskMyDocs](https://github.com/Vinodkumar80744/AskMyDocs)
- Branch: `main`
- Entry point: `app.py`
- API configuration: Streamlit Secrets

## Future Improvements

- Improve source citations with exact PDF names and page numbers.
- Add persistent cloud storage for uploaded documents.
- Implement conversation memory.
- Support additional document formats.
- Explore retrieval reranking.

## Author

**Nathi Vinod Kumar**

Built with Python, Streamlit, FAISS, Sentence Transformers, and Groq.

---

If you find this project useful, consider giving the repository a ⭐ star!
