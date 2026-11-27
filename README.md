# RAG Studio – AI-Powered PDF Question Answering System

RAG Studio is an AI-powered PDF question-answering application built using **Retrieval-Augmented Generation (RAG)**. It allows users to upload PDF documents, retrieve relevant information using semantic search, and generate context-aware answers using a Large Language Model (LLM).

The application combines document processing, vector embeddings, FAISS similarity search, and Groq-powered answer generation to provide an interactive document-based AI assistant.

## Features

- **PDF Upload:** Upload and manage multiple PDF documents.
- **Document Processing:** Extract text from PDFs using PyMuPDF.
- **Text Chunking:** Split documents into smaller overlapping chunks.
- **Semantic Embeddings:** Generate text embeddings using Sentence Transformers.
- **Vector Search:** Store and retrieve document embeddings using FAISS.
- **AI-Powered Answers:** Generate answers using the Groq API.
- **Source-Based Responses:** Retrieve relevant document passages to support answers.
- **Knowledge Base:** View document and vector information.
- **Interactive Interface:** Use a Streamlit-based web application.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| PDF Processing | PyMuPDF |
| Text Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| LLM API | Groq |
| LLM Model | openai/gpt-oss-20b |
| Environment Management | Python venv, python-dotenv |

## RAG Architecture

The application follows a Retrieval-Augmented Generation pipeline.

1. **Document Ingestion:** Load PDF documents and extract their text.
2. **Text Chunking:** Divide extracted text into manageable chunks.
3. **Embedding Generation:** Convert text chunks into numerical vectors.
4. **Vector Storage:** Store embeddings in a FAISS index.
5. **Query Processing:** Convert the user's question into an embedding.
6. **Semantic Retrieval:** Retrieve relevant document chunks using similarity search.
7. **Answer Generation:** Send the retrieved context and question to Groq.
8. **Response:** Generate an answer grounded in the retrieved document content.

## Project Structure

```text
Rag_project/
│
├── data/
│   └── documents/
│       └── Sample PDF documents
│
├── src/
│   ├── chunking/
│   │   └── text_splitter.py
│   │
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   ├── generation/
│   │   └── llm.py
│   │
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   └── ingestion_pipeline.py
│   │
│   ├── rag/
│   │   └── pipeline.py
│   │
│   └── retrieval/
│       └── vector_store.py
│
├── .streamlit/
│   └── config.toml
│
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
git clone YOUR_GITHUB_REPOSITORY_URL
cd Rag_project
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

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

Get your API key from the [Groq Console](https://console.groq.com/).

**Important:** Never upload your `.env` file or expose your API key publicly.

### 5. Add PDF documents

Place your PDF documents inside:

```text
data/documents/
```

### 6. Run the application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## Running the CLI Version

You can also run the RAG pipeline through:

```bash
python main.py
```

## Rebuilding the Vector Index

If you add or replace PDF documents outside the application, rebuild the vector index using:

```bash
python rebuild_index.py
```

The application can also rebuild its index when documents are managed through its interface.

## Deployment

RAG Studio is designed to be deployable using **Streamlit Community Cloud**.

General deployment steps:

1. Push the project to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Create a new application.
4. Select your GitHub repository.
5. Set `app.py` as the application entry point.
6. Add your `GROQ_API_KEY` under application Secrets.
7. Deploy the application.

Example Streamlit Cloud secret:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

## Future Improvements

- Support additional document formats.
- Improve retrieval accuracy through reranking.
- Add conversation memory.
- Support more embedding models and LLM providers.
- Add persistent cloud storage for uploaded documents.
- Improve source citations and document references.

## Author

**Nathi Vinod Kumar**

Built with Python, FAISS, Sentence Transformers, Groq, and Streamlit.

---

If you find this project useful, consider giving the repository a star!
