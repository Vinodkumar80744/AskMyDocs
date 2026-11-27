from pathlib import Path

from src.ingestion.pdf_loader import PDFLoader
from src.chunking.text_splitter import TextSplitter
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore


class IngestionPipeline:
    """Handles ingestion of multiple PDF documents."""

    def __init__(
        self,
        embedder: Embedder,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.embedder = embedder
        self.loader = PDFLoader()

        self.splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def ingest_directory(
        self,
        directory: str,
        vectorstore_path: str,
    ) -> VectorStore:

        directory_path = Path(directory)

        if not directory_path.exists():
            raise FileNotFoundError(
                f"Document directory not found: {directory_path}"
            )

        pdf_files = sorted(
            directory_path.glob("*.pdf")
        )

        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {directory_path}"
            )

        print(
            f"Found {len(pdf_files)} PDF(s)."
        )

        all_chunks = []

        # --------------------------------------------------
        # Process every PDF
        # --------------------------------------------------

        for pdf_path in pdf_files:

            print(
                f"\nProcessing: {pdf_path.name}"
            )

            documents = self.loader.load(
                str(pdf_path)
            )

            print(
                f"  Pages loaded: {len(documents)}"
            )

            chunks = self.splitter.split(
                documents
            )

            print(
                f"  Chunks created: {len(chunks)}"
            )

            all_chunks.extend(chunks)

        # --------------------------------------------------
        # Generate embeddings for all chunks
        # --------------------------------------------------

        texts = [
            chunk["text"]
            for chunk in all_chunks
        ]

        embeddings = self.embedder.encode(
            texts
        )

        print(
            f"\nTotal chunks: {len(all_chunks)}"
        )

        print(
            f"Embedding shape: {embeddings.shape}"
        )

        # --------------------------------------------------
        # Create FAISS vector store
        # --------------------------------------------------

        vector_store = VectorStore(
            dimension=embeddings.shape[1]
        )

        vector_store.add(
            embeddings,
            all_chunks
        )

        print(
            f"Vectors stored: "
            f"{vector_store.index.ntotal}"
        )

        # --------------------------------------------------
        # Save vector store
        # --------------------------------------------------

        vector_store.save(
            vectorstore_path
        )

        print(
            "Vector store saved."
        )

        return vector_store