
from src.embeddings.embedder import Embedder
from src.ingestion.ingestion_pipeline import IngestionPipeline


DOCUMENTS_PATH = "data/documents"
VECTORSTORE_PATH = "vectorstore"


def main():

    print("\nRebuilding RAG vector database...")

    embedder = Embedder()
    ingestion = IngestionPipeline(
    embedder=embedder
)

    vector_store = ingestion.ingest_directory(
        directory=DOCUMENTS_PATH,
        vectorstore_path=VECTORSTORE_PATH
    )

    print("\n" + "=" * 50)
    print("REBUILD COMPLETED")
    print("=" * 50)

    print(f"Total vectors: {vector_store.index.ntotal}")
    print("New chunking: 1000 characters")
    print("New overlap: 200 characters")
    print("Vector database saved successfully.")


if __name__ == "__main__":
    main()
