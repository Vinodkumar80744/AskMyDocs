
from pathlib import Path

from src.embeddings.embedder import Embedder
from src.generation.llm import GroqLLM
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.rag.pipeline import RAGPipeline
from src.retrieval.vector_store import VectorStore


DOCUMENTS_PATH = "data/documents"
VECTORSTORE_PATH = "vectorstore"

TOP_K = 5
SCORE_THRESHOLD = 0.25


def display_sources(sources):

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    if not sources:
        print("No relevant sources found.")
        return

    for index, source in enumerate(sources, start=1):
        print(
            f"\nSource {index}:"
            f"\nDocument: {source['source']}"
            f"\nPage: {source['page']}"
            f"\nChunk: {source['chunk']}"
            f"\nSimilarity Score: {source['score']:.4f}"
        )


def main():

    print("\n" + "=" * 60)
    print("MY RAG ASSISTANT")
    print("=" * 60)

    print("\nInitializing RAG application...")

    embedder = Embedder()
    llm = GroqLLM()

    vectorstore_exists = (
        Path(VECTORSTORE_PATH, "index.faiss").exists()
        and Path(VECTORSTORE_PATH, "documents.json").exists()
    )

    if vectorstore_exists:

        print("\nLoading existing vector store...")

        vector_store = VectorStore.load(VECTORSTORE_PATH)

        print(f"Vectors loaded: {vector_store.index.ntotal}")

    else:

        print("\nBuilding vector store...")

        ingestion = IngestionPipeline(embedder=embedder)

        vector_store = ingestion.ingest_directory(
            directory=DOCUMENTS_PATH,
            vectorstore_path=VECTORSTORE_PATH,
        )

    rag = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
    )

    print("\nRAG assistant is ready!")
    print("Ask questions about your PDF documents.")
    print("Type 'exit' to quit.")

    while True:

        question = input("\nAsk your question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("\nThank you for using My RAG Assistant!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:

            result = rag.ask(
                question,
                top_k=TOP_K,
                score_threshold=SCORE_THRESHOLD,
            )

            print("\n" + "=" * 60)
            print("RAG ANSWER")
            print("=" * 60)

            print(result["answer"])

            display_sources(result["sources"])

        except RuntimeError as error:
            print(f"\nRAG Error: {error}")

        except Exception as error:
            print(f"\nUnexpected error: {error}")


if __name__ == "__main__":
    main()
