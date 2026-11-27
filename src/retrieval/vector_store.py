import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """Stores embeddings and performs similarity search."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

    def add(
        self,
        embeddings: np.ndarray,
        documents: list[dict]
    ) -> None:

        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match number of documents"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> list[dict]:

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            results.append({
                "score": float(score),
                "document": self.documents[index]
            })

        return results

    def save(self, directory: str) -> None:
        """Save the FAISS index and document metadata to disk."""

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(path / "index.faiss")
        )

        with open(
            path / "documents.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                self.documents,
                file,
                ensure_ascii=False,
                indent=2
            )

    @classmethod
    def load(cls, directory: str):
        """Load a previously saved vector store."""

        path = Path(directory)

        index_path = path / "index.faiss"
        documents_path = path / "documents.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not documents_path.exists():
            raise FileNotFoundError(
                f"Documents metadata not found: {documents_path}"
            )

        index = faiss.read_index(
            str(index_path)
        )

        with open(
            documents_path,
            "r",
            encoding="utf-8"
        ) as file:
            documents = json.load(file)

        store = cls(index.d)
        store.index = index
        store.documents = documents

        return store