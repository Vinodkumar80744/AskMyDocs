
import re


class TextSplitter:
    """Splits documents into overlapping, context-aware text chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def find_boundary(self, text: str, start: int, end: int) -> int:

        if end >= len(text):
            return len(text)

        minimum_boundary = start + int(
            self.chunk_size * 0.6
        )

        paragraph_boundary = text.rfind(
            "\n\n",
            minimum_boundary,
            end
        )

        if paragraph_boundary != -1:
            return paragraph_boundary

        sentence_matches = list(
            re.finditer(r'[.!?](?:["\']?)\s+', text[minimum_boundary:end])
        )

        if sentence_matches:
            return (
                minimum_boundary
                + sentence_matches[-1].end()
            )

        word_boundary = text.rfind(
            " ",
            minimum_boundary,
            end
        )

        if word_boundary != -1:
            return word_boundary

        return end

    def split(self, documents: list[dict]) -> list[dict]:

        chunks = []

        for document in documents:

            text = document["text"].strip()
            metadata = document["metadata"]

            if not text:
                continue

            start = 0
            chunk_index = 0

            while start < len(text):

                end = min(
                    start + self.chunk_size,
                    len(text)
                )

                end = self.find_boundary(
                    text,
                    start,
                    end
                )

                chunk_text = text[start:end].strip()

                if chunk_text:

                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk": chunk_index
                        }
                    })

                    chunk_index += 1

                if end >= len(text):
                    break

                next_start = max(
                    start + 1,
                    end - self.chunk_overlap
                )

                while (
                    next_start < len(text)
                    and not text[next_start].isspace()
                ):
                    next_start += 1

                start = next_start

        return chunks
