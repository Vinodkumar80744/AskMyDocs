
import re

from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.generation.llm import GroqLLM


class RAGPipeline:
    """Coordinates retrieval and generation."""

    def __init__(self, embedder, vector_store, llm):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def get_answer_instructions(self, question: str) -> str:
        question_lower = question.lower()

        word_match = re.search(
            r"\b(?:in|within|around|approximately)\s+(\d+)\s+words?\b",
            question_lower
        )

        if word_match:
            word_count = int(word_match.group(1))
            return (
                f"Write approximately {word_count} words. "
                "Stay close to the requested length."
            )

        if any(
            word in question_lower
            for word in ["briefly", "short answer", "in short", "one line"]
        ):
            return "Provide a short and direct answer."

        if any(
            word in question_lower
            for word in [
                "explain",
                "describe",
                "elaborate",
                "in detail",
                "how does",
                "why does"
            ]
        ):
            return (
                "Provide a detailed and well-organized explanation. "
                "Cover all relevant information supported by the passages."
            )

        return (
            "Provide a clear and informative answer. "
            "Include relevant details from the supplied passages."
        )

    def ask(
        self,
        question: str,
        top_k: int = 5,
        score_threshold: float = 0.25,
    ) -> dict:

        query_embedding = self.embedder.encode([question])

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        results = [
            result
            for result in results
            if result["score"] >= score_threshold
        ]

        if not results:
            return {
                "answer": (
                    "I could not find the answer in the "
                    "provided documents."
                ),
                "sources": [],
            }

        context_parts = []

        for index, result in enumerate(results, start=1):
            document = result["document"]
            metadata = document["metadata"]

            context_parts.append(
                f"""
DOCUMENT PASSAGE {index}
Source: {metadata['source']}
Page: {metadata['page']}

Content:
{document['text']}
"""
            )

        context = "\n\n".join(context_parts)

        print("\n" + "=" * 60)
        print("RETRIEVED CONTEXT")
        print("=" * 60)
        print(context)
        print("=" * 60)

        answer_instructions = self.get_answer_instructions(question)

        prompt = f"""
You are a document-grounded question-answering assistant.

Your task is to answer the user's question using ONLY the supplied
document passages.

STRICT GROUNDING RULES:
1. Every factual statement must be supported by the passages.
2. Do not use external knowledge, even if you already know the answer.
3. Do not add definitions, explanations, examples, or details that
   are not explicitly supported by the passages.
4. Never guess or invent missing information.
5. If only part of the question is answered by the passages,
   answer that part and clearly identify what is missing.
6. If the passages contain no relevant information, say:
   "I could not find the answer in the provided documents."
7. Do not treat a topic being mentioned as evidence that its
   definition or explanation is available.

ANSWER LENGTH AND STYLE:
{answer_instructions}

CONTEXT UTILIZATION:
1. Review every supplied passage.
2. Combine relevant information from multiple passages.
3. Include only passages that contribute to answering the question.
4. Do not force unrelated information into the answer.
5. Avoid repeating the same information.
6. Organize detailed answers with headings and bullet points.
7. Use examples only when explicitly supported by the passages.

FORMATTING:
- Use Markdown headings where appropriate.
- Use bullet points for lists.
- Use tables when they improve clarity.
- Use properly formatted code blocks for programming examples.

DOCUMENT PASSAGES:
------------------------------
{context}
------------------------------

USER QUESTION:
{question}

Before responding, verify that every factual statement in your answer
is supported by the supplied document passages.

Write the final answer now.
"""

        answer = self.llm.generate(prompt)

        sources = []

        for result in results:
            document = result["document"]
            metadata = document["metadata"]

            sources.append({
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk": metadata["chunk"],
                "score": result["score"],
            })

        return {
            "answer": answer,
            "sources": sources,
        }
