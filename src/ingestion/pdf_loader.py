from pathlib import Path
import re

import pymupdf


class PDFLoader:
    """Loads text from PDF documents."""

    def load(self, file_path: str) -> list[dict]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PDF not found: {path}"
            )

        documents = []

        with pymupdf.open(path) as pdf:

            for page_number, page in enumerate(
                pdf,
                start=1
            ):
                text = page.get_text()

                text = self._clean_text(text)

                if not text:
                    continue

                documents.append({
                    "text": text,
                    "metadata": {
                        "source": path.name,
                        "page": page_number,
                    },
                })

        return documents

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove repeated PDF headers and test boilerplate."""

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            return ""

        if len(lines) >= 2:
            page_header_pattern = r"^Page\s+\d+:\s+(.+)$"

            match = re.match(
                page_header_pattern,
                lines[1],
                re.IGNORECASE,
            )

            if match:
                page_title = match.group(1).strip()

                if page_title.lower() == lines[0].lower():
                    lines.pop(1)

        cleaned_lines = []

        for line in lines:

            if line.startswith(
                "This is selectable text for testing PDF extraction"
            ):
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()
