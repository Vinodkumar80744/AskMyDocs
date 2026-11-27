
import os

from dotenv import load_dotenv
from groq import Groq


class GroqLLM:
    """Wrapper around the Groq API."""

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                api_key = None

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY was not found. Configure it in your .env "
                "file or Streamlit Cloud Secrets."
            )

        self.client = Groq(api_key=api_key)

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_completion_tokens=2048,
                timeout=60
            )

            if not response.choices:
                raise RuntimeError(
                    "Groq returned no response choices."
                )

            choice = response.choices[0]
            answer = choice.message.content

            if not answer or not answer.strip():
                raise RuntimeError(
                    f"Groq returned an empty response. "
                    f"Finish reason: {choice.finish_reason}"
                )

            return answer.strip()

        except Exception as error:
            raise RuntimeError(
                f"Groq generation failed: {error}"
            ) from error
