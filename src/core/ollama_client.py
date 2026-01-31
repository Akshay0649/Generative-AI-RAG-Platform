import ollama
import os
from .base_llm import BaseLLM

class OllamaLLM(BaseLLM):
    def __init__(self, model="mistral"):
        self.model = model
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=host)

    def generate(self, prompt):
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
