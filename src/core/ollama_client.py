import ollama
from .base_llm import BaseLLM

class OllamaLLM(BaseLLM):
    def __init__(self, model="mistral"):
        self.model = model

    def generate(self, prompt):
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
