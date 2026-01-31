import os
import requests
from .base_llm import BaseLLM

class OllamaLLM(BaseLLM):
    def __init__(self, model="mistral"):
        self.model = model
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, prompt):
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        r = requests.post(url, json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]
