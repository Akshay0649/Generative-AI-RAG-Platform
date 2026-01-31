import os
import requests

class Embedder:
    def __init__(self, model="nomic-embed-text"):
        self.model = model
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def embed(self, texts):
        vectors = []
        for t in texts:
            r = requests.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": t}
            )
            r.raise_for_status()
            vectors.append(r.json()["embedding"])
        return vectors
