from src.core.ollama_client import OllamaLLM
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.generate import RAG

# Sample data
texts = [
    "Ollama is a local LLM runner.",
    "FAISS is used for vector similarity search.",
    "RAG combines retrieval with generation."
]

embedder = Embedder()
embeddings = embedder.embed(texts)

vs = VectorStore(dim=len(embeddings[0]))
vs.add(embeddings, texts)

llm = OllamaLLM()
rag = RAG(llm, embedder, vs)

while True:
    q = input("Ask> ")
    print(rag.ask(q))
