class RAG:
    def __init__(self, llm, embedder, vector_store):
        self.llm = llm
        self.embedder = embedder
        self.vs = vector_store

    def ask(self, question):
        q_emb = self.embedder.embed([question])[0]
        context = self.vs.search(q_emb)

        prompt = f"""
Use the following context to answer:

{context}

Question: {question}
Answer:
"""
        return self.llm.generate(prompt)
