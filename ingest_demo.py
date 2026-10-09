from app.llm.rag_engine import RAGEngine

rag = RAGEngine()
n = rag.load_logs("./data/logs")
print(f"[INGEST] Log directory : ./data/logs")
print(f"[INGEST] Chunks indexed: {n}")
print(f"[INGEST] Vector store  : ./data/vector_store")

# optional: quick proof that retrieval works on the new index
hits = rag.retrieve("failed login", k=3)
print(f"[CHECK] Retrieved {len(hits)} fragments for 'failed login'")
