from services.vectorstore.qdrant import QdrantVectorDB



def load_vectorstore() -> QdrantVectorDB:
    return QdrantVectorDB()