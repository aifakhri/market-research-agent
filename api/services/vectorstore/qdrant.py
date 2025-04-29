from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from services.vectorstore.base import BaseVectorStore


from const import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    QDRANT_API_KEY,
    OPENAI_EMBEDDING
)


class QdrantVectorDB(BaseVectorStore):
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING)
        self.vector_size = len(self.embeddings.embed_query("probe"))


        if not self.client.collection_exists(collection_name=QDRANT_COLLECTION):
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=QDRANT_COLLECTION,
            embedding=self.embeddings
        )

    def upsert_documents(self, documents: List[Document]) -> None:
        """Function to update and insert to the vectorstore
        """

        self.vectorstore.add_documents(documents=documents)

    def  as_retriever(self):
        return self.vectorstore.as_retriever()