from uuid import uuid4
from typing import List, Any


class BaseVectorStore:
    def upsert_documents(self, documents: List) -> None:
        """Insert or Update the Vector Database
        """

        raise NotImplemented

    def as_retriever(self) -> Any:
        """Return a retriever object for tools in RAG"""

        raise NotImplemented