from typing import List

from langchain.callbacks.manager import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever, Document
from langchain.schema.embeddings import Embeddings

from ..graph.document_repository import DocumentRepository


class Neo4jRetriever(BaseRetriever):
    document_repository: DocumentRepository
    embedder: Embeddings
    file_paths: List[str]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        vector = self.embedder.embed_query(query)
        chunks = self.document_repository.find_similar_chunks(self.file_paths, vector, 5)
        return list(map(lambda chunk: Document(page_content=chunk), chunks))

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
        raise NotImplementedError()
