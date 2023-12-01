from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import Chroma

from .retriever_neo4j import Neo4jRetriever
from ..api_tools import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from ..graph.document_repository import DocumentRepository


class ConversationChainBuilder:
    llm: BaseChatModel
    embedder: Embeddings
    document_repository: DocumentRepository

    def __init__(self, llm: BaseChatModel, embedder: Embeddings, document_repository: DocumentRepository):
        self.llm = llm
        self.embedder = embedder
        self.document_repository = document_repository

    def get_conversation_chain(self, file_paths, messages):
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        memory.chat_memory = self._parse_chat_history(messages)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=Neo4jRetriever(document_repository=self.document_repository, embedder=self.embedder, file_paths=file_paths),
            memory=memory,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
        return conversation_chain

    @staticmethod
    def _parse_chat_history(messages):
        chat_history = ChatMessageHistory()
        for message in messages:
            if message["type"] == "HUMAN":
                chat_history.add_message(HumanMessage(
                    content=message["content"]
                ))
            if message["type"] == "AI":
                chat_history.add_message(AIMessage(
                    content=message["content"]
                ))
        return chat_history
