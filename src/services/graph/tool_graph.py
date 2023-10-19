from langchain import LLMChain
from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT
from langchain.chat_models.base import BaseChatModel
from langchain.graphs import Neo4jGraph
from langchain.schema.embeddings import Embeddings
from langchain.tools import BaseTool

from .schema import INDEX_EMBEDDED_NODE, PROPERTY_EMBED_TEXT


class QuestionGraphTool(BaseTool):
    name = "QuestionGraph"
    description = "useful for when you need to retrieve context from a graph via vectorized search when a structured data extraction didn't work. The retrieved context might be incomplete"

    embedder: Embeddings = None
    graph: Neo4jGraph = None

    def __init__(self, embedder: Embeddings, graph: Neo4jGraph, **data):
        super().__init__(**data)
        self.embedder = embedder
        self.graph = graph

    def _run(self, question: str):
        question_embedding = self.embedder.embed_query(question)
        answer_context_parts = self.graph.query(f"""
            CALL db.index.vector.queryNodes('{INDEX_EMBEDDED_NODE}', 50, $question_embedding) 
            YIELD node, score
            RETURN node.{PROPERTY_EMBED_TEXT} AS text, score
        """, {"question_embedding": question_embedding})

        answer_context = '\n'.join(map(lambda context_part: context_part["text"], answer_context_parts))
        return f"Here are the most relevant data to your question. Beware as it might be incomplete:\n{answer_context}"


class QueryGraphTool(BaseTool):
    name = "QueryGraph"
    description = "useful for when you need to retrieve structured data from a graph. The resulting data is always an extraction of the graph that anwsers the question"

    llm: BaseChatModel = None
    graph: Neo4jGraph = None

    def __init__(self, llm: BaseChatModel, graph: Neo4jGraph, **data):
        super().__init__(**data)
        self.llm = llm
        self.graph = graph

    def _run(self, question: str):
        chain = LLMChain(llm=self.llm, prompt=CYPHER_GENERATION_PROMPT)
        prompt = f"""Retrieve as much details as you can to answer the following question, but do not ever return the vector fields: {question}"""
        query = chain.run(
            {"question": prompt, "schema": self.graph.get_schema}
        )
        print(query)
        result = self.graph.query(query)
        return f"Here are the structured data that answer the question: {result}"
