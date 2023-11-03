import uuid

from langchain.graphs import Neo4jGraph
from langchain.text_splitter import CharacterTextSplitter, TextSplitter, TokenTextSplitter

from .schema import LABEL_DOCUMENT_CHUNK, PROPERTY_EMBED_TEXT, LABEL_DOCUMENT

# DEFAULT_TEXT_SPLITTER = CharacterTextSplitter(
#     separator="",
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len
# )
DEFAULT_TEXT_SPLITTER = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


class DocumentRepository:
    graph: Neo4jGraph
    text_splitter: TextSplitter

    def __init__(self, graph: Neo4jGraph, text_splitter: TextSplitter = DEFAULT_TEXT_SPLITTER, **data):
        super().__init__(**data)
        self.graph = graph
        self.text_splitter = text_splitter

    def save(self, name: str, content: str, path: str, metadata: dict[str], trace: [str] = []) -> str:
        chunks = self.text_splitter.split_text(content)

        doc_identifier = self.graph.query(f"""
            MERGE (doc:{LABEL_DOCUMENT} {{path: $path}})
            ON CREATE
                SET doc.identifier = 'kb_{str(uuid.uuid4()).replace('-', '_')}'
            WITH doc
            SET doc.name = $name
            RETURN doc.identifier as identifier
        """, {"path": path, "name": name})[0]["identifier"]

        deleted_count = self.graph.query(f"""
            MATCH (doc:{LABEL_DOCUMENT} {{path: $path}})--(chunk: {LABEL_DOCUMENT_CHUNK})
            DETACH DELETE chunk
            RETURN COUNT(chunk) as deleted_count
        """, {"path": path})[0]["deleted_count"]
        
        if deleted_count > 0:
            trace.append(f"{deleted_count} existing chunks deleted")

        query = f"MATCH ({doc_identifier}:{LABEL_DOCUMENT} {{identifier: '{doc_identifier}'}})"
        query_params = {}
        counter = 0

        for chunk in chunks:
            chunk_identifier = f"{doc_identifier}_{counter}"
            counter += 1
            query += self.__prepare_chunk_cypher(doc_identifier, chunk_identifier, metadata)
            query_params[f"text_{chunk_identifier}"] = chunk
            for key, value in metadata.items():
                query_params[f"{key}_{chunk_identifier}"] = value

        self.graph.query(query, query_params)
        trace.append(str(len(chunks)) + " chunks created")
        return doc_identifier

    def find_similar_chunks(self, file_paths: [str], vector, limit: int) -> [str]:
        chunks = self.graph.query("""
            MATCH (doc:Document)<--(chunk:DocumentChunk)
            WHERE doc.path in $file_paths
            WITH chunk, gds.similarity.cosine($vector, chunk.embedVector) as score
            RETURN chunk.identifier as identifier, chunk.embedText as text, score
            ORDER BY score DESC
            LIMIT $limit
        """, {"vector": vector, "limit": limit, "file_paths": file_paths})

        return list(map(lambda chunk: chunk["text"], chunks))

    @staticmethod
    def __prepare_chunk_cypher(doc_identifier: str, chunk_identifier: str, metadata: dict[str]):
        metadata_arr = []
        for key in metadata.keys():
            metadata_arr.append(f"{key}: ${key}_{chunk_identifier},")
        metadata_str = "\n".join(metadata_arr)

        return f"""CREATE ({chunk_identifier}:{LABEL_DOCUMENT_CHUNK} {{
            identifier: "{chunk_identifier}",
            {metadata_str}
            {PROPERTY_EMBED_TEXT}: $text_{chunk_identifier}
        }})
        CREATE ({chunk_identifier})-[:EXTRACTED_FROM]->({doc_identifier})
        """
