from langchain.graphs import Neo4jGraph
from langchain.schema.embeddings import Embeddings

from .schema import INDEX_EMBEDDED, LABEL_EMBEDDED, PROPERTY_EMBED_VECTOR, PROPERTY_EMBED_TEXT, LABEL_REQUIREMENT, LABEL_INFORMATION_CONCEPT, \
    LABEL_DATA_UNIT, LABEL_DOCUMENT, EMBEDDABLE_LABELS, LABEL_DOCUMENT_CHUNK, LABEL_EMBEDDED_NODE, EMBEDDABLE_NODE_LABELS, INDEX_EMBEDDED_NODE


class GraphEmbedder:
    embedder: Embeddings = None
    graph: Neo4jGraph = None

    def __init__(self, embedder: Embeddings, graph: Neo4jGraph, **data):
        super().__init__(**data)
        self.embedder = embedder
        self.graph = graph

    def init_vector_index(self):
        self.graph.query(
            f"CALL db.index.vector.createNodeIndex('{INDEX_EMBEDDED}', '{LABEL_EMBEDDED}', '{PROPERTY_EMBED_VECTOR}', 1536, 'cosine')"
        )
        self.graph.query(
            f"CALL db.index.vector.createNodeIndex('{INDEX_EMBEDDED_NODE}', '{LABEL_EMBEDDED_NODE}', '{PROPERTY_EMBED_VECTOR}', 1536, 'cosine')"
        )

    def embed_graph(self):
        nodes = self.graph.query(f"""
            MATCH (n:{'|'.join(EMBEDDABLE_LABELS)})
            WHERE NOT '{LABEL_EMBEDDED}' in labels(n)
            RETURN n as content, labels(n) as labels, id(n) as id
        """)
        for node in nodes:
            self.embed_node(id=node["id"], label=node["labels"][0], content=node["content"])

    def embed_document(self, identifier: str):
        document_node = self.graph.query(f"""
            MATCH (doc: {LABEL_DOCUMENT} {{identifier: $identifier}})
            RETURN id(doc) as id, doc as content
        """, {"identifier": identifier})[0]

        self.embed_node(document_node["id"], LABEL_DOCUMENT, document_node["content"])

        chunk_nodes = self.graph.query(f"""
            MATCH (doc: {LABEL_DOCUMENT} {{identifier: $identifier}})--(chunk: {LABEL_DOCUMENT_CHUNK})
            RETURN id(chunk) as id, chunk as content
        """, {"identifier": identifier})

        for chunk in chunk_nodes:
            self.embed_node(chunk["id"], LABEL_DOCUMENT_CHUNK, chunk["content"])

    def embed_node(self, id: int, label: str, content: dict[str]):
        field_source = "source"
        field_type = "type"
        field_target = "target"
        relationships = self.graph.query(f"""
                MATCH (n1)-[r]->(n2)
                WHERE id(n1) = $id
                RETURN n1 as {field_source}, type(r) as {field_type}, n2 as {field_target}
            """, {"id": id})

        embedded_text = self.node_to_text(label, content)

        for relationship in relationships:
            r_type = relationship[field_type]
            r_source = relationship[field_source]
            r_target: dict[str] = relationship[field_target]
            embedded_text += f"""\n{self.relationship_to_text(
                type=r_type,
                node_source=r_source.get("identifier", r_source.get("name", "???")),
                node_target=r_target.get("identifier", r_target.get("name", "???"))
            )}"""

        embedding = self.embedder.embed_query(embedded_text)

        self.graph.query(f"""
                MATCH (n)
                WHERE id(n) = $node_id
                CALL db.create.setVectorProperty(n, '{PROPERTY_EMBED_VECTOR}', $embedding)
                YIELD node
                SET n.{PROPERTY_EMBED_TEXT} = $embedded_text
                SET n:{':'.join(self.__node_embedded_labels(label))}
            """, {"node_id": id, 'embedding': embedding, 'embedded_text': embedded_text})

    @staticmethod
    def node_to_text(label: str, content: dict[str]) -> str:
        if label == LABEL_REQUIREMENT:
            return f"""{label}
             identifier: {content["identifier"]} 
             name: {content["name"]} 
             description: {content["description"]}"""
        elif label == LABEL_INFORMATION_CONCEPT:
            return f"""{label}
            identifier: {content["identifier"]}
            name: {content["name"]}
            description: {content["description"]}"""
        elif label == LABEL_DATA_UNIT:
            return f"""{label}
            identifier: {content["identifier"]}
            name: {content["name"]}
            description: {content["description"]}
            type: {content["type"]}"""
        elif label == LABEL_DOCUMENT:
            return f"""{label}
            identifier: {content["identifier"]}
            name: {content["name"]}
            """
        elif label == LABEL_DOCUMENT_CHUNK:
            return f"""{label}
            identifier: {content["identifier"]}
            content: {content[PROPERTY_EMBED_TEXT]}
            """
        else:
            print(f"Unsupported label {label}")

    @staticmethod
    def relationship_to_text(type: str, node_source: str, node_target: str) -> str:
        if type == 'HAS_CONCEPT':
            type_txt = "USES_INFORMATION"
        else:
            type_txt = type
        return f"{node_source} {type_txt} {node_target}"

    @staticmethod
    def __node_embedded_labels(label: str) -> [str]:
        base_labels = [label, LABEL_EMBEDDED]
        if label in EMBEDDABLE_NODE_LABELS:
            base_labels.append(LABEL_EMBEDDED_NODE)
        return base_labels
