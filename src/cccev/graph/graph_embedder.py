from langchain.graphs import Neo4jGraph
from langchain.schema.embeddings import Embeddings


INDEX_EMBEDDED = "embeddedIndex"
LABEL_EMBEDDED = "Embedded"
PROPERTY_EMBED_TEXT = "embedText"
PROPERTY_EMBED_VECTOR = "embedVector"


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

    def embed_graph(self):
        nodes = self.graph.query(f"""
            MATCH (n)
            WHERE NOT '{LABEL_EMBEDDED}' in labels(n)
            RETURN n as content, labels(n) as labels, id(n) as id
        """)
        for node in nodes:
            self.embed_node(id=node["id"], label=node["labels"][0], content=node["content"])

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
                node_source=r_source.get("identifier", r_source["name"]),
                node_target=r_target.get("identifier", r_target["name"])
            )}"""

        embedding = self.embedder.embed_query(embedded_text)

        self.graph.query(f"""
                MATCH (n)
                WHERE id(n) = $node_id
                CALL db.create.setVectorProperty(n, '{PROPERTY_EMBED_VECTOR}', $embedding)
                YIELD node
                SET n.{PROPERTY_EMBED_TEXT} = $embedded_text
                SET n:{label}:{LABEL_EMBEDDED}
            """, {"node_id": id, 'embedding': embedding, 'embedded_text': embedded_text})

    @staticmethod
    def node_to_text(label: str, content: dict[str]) -> str:
        if label == "Requirement":
            return f"""{label}
             identifier: {content["identifier"]} 
             name: {content["name"]} 
             description: {content["description"]}"""
        elif label == "InformationConcept":
            return f"""{label}
            identifier: {content["identifier"]}
            name: {content["name"]}
            description: {content["description"]}"""
        elif label == "DataUnit":
            return f"""{label}
            identifier: {content["identifier"]}
            name: {content["name"]}
            description: {content["description"]}
            type: {content["type"]}"""
        elif label == "Document":
            return f"""{label}
            name: {content["name"]}
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
